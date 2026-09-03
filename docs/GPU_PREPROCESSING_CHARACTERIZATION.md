# GPU Preprocessing Characterization (2026-08-04)

**Status**: research only. `core/vision_embeddings.py` was NOT modified.
No production code path changed. All GPU preprocessing code exists only
as local, experimental functions in `benchmark/gpu_preprocessing_
stage34_prototypes.py` — not promoted anywhere.

## Why this exists

The images-in-flight sweep (same session) found GPU utilization never
exceeded ~40% while CPU utilization reached 90-100% during the same
runs — CPU-side preprocessing, not GPU inference, is the actual
throughput ceiling. This characterizes whether any of that CPU
preprocessing work can safely move to GPU without changing embedding
outputs.

## Stage 1 — Characterize current pipeline (measured, not assumed)

Every one of the 8 `QUALIFIED_ENCODERS`' real `timm`-built transforms
was directly inspected (not assumed from typical `timm` defaults). All
8 are structurally identical 4-step pipelines: `Resize → CenterCrop →
MaybeToTensor → Normalize`, differing only in target size/interpolation
and per-channel mean/std. Six of the eight land on the same 224×224
final crop size, but their *pre-crop resize target differs* (256 vs
248) — confirmed directly, so naive resize-sharing across those six
encoders is not safe even though their final tensor shape matches.

Per-operation timing (30-image sample, 8 encoders, `N_TIMING_REPS=5`
repeats averaged per image):

| Operation | Device | Library | Per-image | Per-encoder | Mean cost |
|---|---|---|---|---|---|
| Decode (`Image.open`+`convert("RGB")`) | CPU | PIL | yes (once, shared) | no | 13.05ms/image |
| Resize | CPU | torchvision (via timm) | no | yes | **16.7–21.1ms/pair** |
| CenterCrop | CPU | torchvision | no | yes | 0.03–0.26ms/pair |
| MaybeToTensor | CPU | torchvision | no | yes | 0.22–0.69ms/pair |
| Normalize | CPU | torchvision | no | yes | 0.09–0.20ms/pair |

Total decode work (30 images, once each): 391.5ms. Total transform work
(30 images × 8 encoders): 4341.8ms. Ratio: **11.1 : 1** — the
per-encoder-repeated transform work dominates the once-per-image decode
by an order of magnitude.

**Resize alone is 95%+ of the entire per-(image,encoder) transform
cost.** CenterCrop, MaybeToTensor, and Normalize combined are under
1.2ms — comparatively trivial regardless of which device they run on.

## Stage 2 — GPU candidates (evidence-based priority, not the suggested order)

The suggested testing order was normalize → resize → type-conversion.
Stage 1's measured data does not support that priority — Resize is
overwhelmingly the dominant cost and the only operation whose device
could plausibly matter for throughput; Normalize/ToTensor/CenterCrop
are cheap enough that moving them alone cannot meaningfully move the
needle regardless of numerical outcome. Prioritized by measured cost,
not the suggested sequence, per this pass's own "evidence over
intuition" instruction:

| Candidate | Expected CPU savings | Expected GPU cost | Implementation complexity | Numerical risk |
|---|---|---|---|---|
| Resize | Large (~95% of transform time) | Requires pipeline reorder (tensorize before resize, not after) | High — different libraries, different bicubic kernels | **High — confirmed real, not assumed (see below)** |
| CenterCrop | Negligible standalone | Trivial (tensor slicing) | Low, but structurally coupled to Resize's output location | Low (pure indexing, no interpolation) |
| Normalize | Negligible standalone | Trivial (elementwise arithmetic) | Low | Very low (no interpolation, exact arithmetic) |
| ToTensor (uint8→float, HWC→CHW) | Negligible standalone | Trivial | Low, but timing depends on where in the pipeline it happens | Very low |

**Pre-check before prototyping**: does GPU bicubic resize
(`torch.nn.functional.interpolate`) even match PIL's CPU bicubic resize
numerically? Measured directly on one real image before building
anything further: mean abs pixel difference **0.31**, max **12.16**
(0-255 space) — a real, measured difference, not floating-point noise.
PIL and PyTorch implement bicubic interpolation differently; this is
not the same computation on different hardware (unlike the CPU-vs-GPU
*inference* equivalence found earlier this session, which was the same
math on different hardware and matched almost exactly).

## Stage 3 — Prototypes (each isolated, never combined)

Two configurations, each compared independently against the exact
current production pipeline (30 images × 8 encoders = 240 pairs):

- **`gpu_normalize`**: resize/crop/totensor stay CPU (PIL/torchvision,
  unchanged); only normalize moves to GPU. Inference also runs on GPU
  in this config, deliberately - GPU vs CPU inference was already
  independently validated as equivalent earlier this session (cosine
  similarity ≥0.99999987 across the full corpus, zero decision
  changes), so combining "GPU normalize" with "GPU inference" here
  still isolates normalize as the only *new* variable, rather than
  conflating two untested changes.
- **`gpu_resize_crop`**: decode stays CPU; the full-size image is
  tensorized on CPU (unavoidable - needed to transfer to GPU at all),
  then resize+crop happen on GPU; normalize and inference are moved
  back to CPU afterward, exactly matching the baseline's own device
  placement for those steps - isolates resize+crop as the only
  variable.

## Stage 4 — Numerical equivalence (measured)

| Config | Cosine sim (min/mean/max) | L2 distance (min/mean/max) | Max abs elem diff (min/mean/max) | Pairs failing 0.9999 threshold |
|---|---|---|---|---|
| `gpu_normalize` | 0.99999995 / 1.00000000 / 1.00000000 | 0.000008 / 0.000452 / 0.005566 | 0.000002 / 0.000074 / 0.001109 | **0 / 240** |
| `gpu_resize_crop` | 0.33575560 / 0.92254970 / 0.99995408 | 0.157165 / 6.862680 / 50.587217 | 0.039894 / 1.005344 / 7.146369 | **238 / 240** |

`gpu_normalize` passes cleanly. `gpu_resize_crop` fails decisively -
mean cosine similarity 0.923 (not close to 1.0), minimum 0.336, nearly
every pair (238/240) exceeds even a loose 0.9999 threshold. The
pixel-level difference found in the Stage 2 pre-check does **not** get
absorbed by the network - it propagates into substantially different
embeddings.

**Operational spot-check** (10 images, dinov2 encoder, nearest-bucket
prediction against the CPU reference set): 0/10 flipped. Not treated as
evidence of safety - a 10-image, 1-encoder sample is far too small to
establish operational equivalence given embedding-level divergence this
large, and the earlier ensemble characterization (same session) already
found some encoders operate with very thin margins corpus-wide; a
larger sample would very plausibly surface flips. `gpu_resize_crop` is
rejected on the embedding-level evidence alone - the operational check
was not pursued further given it already fails decisively.

## Stage 5 — Performance (measured)

| Configuration | Elapsed (240 pairs) | Pairs/sec | Speedup vs. baseline |
|---|---|---|---|
| baseline (all CPU) | 30.0s | 8.00 | 1.00x |
| `gpu_normalize` (+ GPU inference) | 8.9s | 27.09 | 3.39x |
| `gpu_resize_crop` | 31.7s | 7.56 | **0.95x (no gain - slightly slower)** |

`gpu_resize_crop` shows no performance benefit even setting correctness
aside - the CPU↔GPU transfer overhead for a single image at a time
outweighs any resize speedup, since inference stays on CPU in this
isolated config and there's no batching to amortize the transfer cost.

## Stage 6 — Bottleneck analysis

Has CPU utilization decreased? For `gpu_normalize`: yes, indirectly -
but the measured 3.39x speedup is the combined effect of GPU normalize
(negligible on its own, per Stage 1's cost table) plus GPU inference
(already validated separately). Normalize's own isolated contribution
to the CPU bottleneck is not meaningfully different from zero, because
its own standalone CPU cost was already negligible (0.09-0.20ms out of
17-22ms per pair) - there was very little CPU load to remove.

Has GPU utilization increased? Measured directly in a follow-up pass
(`benchmark/gpu_normalize_flight_sweep.py` +
`benchmark/gpu_normalize_full_corpus.py`, 2026-08-04): **no.** With
inference already GPU-resident (the images-in-flight pipeline's own
baseline), adding GPU normalize on top left GPU utilization unchanged
(54.5% -> 54.4% mean, full corpus, k=4) and CPU utilization unchanged
(98.9% -> 99.1% mean, still pinned near 100%). Throughput was flat:
0.995x on the full 1750-image corpus, and never exceeded 1.02x (noise)
at any of 6 tested flight levels (k=1,2,3,4,6,8) on a 60-image sample,
with a real 0.94x regression at k=4 and k=6. Correctness held perfectly
at every scale (0/480 pairs below threshold on the sample at every k;
0/14000 on the full corpus, cosine similarity exactly 1.0 min/mean/max).

This sharpens the Stage 5 finding above: the isolated prototype's 3.39x
speedup for `gpu_normalize` is now confirmed to be **entirely**
attributable to also moving inference to GPU (already independently
validated earlier this session), not partially - normalize's own
contribution, once inference is already on GPU, is measured at
essentially zero, consistent with its negligible standalone CPU cost
found in Stage 1 (0.09-0.20ms of the 17-22ms per-pair total).

Has throughput increased? Yes for `gpu_normalize` (3.39x, but see the
attribution caveat above), no for `gpu_resize_crop` (0.95x).

Has the bottleneck moved? **No, and it cannot with the operations
tested here.** The dominant cost (Resize, ~95% of transform time) is
the one operation that failed numerical equivalence decisively. The
operations that passed equivalence cleanly (Normalize, and by
extension the untested CenterCrop/ToTensor) are too cheap individually
to relocate the bottleneck even if moved. The CPU preprocessing
ceiling identified by the images-in-flight sweep remains CPU-bound
after this pass, because its actual cause (Resize) cannot safely move.

## [A] Source Evidence

- Resize is 95%+ of per-(image,encoder) transform cost (16.7-21.1ms vs.
  a combined <1.2ms for CenterCrop+ToTensor+Normalize), measured across
  30 images × 8 encoders.
- Decode (PIL, once per image) totals 391.5ms for 30 images; transform
  work (8× per image) totals 4341.8ms for the same 30 images - an
  11.1:1 ratio.
- PIL-CPU bicubic resize and PyTorch-GPU bicubic resize (with
  `antialias=True`, matching torchvision's own default) differ by mean
  abs pixel value 0.31, max 12.16, on a real corpus image (0-255 scale).
- `gpu_normalize` (+ GPU inference): cosine similarity min 0.99999995,
  mean 1.00000000; 0/240 pairs fail a 0.9999 threshold; 3.39x speedup
  (8.00 → 27.09 pairs/sec).
- `gpu_resize_crop`: cosine similarity min 0.33575560, mean 0.92254970;
  238/240 pairs fail the same threshold; 0.95x speedup (no gain).
- 10-image/1-encoder spot-check of `gpu_resize_crop`'s effect on
  nearest-bucket prediction: 0/10 flipped.

## [B] Interpretation

The CPU preprocessing bottleneck identified by the images-in-flight
sweep is real and is caused overwhelmingly by one operation (Resize),
not spread evenly across the pipeline. That concentration is actually
informative: it means the fix, if one exists, has to specifically
address Resize - moving the other three operations would not
meaningfully help even if perfectly safe, because they were never a
meaningful fraction of the cost.

The Resize numerical divergence is not surprising in hindsight -
PIL and PyTorch's bicubic implementations are independently-written
algorithms, not the same math run on different hardware, unlike the
CPU-vs-GPU inference comparison (same model weights, same operations,
different hardware) that produced near-bit-identical results earlier
this session. Conflating "GPU execution" as a single risk category
across both cases would have been a mistake - inference-device
equivalence and preprocessing-algorithm equivalence are different
questions with different answers here, and treating them as
interchangeable is exactly the kind of assumption this project's own
discipline (measure, don't infer) exists to catch.

The `gpu_normalize` result is real but should not be over-read as "GPU
preprocessing helps" - the speedup is attributable to also moving
inference to GPU (already known and validated), not to normalize's own
negligible cost. Isolating normalize's true individual contribution
would require a config with GPU normalize + CPU inference, which was
not built in this pass.

## [C] Conclusions — answering only the questions asked

**Which preprocessing operations can safely execute on GPU?**
Normalize, based on measured evidence (near-perfect embedding
equivalence). CenterCrop and ToTensor were not independently prototyped
in this pass, but their negligible individual cost (Stage 1) and the
absence of any interpolation/algorithm choice in either operation make
them very likely low-risk by the same reasoning that made Normalize
safe - not confirmed by direct measurement, stated as inference, not
evidence.

**Which operations should remain on CPU?**
Resize. Confirmed unsafe (238/240 pairs fail equivalence, mean cosine
similarity 0.923) and confirmed to offer no performance benefit in
isolation (0.95x). Both reasons are independently sufficient to reject
it; neither depends on the other.

**Did moving preprocessing increase end-to-end throughput?**
No. A direct follow-up test isolated normalize's contribution with
inference already GPU-resident (removing the earlier confound) and
found flat throughput at every tested concurrency level and at full
corpus scale (0.995x on 1750 images; never exceeded 1.02x noise on a
60-image sample across 6 flight levels, with real regression at two of
them). Normalize's apparent speedup in the original isolated prototype
is now confirmed attributable entirely to moving inference to GPU, not
to normalize itself. For the operation that actually dominates the cost
(Resize), no throughput gain was observed either, and it failed
correctness regardless.

**Did the bottleneck move?**
No. The CPU preprocessing ceiling identified by the images-in-flight
sweep remains CPU-bound, because the operation responsible for it
(Resize) cannot safely move to GPU with the approach tested here.

**Is this worth promoting into the production pipeline?**
No, stated explicitly per this pass's own instruction to say so if
true: no measurable improvement to the actual bottleneck exists from
what was tested. Moving Normalize alone is numerically safe but
addresses a cost that was already negligible. Moving Resize - the
operation that would matter - fails numerical equivalence decisively
and shows no performance benefit even before considering correctness.
Nothing here should be promoted or implemented in production code, per
this pass's own scope.
