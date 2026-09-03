# CPU vs GPU Equivalence Characterization — Full Corpus (2026-08-04)

**Status**: characterization/evidence only. `core/vision_embeddings.py`
was NOT modified. No production code path was changed. GPU code exists
only as experimental, local functions in `benchmark/gpu_cpu_equivalence_
phaseB_gpu.py` — not promoted anywhere.

## Why this exists

Forced by a real incident this session: a mistaken diagnostic write
destroyed the corpus's real `data/baseline_embeddings.json`, making full
recomputation unavoidable. Rather than just redoing the same CPU-only
capture, this was used to answer a standing architectural question:
does GPU execution produce operationally identical results to CPU
execution for the vision-tower stage? Earlier research this session
found `core/vision_embeddings.py` never calls `.cuda()` anywhere — CPU
execution by default despite CUDA being available — and flagged that
GPU migration is not automatically "zero behavioral change," since CPU
(MKL/oneDNN) and GPU (cuDNN/cuBLAS) don't guarantee bit-identical
floating-point results. This is the full-corpus measurement of that
question, not a sample.

## [A] Source evidence — measured values only

### Phase A — CPU regeneration
Threaded ×4, 4 intra-op threads/worker (the config `benchmark/baseline_
embeddings_concurrency_experiment.py` measured as best on a 16-image
sample this same session) — same underlying computation as production
(`core/vision_embeddings.py`'s own `build_model_and_transform()`/
`embed_pooled()`, unmodified), scheduling-only change.

- **1751 images, 1290.5s (21.5 min), 1.357 img/s**
- CPU utilization: mean 56.0%, max 88.0%
- torch 2.13.0+cu130, timm 1.0.28
- Output: `data/outputs/gpu_cpu_equivalence/baseline_embeddings_cpu.json`

### Phase B — GPU regeneration
Experimental, local-only device-parameterized copies of the same two
functions (`.to("cuda")` added), sequential (no threading tested for
GPU in this pass).

- **1751 images, 750.4s (12.5 min), 2.334 img/s**
- Peak VRAM: 429.9 MB
- GPU: NVIDIA GeForce RTX 5060 Ti, CUDA 13.0, torch 2.13.0+cu130
- Output: `data/outputs/gpu_cpu_equivalence/baseline_embeddings_gpu.json`

### Phase C — Full corpus embedding-vector equivalence
1751 images, 14008 (image × encoder) pairs (1751 × 8).

| metric | min | max | mean | median | stdev |
|---|---|---|---|---|---|
| cosine similarity | 0.99999987 | 0.99999999999977 | 0.99999999774 | 0.99999999999 | 7.01e-09 |
| L2 distance | 1.08e-05 | 8.36e-03 | 4.41e-04 | 6.23e-05 | 1.03e-03 |
| max abs element diff | 1.00e-06 | 3.13e-03 | 7.57e-05 | 9.00e-06 | 1.97e-04 |

Outliers (cosine similarity < 0.9999): **0 / 14008.**

### Phase D — Operational equivalence (independent per-set replay)
Each embedding set used independently (its own vectors as both query
and reference pool, via the unmodified `core.decision_engine.compute_
tower_consensus_for_image()`/`build_reference_embeddings()`).

- Images compared: 1751. Embedding pairs: 14008.
- Nearest-neighbour (per-encoder winner) changes: **0 / 14008**
- Tower vote changes: **0 / 14008** (same measurement)
- Consensus category changes: **0 / 1751**
- Vote-count distribution changes: **0 / 1751**
- Overall winning-bucket changes: **0 / 1751**
- Margin changes (abs delta): min 0.0, max 2.0e-4, mean 2.68e-06, median 0.0, stdev 1.64e-05
- **Images with any change at all: 0 / 1751.**

### Phase E — Downstream equivalence (disagreement-audit reproduction)
CPU-based and GPU-based reproductions of the tower-consensus-vs-Gemma
disagreement classification, computed independently, then compared to
each other AND to the original, previously-reported audit
(`data/outputs/tower_consensus_audit/20260804T010436Z/disagreements.csv`,
692 rows — this file survived; the embeddings that produced it did not).

- CPU-based: 654 total disagreements, distribution `{unanimous: 94, majority: 411, split: 144, complete_disagreement: 5}`
- GPU-based: 654 total disagreements, distribution `{unanimous: 94, majority: 411, split: 144, complete_disagreement: 5}`
- **CPU vs GPU disagreement sets: identical. 0 images differ (0 CPU-only, 0 GPU-only).**
- CPU/GPU (654/94) vs originally-reported (692/47): **does not match.**
  - 58 images: disagreed originally, no longer disagree
  - 20 images: newly disagree, did not originally
  - 634 images: disagree in both original and regenerated sets
  - Of those 634: **201 changed `consensus_category`** (e.g. majority→split, split→majority, majority→unanimous)

### Phase F — Performance table

| Configuration | Time | Images/sec | Speedup vs. sequential-CPU-16-thread baseline (16-image sample) |
|---|---|---|---|
| Sequential CPU (16 intra-op threads, default) | — (not re-run full-corpus; 26.2s/16 images in the smaller sample) | 0.612 (sample) | 1.00x |
| Threaded CPU ×4, 4 threads/worker (Phase A, full corpus) | 1290.5s | 1.357 | ~1.31x (matches sample-measured ratio) |
| Sequential GPU (Phase B, full corpus) | 750.4s | 2.334 | ~1.76x vs sequential CPU; **1.72x vs threaded CPU (Phase A)** |

## [B] Interpretation — clearly separated from the evidence above

**Numerical equivalence**: CPU and GPU produce embeddings that are, for
practical purposes, numerically indistinguishable on this hardware and
model set — cosine similarity effectively 1.0 across all 14008 pairs,
zero outliers even at a strict 0.9999 threshold. This is tighter than
the earlier 16-image sample suggested was even likely, and rules out
"GPU precision drift" as a meaningful concern for this specific pipeline.

**Operational equivalence**: with zero changes across every measured
decision point (nearest-neighbour, vote, consensus category, vote
counts, overall winner) across all 1751 images, CPU and GPU are
operationally interchangeable for the tower-consensus computation *as
long as both are evaluating the same input pixels*.

**The 692/47 vs 654/94 gap is not a CPU/GPU phenomenon** — it can't be,
given CPU and GPU agree with each other perfectly. The actual cause,
confirmed directly: every image in this corpus is at `current_stage=5`
(already classified), meaning Stage 3 preprocessing (deskew +
autocontrast) had already been applied to every working copy before
Phase A/B ever measured them. The original 692/47 was captured on true
pre-preprocessing pixels (Stage 1 ran before Stage 3, correctly, during
the original fresh-corpus pipeline run). Phase A/B measured
post-preprocessing pixels instead, because the working copies had
already been modified in place and there was no surviving
pre-preprocessing copy of them to re-read. This is the same class of
mistake `docs/REFERENCE_PIPELINE_V1.md` already documented once this
project's history — a "baseline" captured against already-processed
pixels — recreated here in miniature, not a new failure mode.

**Performance**: GPU is faster (1.76x vs sequential CPU, 1.72x vs the
already-threaded CPU config) but not dramatically so for this model set
at batch size 1 — consistent with these being relatively small,
CPU-efficient models rather than a workload GPU batching would
transform.

## [C] Conclusions — answering only the questions asked

**Does GPU execution produce materially different embeddings?**
No. Cosine similarity ≥0.99999987 across all 14008 (image, encoder)
pairs in the full corpus, zero outliers at a 0.9999 threshold.

**Does GPU execution change any downstream tower decisions?**
No. Zero changes across nearest-neighbour, vote, consensus category,
vote counts, and overall winning bucket, for all 1751 images, when CPU
and GPU are each evaluated against their own internally-consistent
reference set.

**Does GPU execution change any downstream audit results?**
No, relative to CPU — the CPU-based and GPU-based disagreement audits
are identical (654/654, same category distribution, 0 images differ).
**Yes, relative to the original 692/47** — but this difference is
caused by a pixel-state mismatch (post- vs pre-preprocessing), not by
GPU vs CPU. Both regenerated sets agree with each other and both differ
from the original for the same, identified, non-GPU reason.

**Is CPU vs GPU effectively interchangeable for this pipeline based on
the measured corpus?**
Yes, stated precisely: interchangeable for producing the embeddings and
every downstream decision measured, GIVEN the same input pixels. This
characterization does not speak to whether either execution mode is
"correct" relative to a true pre-preprocessing baseline — that question
requires re-capturing from pixels that predate Stage 3, which this pass
did not have access to (see the pixel-state caveat above), and is a
separate question from CPU/GPU equivalence.

No differences observed between CPU and GPU, stated precisely as
requested: 0 embedding outliers, 0 decision changes, 0 audit
differences, across the entire 1751-image corpus, not a sample.
