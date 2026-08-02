# Benchmark 2 — Metadata Layer Qualification

**Status**: design + first real measurement pass. Renamed per Jon's
suggestion from "Multi-Tower Metadata Qualification" — the shift in
name matters: this isn't evaluating models anymore (Benchmark 1 did
that), it's evaluating *what the framework should permanently know*
about every image.

**Relationship to Benchmark 1**: everything in this document is built
on real, measured findings from `docs/VISION_IR_RESEARCH.md`'s 9
qualification rounds and 3 localization experiments — not
re-speculated from scratch. Where a claim below is a new proposal
rather than something already validated, it's labeled **PROPOSED**,
not **MEASURED**, and the distinction is load-bearing throughout this
document, same discipline as Benchmark 1.

---

## 1. Benchmark design

**Question**: for each of the 8 encoders already qualified in
Benchmark 1, what metadata can it *reliably* produce, does that
metadata overlap another encoder's, and is any of it operationally
useful elsewhere in the pipeline? This is explicitly not a "which
encoder wins" exercise — Benchmark 1 already established that
different training objectives produce genuinely different, sometimes
complementary behavior (the inversion-sensitivity hypothesis chain is
the clearest example: DINOv2/EVA-02/MAE react to polarity in ways
ConvNeXt/Swin/SigLIP/BEiT don't at all).

**Method**: rather than run new qualification rounds, this benchmark
synthesizes the *already-measured* per-encoder behavior from
Benchmark 1 into a capability matrix (§3), validates the practical
cost assumption with real combined measurement (§4), and designs the
persistent artifact this behavior should feed (§2, §5, §6).

---

## 2. Metadata schema proposal

Versioned, per Jon's example, extended with the two-tier discipline
already established in the original Architecture D research
(`docs/VISION_IR_RESEARCH.md` §1/§26): every field is either a
**raw/derived measurement** (numbers, always retained) or a
**calibrated category** (bins, retunable without re-computing), never
collapsed into one undifferentiated blob. Fields also tag which
**consumption channel** they're meant for — machine-consumed (a
program branches on it directly) or text-hint (rendered into a
prompt) — per the distinction established earlier in this research,
since that determines which validation discipline applies (§20/§21).

```json
{
  "metadata_version": 1,
  "image_id": "oocihm.lac_reel_c10264.767.jpg",
  "generated_at": "2026-08-01T00:00:00Z",

  "structural": {
    "aspect_ratio": 0.76,
    "orientation_candidate_deg": 0,
    "orientation_confidence": "high",
    "orientation_method": "aspect_ratio_flip_v1",
    "deskew_angle_deg_deterministic": -1.1,
    "layout_similarity": null
  },

  "quality": {
    "blur_laplacian_var": 9793.2,
    "contrast_p5_p95_spread": 217,
    "artifact_score": null,
    "document_completeness": null
  },

  "routing": {
    "recommended_action": "route_to_manual_review",
    "confidence_margin": 0.009,
    "anomaly_flag": false
  },

  "encoders": {
    "dinov2": {
      "schema_version": 1,
      "encoder_checkpoint": "vit_small_patch14_dinov2.lvd142m",
      "nearest_validated_neighbors": [
        {"image_id": "...", "similarity": 0.94}
      ],
      "bucket_similarity": {"dense_tabular_rows": 0.81, "printed_document": 0.34},
      "cluster_id": null,
      "duplicate_candidate": false
    },
    "convnext_gradcam": {
      "schema_version": 1,
      "encoder_checkpoint": "convnext_tiny.fb_in22k",
      "structural_hotspot_centroid_norm": [0.47, 0.44],
      "structural_entropy": 0.71,
      "mass_inside_expected_region": 0.66
    },
    "mae": {
      "schema_version": 1,
      "encoder_checkpoint": "vit_base_patch16_224.mae",
      "polarity_anomaly_score": null,
      "note": "PROPOSED use, not yet validated - see §3"
    }
  }
}
```

**Why this shape**: `structural`/`quality`/`routing` are top-level and
encoder-agnostic — a consumer (deskew, routing, quarantine UI) reads
these without needing to know or care which encoder produced them,
same reasoning as `image_analysis.py`'s existing top-level/region split.
The `encoders` block is where provenance lives, keyed by encoder name
with its own `schema_version` — **this is what makes the schema support
future encoders without breaking compatibility**: adding encoder #9
means adding a new key under `encoders`, never touching the top-level
shape or any existing encoder's block. Any top-level field can trace
back to which encoder produced it and under what checkpoint version —
the same audit-trail discipline `GenerationConfig.content_hash()`
already enforces for generation.

---

## 3. Recommended encoder responsibilities (capability matrix)

Every row below is grounded in a specific measured Benchmark 1
finding, cited directly, not asserted.

| Encoder | Measured, reliable metadata | Measured limitation | Overlaps with |
|---|---|---|---|
| **DINOv2** | Nearest-neighbor/duplicate-detection signal — bucket-precision@5 real above the ~0.13 random baseline at n=24, replicated at n=87; stable (recall@1=1.00) under mild deskew/contrast/denoise/resize | **Cannot serve as a rotation/orientation anomaly detector** — just demonstrated directly: 0/126 flagged images scored as anomalous by DINOv2 bucket-similarity, because it was specifically qualified for invariance to exactly this class of transform | SigLIP/NaFlex on retrieval (both give similar precision — see Round 4's scaled-up replication in `docs/VISION_IR_RESEARCH.md`) |
| **ConvNeXt (+ Grad-CAM)** | **Structural/layout signal, via Grad-CAM specifically** — Experiment 3.3 validated this (entropy dropped from ~0.96 to ~0.70, passed the clean-image consistency check both prior methods failed, visually confirmed content-relevant migration) | Its own *unconditional* channel-mean pooling does **not** reliably localize content (Experiment 3.2 — falsified/insufficient evidence) — the raw embedding is not the useful artifact here, the Grad-CAM map is | Swin (both genuine multi-scale pyramids, Swin's Grad-CAM untested — **PROPOSED**, not measured) |
| **SigLIP (fixed-res)** | Same retrieval role as DINOv2, not distinguishably better or worse (0.809 vs. 0.798 at n=87) | No unique validated contribution beyond duplicating DINOv2's role | DINOv2 (near-total overlap on retrieval) |
| **SigLIP-NaFlex** | Aspect-ratio-sensitivity is a real, replicated mechanism (corr ≈ −0.80 at both n=24 and n=87) | **Does not translate into better retrieval** than fixed SigLIP at either sample size — mechanism real, practical benefit not demonstrated | SigLIP-fixed (same role, extra architectural complexity not currently justified by evidence) |
| **EVA-02** | Inversion-sensitive (recall 0.92, similar magnitude to DINOv2) — a **PROPOSED** polarity-anomaly role, same idea as MAE below but weaker signal | No distinctly validated role beyond what DINOv2 already covers | DINOv2 (retrieval profile untested head-to-head, plausibly similar) |
| **BEiT** | Robust to inversion; no standout unique property found across 9 rounds | No clearly unique validated contribution | ConvNeXt/Swin/SigLIP (all "not inversion-sensitive," no further differentiation found) |
| **Swin** | Genuine multi-scale pyramid, architecturally comparable to ConvNeXt | Grad-CAM never actually tested on Swin (**PROPOSED**, not measured); cross-representation consistency of exactly 1.000 is a pooling-mechanism artifact, not a finding | ConvNeXt (same structural role, redundant unless Swin's own Grad-CAM shows something ConvNeXt's doesn't) |
| **MAE** | **The single clearest, most specific signal in the whole qualification set**: recall@1 collapses to 0.21 under inversion while showing near-zero drift (0.0001–0.0024) on every other transform — an almost pure, dedicated polarity/inversion-anomaly detector by construction. **PROPOSED** repurposing, not yet validated for this specific use (Benchmark 1 measured the sensitivity, not whether it's exploitable as a detector) | Worst retrieval-relevant profile of any candidate (lowest cross-representation consistency, 0.696, unexplained) — not useful for retrieval at all | None — this is the most genuinely non-overlapping candidate in the set |

**The clearest worked example of "complementary, not redundant" metadata**,
directly answering the research question this whole benchmark is
built around: **DINOv2 and MAE are near-opposites, and that's exactly
what makes them complementary.** DINOv2 was qualified *because* it's
invariant to routine transforms including (implicitly) the kind of
geometric change that comes with orientation problems — which is
precisely why it just failed, empirically, to serve as a rotation/
polarity anomaly detector (0/126 flagged as anomalous in the practical
exercise immediately preceding this benchmark). MAE has the opposite
profile — hypersensitive to exactly one specific low-level pixel
transform (inversion), inert to everything else. Where DINOv2 has a
blind spot, MAE may have precisely the missing signal. This is a real
design insight, not a hoped-for one, and it's the strongest concrete
argument in this whole document for running more than one encoder.

**Overlap summary**: of the 8, at least 3 pairs are substantially
redundant on currently-measured evidence (DINOv2/SigLIP-fixed on
retrieval; NaFlex/SigLIP-fixed on retrieval, NaFlex's extra complexity
unjustified; ConvNeXt/Swin on structural role, pending Swin's own
Grad-CAM test). BEiT currently has no demonstrated unique role at all.

---

## 4. Runtime measurements (real, not estimated)

Real sequential run, all 8 qualified encoders, same 24-image sample
used throughout Benchmark 1 (`benchmark/benchmark2_sequential_runtime.py`,
`data/outputs/benchmark2_sequential_runtime_log.jsonl`), each encoder
loaded, run, released before the next — never concurrent, matching this
project's existing GPU discipline:

| Encoder | Load time | Inference (24 imgs) | Images/sec | Peak GPU memory |
|---|---|---|---|---|
| DINOv2 | 0.85s | 1.15s | 20.8 | 151MB |
| ConvNeXt | 0.51s | 1.34s | 18.0 | 162MB |
| NaFlex | 1.02s | 0.87s | 27.5 | 430MB |
| SigLIP-fixed | 0.98s | 0.78s | 30.6 | 413MB |
| EVA-02 | 1.02s | 0.97s | 24.6 | 390MB |
| BEiT | 0.96s | 0.85s | 28.3 | 389MB |
| Swin | 1.14s | 1.35s | 17.8 | 411MB |
| MAE | 0.98s | 0.75s | 32.1 | 385MB |

**Totals, all 8 encoders combined**: 7.46s one-time load, 8.06s
inference for 8×24 = 192 embeddings, **0.336s/image for the entire
8-encoder metadata layer**, peak GPU memory 430MB (the single largest
encoder's footprint — never summed, since encoders never run
concurrently), 15.57s total wall-clock for this whole run.

**Corpus-scale extrapolation** (the project's stated eventual scale —
hundreds of thousands of images, e.g. 300,000 as a working figure):
300,000 × 0.336s ≈ **28 hours of GPU-bound processing for all 8
encoders combined**, one-time load overhead (7.46s) negligible at that
scale. This is a real, substantial but genuinely feasible number for a
one-time corpus preprocessing pass (comparable to an overnight-to-
multi-day batch job) — and this measurement used batch size 1
throughout; batching multiple images per forward pass (not attempted
in this test) would very likely cut this further, an easy optimization
left for implementation time rather than benchmark time.

**Working assumption from Benchmark 1 — validated, not just assumed**:
"lightweight encoders are cheap enough to justify sequential
execution" holds up against real combined measurement, not just
per-encoder extrapolation. CPU impact was negligible throughout
(consistent with Jon's own direct observation during the rotation-
triage work — 2-3% CPU utilization, GPU utilization brief and low).

---

## 5. Storage requirements

**Recommendation: derived/calibrated metadata as the primary persisted
artifact; raw embeddings as optional, not required.** This follows the
same principle already established for Architecture D generally
(§26 of the original research): don't store expensive information
unless it has demonstrated future value, and a raw embedding vector is
exactly the kind of thing that's expensive to keep meaningful (tied to
one encoder checkpoint version forever, per §7's original versioning
risk) without necessarily being needed by any current consumer.

| Storage option | Per-image cost (8 encoders) | Recommendation |
|---|---|---|
| Raw float32 embeddings, all 8 encoders | (384+768+768+768+768+768+1024+768) × 4 bytes ≈ **24.4KB** | Not recommended as default — ties every stored value to today's 8 checkpoint versions permanently (§7's versioning risk), for no currently-identified consumer |
| Raw float16 embeddings, all 8 | ≈ 12.2KB | Same objection, half the cost — still not recommended by default |
| Similarity scores + nearest-neighbor IDs only | ~200 bytes (a handful of floats + a few string IDs) | **Recommended primary artifact** — this is what every currently-identified consumer (§6) actually needs |
| Full derived/calibrated metadata (schema in §2, no raw embeddings) | ~500 bytes–1KB (JSON, mostly small numbers/strings) | **Recommended, alongside similarity scores** |

**At corpus scale** (300,000 images): derived metadata alone ≈
150–300MB total — trivial. Raw embeddings for all 8 encoders ≈ 3.7–7.3GB
(float16/float32) — not prohibitive in absolute terms, but not justified
by any consumer identified in §6 either. **Recommendation: do not
persist raw embeddings by default.** If a future consumer genuinely
needs them (e.g. re-clustering the whole corpus against a new
reference set without re-running inference), regenerating them from the
original images is cheap per §4's measurements (0.336s/image for all 8
encoders combined) — cheaper than the ongoing storage-versioning
burden of keeping them "just in case."

---

## 6. Integration plan

Matches the "Vision Analysis / Pipeline Context" framing already
established earlier in this research (`docs/VISION_IR_RESEARCH.md`
§27–32) — this metadata layer is additive, sits upstream of loader
dispatch, and every consumer reads it as either a machine-consumed
field or a text-hint field, never both without an explicit tag (§20's
lock-on risk applies only to the latter).

| Consumer | Reads | Channel |
|---|---|---|
| Deskew / orientation detection | `structural.orientation_candidate_deg`, `structural.deskew_angle_deg_deterministic` | Machine-consumed |
| Bucket routing | `encoders.dinov2.bucket_similarity`, `routing.confidence_margin` | Machine-consumed |
| OCR preprocessing | `quality.*`, `structural.aspect_ratio` | Machine-consumed |
| Duplicate detection | `encoders.dinov2.duplicate_candidate`, `nearest_validated_neighbors` | Machine-consumed |
| Quarantine UI | `routing.anomaly_flag`, per-encoder scores rendered for human review | Machine-consumed (drives what's shown), not a text-hint |
| Corpus search / future retrieval | `encoders.dinov2.nearest_validated_neighbors`, `cluster_id` | Machine-consumed |
| Gemma prompt context (if ever used) | Calibrated categories only (e.g. `structural.orientation_confidence: "high"`), never raw scores | Text-hint — subject to §20's lock-on-risk evaluation before use |

**Where this plugs into the existing pipeline**: as a new stage
between Stage 0 (preprocessing) and Stage 1 (classification) — runs
once per image, writes the metadata artifact, and every listed
consumer reads from it rather than re-deriving. Existing deterministic
Stage A (`image_analysis.py`) is untouched and still feeds the
`structural`/`quality` top-level fields directly; this benchmark adds
the `encoders` block alongside it, not instead of it.

---

## 7. Recommendation: should multi-encoder preprocessing replace single-encoder?

**Yes, but narrowly — not "run all 8."** The real evidence supports a
small, deliberately complementary set, not exhaustive coverage:

- **DINOv2** — semantic retrieval/duplicate detection (validated, §3).
- **ConvNeXt, consumed via Grad-CAM specifically, not raw pooling** —
  structural/layout signal (validated, §3; the raw-pooling route is
  explicitly *not* recommended, per Experiment 3.2's falsification).
- **MAE** — a dedicated polarity/inversion-anomaly signal (**PROPOSED**,
  the most promising untested idea this benchmark surfaced, not yet
  validated for this specific repurposed use — the natural next
  experiment, not a decision to make now).

**Not currently justified by evidence**: SigLIP-fixed and NaFlex
(redundant with DINOv2's retrieval role, NaFlex's added complexity
unproven), EVA-02 (weaker version of MAE's polarity signal, no other
distinct role found), BEiT (no unique validated role at all), Swin
(redundant with ConvNeXt's structural role unless its own Grad-CAM is
separately tested and shown to add something ConvNeXt's doesn't).

**This is a 3-encoder recommendation, not an 8-encoder one** — cheaper
than the already-cheap 8-encoder measurement in §4 (roughly 3/8 of the
combined per-image cost), and every encoder in it has a demonstrated,
non-overlapping reason to be there rather than "it was qualified, so
include it." The remaining 5 are not disqualified permanently — they're
qualified-but-currently-redundant, and any of them could earn a slot
the same way MAE just did: by showing a validated, non-overlapping
signal none of the other three provide.

**Immediate next step, not yet run**: test whether MAE's inversion
sensitivity actually functions as a usable polarity-anomaly detector in
practice (e.g. does MAE-based self-similarity between an image and
candidate polarity-corrected versions reliably flag real inversion
problems, the same kind of practical validation just run for the
rotation-triage exercise) — this is the one recommendation in this
document that's a proposal rather than a measured conclusion, and
should be resolved before treating the 3-encoder recommendation as
final.

---

## Extension: complete metadata layer acquisition

Prompted by Jon's observation that the measured combined runtime
(§4 above) changes the cost model enough to warrant reconsidering
"only run the minimum necessary encoders." Evidence tagging applied
throughout, per Jon's explicit request: **MEASURED** (directly
observed in this or prior sessions' runs), **PROPOSED** (a design
recommendation made here, not yet tested), **INFERENCE** (a reasoned
judgment call not backed by direct measurement — the weakest of the
three, and the one most likely to be wrong).

### 0. A distinction worth drawing before answering — "cheap to compute" is not the same claim as "cheap to trust"

**INFERENCE**: the real, measured 0.336s/image (§4) settles the
*compute* cost question decisively — it's negligible. It does **not**
settle a separate question: whether a given encoder's derived signal
means anything reliable enough to build a pipeline dependency on. This
document's own capability matrix (§3) already showed that most of the
8 encoders' outputs are either redundant with another encoder's
(SigLIP-fixed/NaFlex vs. DINOv2) or entirely unvalidated for any
specific use (BEiT has no demonstrated role at all; MAE's most
promising use — polarity-anomaly detection — is explicitly **PROPOSED**,
not measured). Getting all 8 encoders' outputs during ingestion for
nearly free does not retroactively validate what any of those outputs
actually mean. The recommendation below is built around keeping these
two questions — *is it cheap to capture* and *is it trustworthy enough
to build on* — separate rather than letting a cheap compute cost imply
an unearned confidence bump.

### 1. Stable image properties vs. corpus-relative properties — a distinction Jon's own list doesn't separate

**PROPOSED**, and the single most important structural point in this
extension: not all of Jon's example fields behave the same way over
time, and the versioning strategy (§6 below) depends on getting this
right.

- **Intrinsic, per-image properties** — orientation, polarity, blur,
  contrast, aspect ratio. These are properties of one image and one
  algorithm/checkpoint version. Compute once; valid forever until the
  *algorithm* changes. Good candidates for permanent Tier 1 metadata.
- **Corpus-relative properties** — nearest-neighbor lists, cluster
  assignment, anomaly/outlier score. These are properties of one image
  *relative to whatever the reference corpus looked like at compute
  time*. A nearest-neighbor list computed today is not "wrong" next
  year, but it's stale the moment 10,000 new images are added that
  might be closer matches. This is the same two-pinned-things
  versioning risk already flagged in the original Architecture D
  research (`docs/VISION_IR_RESEARCH.md` §7/§26: encoder checkpoint
  *and* reference set both need independent version tracking) —
  directly reused here, not re-derived.

Jon's example list (orientation, polarity, semantic neighbourhood,
structural similarity, layout descriptors, quality metrics, clustering
information, anomaly scores, confidence estimates) mixes both
categories. Recommend the schema (§3 below) tag every field with which
kind it is, since "does this need reprocessing when the corpus grows"
has a different answer for each.

### 2. Transient implementation details

**MEASURED + PROPOSED**: raw embeddings, Grad-CAM heatmaps, activation
maps, and intermediate tensors should not be persisted as permanent
metadata — this was already the recommendation in §5 above, and the
complete-acquisition framing makes the argument *stronger*, not
weaker: if regenerating all 8 encoders' raw embeddings costs 0.336s/image
(MEASURED), there is even less reason to pay ongoing storage cost for
them indefinitely. Grad-CAM outputs specifically are also tied to
*which classification target* produced them (Experiment 3.3 used
ConvNeXt's ImageNet-22k head, a semantically arbitrary proxy for this
corpus) — storing them long-term bakes in a specific, not-yet-well-
chosen target choice. **Recommend**: cache these as Tier 2 (regeneratable
on demand), never Tier 1.

### 3. Metadata registry — a real improvement over this document's original per-encoder schema

**PROPOSED**: Jon's registry framing (organize by semantic category,
not by which encoder produced it) is a genuine refinement over §2's
original schema. The original schema's `encoders.<name>.<field>`
nesting means every consumer that wants "orientation information" has
to know which encoder(s) might have produced it. A registry organized
by category, with per-*field* provenance instead of per-*block*
provenance, is cleaner for exactly the future-proofing goal this
extension cares about — adding or removing an encoder becomes "add or
remove a `source` entry on some fields," never a schema restructuring:

```json
{
  "metadata_version": 2,
  "image_id": "oocihm.lac_reel_c10264.767.jpg",

  "orientation": {
    "field_kind": "intrinsic",
    "value_deg": 0,
    "confidence": "high",
    "sources": [
      {"method": "aspect_ratio_flip_v1", "deterministic": true},
      {"method": "vision_rotation_similarity", "encoder": "dinov2", "checkpoint": "vit_small_patch14_dinov2.lvd142m"}
    ]
  },
  "polarity": {
    "field_kind": "intrinsic",
    "anomaly_score": null,
    "sources": [
      {"encoder": "mae", "checkpoint": "vit_base_patch16_224.mae", "status": "PROPOSED - not yet validated as a detector"}
    ]
  },
  "quality": {
    "field_kind": "intrinsic",
    "blur_laplacian_var": 9793.2,
    "sources": [{"method": "image_analysis.py", "deterministic": true}]
  },
  "retrieval": {
    "field_kind": "corpus_relative",
    "reference_set_version": "2026-08-01-v1",
    "nearest_neighbors": [{"image_id": "...", "similarity": 0.94}],
    "sources": [{"encoder": "dinov2", "checkpoint": "vit_small_patch14_dinov2.lvd142m"}]
  },
  "anomaly": {
    "field_kind": "corpus_relative",
    "reference_set_version": "2026-08-01-v1",
    "score": null,
    "sources": []
  },
  "routing": {
    "field_kind": "derived_decision",
    "recommended_action": null,
    "confidence_margin": null
  }
}
```

Every category carries its own `field_kind` (intrinsic /
corpus_relative / derived_decision) so a consumer — or a future
maintenance script — can tell at a glance which fields go stale when
the corpus grows versus which stay valid until the algorithm changes.
`sources` is a list, not a single value, specifically because more than
one encoder can and does contribute to the same category (orientation
above has both a deterministic method and a vision-based one) — this
is what makes "different encoders capture different aspects, some of
them overlapping" (this whole benchmark's founding premise) representable
in the schema itself, not just describable in prose.

### 4. Option A (minimal) vs. Option B (complete) — resolved by decoupling capture from trust

**INFERENCE** (this is a judgment call, not a measured conclusion,
and flagged as the weakest-tagged claim in this section): reprocessing
a corpus that's grown to hundreds of thousands of images specifically
*because* a new metadata need was discovered is very likely more
expensive in engineering/coordination terms than the compute itself —
re-identifying which images need it, re-running a batch job at scale,
handling partial failures, reconciling schema versions across a mixed
population. This has **not been directly measured** in this project
(no reprocessing-at-scale event has actually happened yet to time), so
treat this as a reasoned expectation, not a demonstrated cost.

Given that, and given §0's distinction: **the real decision isn't
"compute 3 encoders vs. compute 8" — that decision is already settled
by §4's measurement (all 8 is cheap). The real decision is which
encoders' outputs get promoted to trusted, pipeline-relied-upon Tier 1
status versus which get computed-and-cached but left unvalidated.**
This dissolves the apparent conflict between this extension's framing
and the original §7 recommendation (3 encoders) — they're answering
different questions. **PROPOSED resolution**: run all 8 during
ingestion (or as many as remain cheap as more are qualified), cache
every output, but only DINOv2 (retrieval), ConvNeXt-via-Grad-CAM
(structural), and — once validated — MAE (polarity) graduate to fields
other pipeline stages are allowed to build hard dependencies on. The
other 5 sit in the registry as computed-but-unpromoted, available for
research, backtesting, or future validation without a corpus-wide
reprocessing pass, exactly the outcome Jon is asking this extension to
secure.

**A real cost this resolution doesn't erase, worth stating rather than
waving away** (INFERENCE): running 8 encoders means maintaining 8
encoders' worth of loader code, prefix-token/tensor-layout handling,
and failure modes in production — each qualified in this research only
after catching a real bug specific to it (Swin's NHWC layout, SigLIP's
zero-prefix-tokens, NaFlex's default-transform trap). Cheap GPU-seconds
don't make that maintenance surface free. This is a genuine argument
for keeping the *actively-relied-upon* set small (the 3 already
recommended) even while capturing the full 8 for caching purposes.

### 5. Storage strategy — three tiers, Tier 2 doing double duty

**PROPOSED**:

- **Tier 1 — persistent, versioned, human-readable.** Only fields
  promoted per §4's resolution: validated orientation/quality/routing
  fields, DINOv2 retrieval fields, ConvNeXt-Grad-CAM structural
  summary fields (not the raw heatmap). Small (§5 of the original
  document already estimated ~500B–1KB/image for a narrower set;
  a wider set of *validated* fields stays in the same range, since
  size is dominated by field count, not encoder count).
- **Tier 2 — cached, regeneratable, not yet trusted.** This tier does
  two jobs, worth keeping distinct even though they share
  infrastructure: (a) genuinely transient artifacts with no long-term
  value (Grad-CAM heatmaps, raw embeddings for already-validated
  fields — the derived summary is what matters, not the vector it came
  from), and (b) a **staging ground** for the 5 currently-unpromoted
  encoders' outputs — computed now while the image is already being
  touched, available for future validation work, not yet a pipeline
  dependency. Decoupling capture from trust (§4) is what makes this
  tier legitimate rather than just "we stored stuff we don't use yet."
- **Tier 3 — original source image.** Canonical, never derived from
  anything else, the ground truth everything above can always be
  regenerated from if a version or schema mistake needs correcting.

### 6. Metadata versioning

**PROPOSED**, building directly on §1's intrinsic/corpus-relative split:

- Every Tier 1 field: `schema_version` (structural — does the *shape*
  of this field's data match what the reading code expects) plus,
  for any encoder-derived field, the producing checkpoint's identifier.
- Every corpus-relative field additionally carries `reference_set_version`
  — independent of `schema_version`, since a corpus-relative field can
  go stale (new reference images added) without the schema or encoder
  changing at all. Conflating these two version axes was flagged as a
  real risk back in the original research (§7) and applies identically
  here.
- **"The image should never require reprocessing simply because the
  schema grows" is satisfied by treating each metadata record as
  field-level append-only, not an atomic blob**: adding encoder #9
  means adding new fields/sources to existing records over time (a
  lazy backfill, image by image, whenever each is next touched for any
  reason) — never rewriting or invalidating what's already there. A
  record missing encoder #9's fields is incomplete, not wrong; reading
  code should treat an absent field as "not yet computed," never as an
  implicit zero or false.

### 7. Recommendation

**MEASURED**: all 8 encoders combined cost 0.336s/image and 430MB peak
GPU memory (§4) — decisively cheap at any corpus scale this project
has discussed.

**PROPOSED, following from §0's decoupling, not a simple "yes" to
Jon's framing**: transition to **complete metadata *capture*** during
the initial ingestion pass — run all 8 qualified encoders (and any
future ones cheaply enough to join them), cache every output in Tier 2.
Do **not** transition to complete metadata *trust* — only the 3
encoders already earning a validated role in §7 above (DINOv2,
ConvNeXt-via-Grad-CAM, and MAE once its polarity-detector use is
actually tested) graduate to Tier 1 fields other stages depend on. This
gets Jon's stated goal — never reprocess the corpus just because a new
use case is discovered later — without quietly relaxing this project's
evidence discipline to "cheap enough to compute" standing in for
"validated enough to trust." The corpus becoming durable infrastructure
(Jon's framing) is a good reason to capture broadly; it is not, on its
own, a reason to skip the same validation discipline every one of this
project's other conclusions has gone through.

---

## Second extension: persistence vs. trust as independent axes, and a reversed recommendation

Jon's correction: the previous extension still justified capture
primarily on compute cost, and Jon is right that this is the weaker
argument. Reframing around two genuinely independent questions —
*should this observation be preserved* and *should downstream stages
consume it automatically* — changes one recommendation from §5 above,
not just the framing. Evidence tags carried through as before.

### Why capture-at-ingestion is the stronger argument, not just a restatement

**MEASURED, from this project's own observed history, not a general
platitude**: images in this corpus have already been reclassified,
relocated, and pruned during the course of this session alone — the
flagged-deskew batch's 10 missing files were confirmed already pruned
mid-conversation, a real, directly observed instance of the corpus
mutating out from under an earlier snapshot. The claim that "the
corpus is mutable and ingestion is a comparatively well-defined moment"
is not hypothetical here — it already happened.

**One honest refinement, not a rejection**: ingestion is the *first*
well-defined checkpoint, not necessarily the *only* one worth a
snapshot — a later re-deskew or manual-review pass also produces a new
known-state worth its own capture. Doesn't weaken the core point;
worth not overstating it as singular.

**INFERENCE, but a well-grounded one**: embeddings are a function of
checkpoint, preprocessing, resize/interpolation policy, and library
versions, not just the image — correct, and this project has already
hit a concrete, *measured* instance of the general phenomenon Jon is
describing: `docs/CODE_MAP.md` records that `cv2.HoughLinesP` returns a
different array shape in OpenCV 5.x than 4.x for the identical
operation on identical input — a real, already-encountered case of a
library-version upgrade silently changing behavior in this exact
codebase. That's not proof embeddings specifically will drift the same
way (that remains untested, hence INFERENCE, not MEASURED), but it's
concrete evidence the general risk class is real here, not
speculative.

### Reversed recommendation: persist raw embeddings, don't regenerate-on-demand

**PROPOSED — this reverses §5's original position, and the reversal is
earned, not conceded.** The original recommendation ("don't store raw
embeddings, they're cheap to regenerate") was justified almost
entirely on compute cost — precisely the framing Jon is now correctly
flagging as insufficient. Re-examining it against the reproducibility
question it never actually addressed:

- **Storage cost is genuinely trivial, including for all 8 encoders.**
  ~24.4KB/image at float32 (8 embeddings, dims 384–1024), ~12.2KB at
  float16. At 300,000 images: 3.7–7.3GB total — smaller than the
  *source images* by two to three orders of magnitude. The original
  recommendation's cost argument was correct but was answering the
  wrong question.
- **Could environment-pinning (recording exact library/checkpoint
  versions) substitute for storing the actual vector?** Considered and
  rejected as a full substitute: pinning versions helps, but does not
  fully guarantee bit-identical reproduction even with an identical
  software environment — floating-point GPU computation is not
  guaranteed bit-reproducible across different hardware or driver
  generations, a well-established, general limitation of the field,
  not unique to this project (**INFERENCE** — not directly tested in
  this project, but not a fabricated concern either). Keeping the
  original output vector sidesteps this entirely, trivially, since it
  *is* the original data, not a reconstruction attempt.
- **Persisted embeddings materially cheapen future corpus-relative
  work.** Refreshing a nearest-neighbor list or re-clustering against a
  grown reference set is a comparison-only operation if the archived
  embeddings exist — versus a full corpus re-embedding pass (with all
  of the above drift risk) if they don't. This is the concrete
  operational payoff of the reproducibility argument, not just a
  principle.

**Net conclusion: persist raw embeddings as an archival tier, keyed by
checkpoint identifier (not a single flat field — a later DINOv2-base
pass, say, adds a new keyed entry alongside the existing DINOv2-small
one, never overwrites it).** float16 recommended over float32 for the
archival copy — halves storage cost, and the added rounding error is
small relative to the cosine-similarity comparisons this data actually
serves (**INFERENCE** — not directly measured against a float32
baseline in this project).

**What does *not* reverse**: Grad-CAM heatmaps, activation maps, and
intermediate tensors remain regenerate-on-demand, not persisted. Their
reproducibility case is materially weaker than the embeddings' —
Experiment 3.3 already established Grad-CAM's output depends on an
essentially arbitrary proxy classification target (ConvNeXt's
ImageNet-22k head, meaningless for this corpus), so there's no stable
ground-truth computation being protected by keeping the old heatmap the
way there is for an embedding vector. Their storage cost is also much
higher (image-like objects, not compact vectors) with no comparable
payoff.

### Provenance fields, made concrete per Jon's list

Every persisted observation (Tier 1 or the newly-reversed embedding
archive) carries all of: `source_encoder`, `checkpoint`,
`preprocessing_hash`, `library_versions`, `pipeline_version`,
`timestamp`, `image_hash`. Non-optional — Jon's framing is correct that
without this, the reproducibility this whole extension is built around
is lost in practice even if the raw data is kept. Schema update from
the registry sketch above:

```json
{
  "orientation": {
    "field_kind": "intrinsic",
    "value_deg": 0,
    "confidence": "high",
    "sources": [
      {
        "method": "vision_rotation_similarity",
        "encoder": "dinov2",
        "checkpoint": "vit_small_patch14_dinov2.lvd142m",
        "preprocessing_hash": "sha256:...",
        "library_versions": {"timm": "1.0.28", "torch": "2.13.0+cu130"},
        "pipeline_version": "benchmark2-v1",
        "timestamp": "2026-08-01T00:00:00Z",
        "image_hash": "sha256:..."
      }
    ]
  },
  "embeddings_archive": {
    "dinov2": {
      "checkpoint": "vit_small_patch14_dinov2.lvd142m",
      "dtype": "float16", "dim": 384,
      "vector_ref": "s3://.../embeddings/dinov2/vit_small_patch14_dinov2/<image_hash>.npy",
      "preprocessing_hash": "sha256:...", "library_versions": {"...": "..."},
      "pipeline_version": "benchmark2-v1", "timestamp": "2026-08-01T00:00:00Z"
    }
  }
}
```

### Architecture, adopting Jon's pipeline shape directly

This is a cleaner statement of the same conclusion already reached
above — adopted as given, since it's correct and better-phrased than
this document's own earlier version:

```
Image
  ↓
Encoder observations
  ↓
Persistent metadata (archive - everything, including embeddings now)
  ↓
Validation benchmark
  ↓
Operational metadata registry (pipeline consumes only this)
```

**The pipeline consumes only validated fields. The archive preserves
everything.** MEASURED-tagged fields (validated through Benchmark 1 or
2) may influence routing automatically. PROPOSED fields are computed,
preserved, and visible to researchers, but never auto-influence a
pipeline decision. INFERENCE is an architectural judgment call, never
presented as evidence.

### Storage tiers, revised

- **Tier 1 (operational, validated)**: unchanged from before — small,
  human-readable, only the fields that earned a role in §7.
- **Tier 2 (persistent archive, revised)**: now includes raw
  embeddings, keyed by checkpoint, with full provenance — no longer
  "optional/regeneratable," but not part of the operational hot path
  either. Grad-CAM/activation maps/intermediate tensors stay
  regenerate-on-demand within this tier, for the reasons above — same
  tier, different retention policy per artifact type.
- **Tier 3 (source image)**: unchanged, canonical.

### What doesn't change from the maintenance-surface caveat already raised

Still true, and Jon's message doesn't remove it: running 8 encoders in
production is 8 encoders' worth of loader-code maintenance (Swin's
NHWC layout, SigLIP's zero-prefix-tokens, NaFlex's default-transform
trap — each caught only by hitting it directly this session). Cheap
compute and now-justified persistence don't make that surface free —
it's a separate, still-standing argument for keeping the
*operationally-trusted* set at 3, independent of how broadly capture
and archival now extend.

---

## Correction: real corpus size is 1,500 images, not 300,000

This changes the actual argument in this document, not just its
numbers — every scale-dependent figure above (the 28-hour extrapolation,
the 3.7–7.3GB storage estimate) was built on the "hundreds of thousands"
framing from the very first research document, which should have been
flagged as an assumption before being extrapolated on, not carried
forward silently. Recomputed against the real figure:

**MEASURED (recomputed from §4's real per-image rates, not re-run)**:

| Quantity | At 300,000 (prior framing) | At 1,500 (actual) |
|---|---|---|
| All 8 encoders, full corpus | ~28 hours | **8.4 minutes** |
| DINOv2 alone, full corpus | — | **~72 seconds** |
| Raw embeddings, all 8, float32 | 3.7–7.3GB | **34.4MB** |
| Raw embeddings, all 8, float16 | — | **17.2MB** |
| Derived Tier-1 metadata | 150–300MB | **1.5MB** |

**What this genuinely weakens**: the strongest argument made above for
*pre-emptive complete capture* was "avoid an expensive future
reprocessing burden." That argument loses most of its force when
reprocessing *everything, with every encoder, from scratch* is an
8.4-minute task. A gap discovered later is cheap to fill after the
fact at this scale — the urgency behind capturing now specifically to
avoid reprocessing pain later was implicitly sized for a corpus two
orders of magnitude larger than the real one.

**What doesn't weaken, because it was never a cost argument**: the
persistence/trust decoupling, the reproducibility case for keeping raw
embeddings (checkpoint and library drift don't become less real in a
smaller corpus), the provenance schema, and the MEASURED/PROPOSED/
INFERENCE discipline all stand independent of corpus size.

**Honest proportionality check, not raised before this correction**:
some of the machinery designed above — `reference_set_version` tracking
for corpus-relative fields, per-checkpoint-keyed archival tiers, a
staging ground for 5 not-yet-promoted encoders' outputs — was reasoned
about as though recomputation were expensive enough to justify
avoiding. At 1,500 images it isn't. **Recommend**: keep the schema and
provenance discipline (designing it correctly costs nothing extra
regardless of scale), but don't over-invest in incremental-backfill
automation or elaborate versioning tooling this corpus size doesn't
need yet — a script that reruns all 8 encoders in under 10 minutes
whenever something changes is itself a perfectly adequate "backfill
strategy" at this scale, and building more than that now would be
solving a problem this corpus doesn't have. The honest framing for
complete capture at this size is **"cheap enough that there's no real
reason not to,"** not **"necessary to avoid a costly future
burden"** — a materially different, more modest justification than
this document argued for before the correction.

---

## Third extension: one canonical baseline embedding, captured pre-preprocessing — with a measured exception

Jon's point: capturing the embedding once, before the preprocessing
pipeline runs, means every downstream stage compares against the *same*
vector — rather than each stage potentially re-embedding a differently-
preprocessed version of the same image and getting a subtly different
result. This is a real design principle, not just a preference, and
it's already backed by this project's own data rather than needing new
justification.

**MEASURED, pulled directly from Round 1's real numbers**: DINOv2's
drift under the transforms this pipeline actually applies is small —
deskew +0.5° (0.029), contrast enhancement (0.037), denoise (0.031) —
and recall@1 stays perfect (1.00) under all three. Capturing the
baseline before these specific steps costs very little retrieval
quality while buying the consistency Jon describes: corpus-relative
operations (nearest-neighbor, clustering, duplicate detection) never
silently drift as the same image is reprocessed over time, and a
similarity score computed today stays comparable to one computed a
year from now against the same stored baseline.

**One refinement, not a rejection**: "before preprocessing" needs to
mean *before the preprocessing steps proven to matter little*, not
*before all preprocessing equally* — and this project already has a
case where that distinction is load-bearing. Inversion causes far more
drift than anything else measured (0.163 — 4–5× deskew/contrast/
denoise's effect), and Round 1/2's own polarity-normalization design
decision already recommends normalizing polarity *before* embedding.
Those two conclusions only fit together one way: **the canonical
baseline should be captured after polarity normalization specifically,
but before deskew/contrast/denoise/autocontrast** — not literally
"before every preprocessing step." Embedding a wrongly-inverted raw
scan as the permanent, never-revisited baseline would bake in exactly
the largest source of spurious drift this whole research effort found,
for the one step proven to matter at that magnitude. The rest are
measured-safe to skip before embedding; polarity isn't.

**A distinction worth keeping precise**: the baseline is stable across
*preprocessing pipeline* changes (a better deskew algorithm next year
doesn't invalidate an already-stored baseline) but should still be
recomputed if the *source image itself* changes (a corrected scan
replaces a known-bad one, or a mis-scanned page is replaced). These are
different events — "the pipeline got better" versus "the ground truth
changed" — and treating them the same would mean either needlessly
recomputing a stable baseline every time preprocessing improves, or
never fixing a baseline that's wrong because the source was wrong.

**Schema implication — one field was missing from the provenance list**:
alongside `checkpoint`/`preprocessing_hash`/`library_versions`, every
archived embedding needs an explicit `preprocessing_stage` marker
(e.g. `"raw_ingested"` vs. `"post_polarity_normalized"`), distinct from
the hash. The hash captures *exact parameters*; the stage captures
*which conceptual point in the pipeline* the input was at — both are
needed, since "was this embedded before or after polarity correction"
is a question a hash alone doesn't answer legibly.

**What this doesn't preclude**: nothing above rules out computing
*additional*, clearly-labeled embeddings at other pipeline stages for
specific other purposes (e.g. a fully-preprocessed-state embedding for
a consumer that specifically benefits from the cleaned-up image) — the
principle is that there must be exactly *one* canonical, non-drifting
baseline that every corpus-relative comparison uses by default, not
that no other embedding may ever be computed. Any additional embedding
must be tagged clearly enough that it can never be silently substituted
for the baseline in a comparison that assumes consistency.

---

## Fourth extension: capture both baseline and post-preprocessing embeddings — the delta is itself new metadata

Jon's proposal: rather than choosing between a pre-preprocessing
baseline and a post-preprocessing snapshot, capture both — rerun the
tower immediately after preprocessing completes, store that as a
second, separate record, at negligible additional cost.

**MEASURED (recomputed, not estimated)**: capturing both stages, all 8
encoders, float16: **34.6MB for the entire 1,500-image corpus**, 16.8
minutes total compute. The "smaller than a 10-second mp4" comparison
holds up under an actual calculation, not just as a figure of speech.

**This resolves the earlier tension more cleanly than picking one
default** — the third extension above landed on "one canonical baseline
plus optional other embeddings, clearly labeled." Capturing exactly
two well-defined snapshots (baseline: post-polarity-normalization,
pre-other-preprocessing; postprocessing: after the full pipeline
completes) is a more concrete, systematic version of that same
principle, not a different one — no ambiguity about which "other"
embeddings might exist, exactly two, both well-defined.

**What this actually buys, beyond consistency (worth drawing out
explicitly, not just accepting the storage argument)**:

- **The baseline-to-postprocessing delta is a real, per-image
  measurement, not a statistical estimate.** Round 1's drift numbers
  (deskew ≈0.03, contrast ≈0.037) are *aggregate* findings from
  synthetic transforms applied to a 24-image sample. Once every real
  image gets both embeddings captured, each image has its own **actual**
  drift value, from whatever preprocessing genuinely happened to it —
  a materially stronger signal than "the aggregate says this class of
  transform is usually small." **PROPOSED**: a per-image drift value
  well outside Round 1's established normal range (the roughly 0.02–0.09
  band DINOv2 showed for deskew/contrast/denoise) is itself a cheap,
  automatic flag that *something atypical happened during this specific
  image's preprocessing* — worth a look, independent of knowing yet
  which specific step caused it.
- **This is a genuinely different anomaly mechanism than the one that
  already failed** in the flagged-deskew rotation-triage work earlier
  in this session — that approach compared an image's embedding to its
  *bucket's* average and found DINOv2 too invariant to geometric
  transforms to flag anything (0/126). This mechanism instead compares
  an image **to itself**, across its own pipeline transition — DINOv2's
  invariance to *typical* preprocessing is exactly what makes an
  *atypical* jump stand out, rather than being swamped by normal
  between-image variation the way a corpus-wide bucket comparison is.
  Complementary to, not a retry of, the earlier failed approach.
- **This is also the natural experiment that validates the still-open
  MAE-as-polarity-detector proposal from §7**, essentially for free.
  If polarity normalization runs between baseline and postprocessing
  capture, images that genuinely needed correction should show MAE's
  large, specific invert-magnitude drift (Round 9: up to 0.047–0.17
  depending on severity) between their two snapshots, while
  already-correct images should show MAE's near-zero drift on
  everything else. No separate dedicated experiment needed — real
  preprocessing runs generate the validation data as a byproduct.

**One implementation question worth resolving before building this,
not glossed over**: "postprocessing" needs one precise, well-defined
trigger point in the actual pipeline, not an assumed single moment —
this corpus's preprocessing isn't uniform across buckets (recall
`dense_tabular_rows` runs a multi-stage row-segmentation workflow with
manually-confirmed deskew, not a single automatic pass like other
buckets). Recommend the trigger be "preprocessing is marked complete/
confirmed for this image," whatever that means per-bucket, not a fixed
pipeline-stage name assumed to apply uniformly.

**Versioning follow-through**: per the second extension's append-only
principle, a later preprocessing re-run (e.g. fixing images from
`flagged_bad_deskew.csv`) should add a **new**, separately-timestamped
postprocessing snapshot, not overwrite the old one — which additionally
means the pipeline can directly answer "did the fix actually change
this image's embedding, and by how much" later, by comparing
postprocessing snapshots across pipeline versions rather than just
baseline-to-single-postprocessing.

---

## Fifth extension: where exactly does "postprocessing complete" mean — before or after Gemma?

Jon's question: should the postprocessing snapshot be captured after
Gemma's bucket classification, or could the tower's own embeddings
pre-classify the image before Gemma ever sees it?

**These aren't competing placements — the baseline embedding (third
extension) already answers the "before Gemma" half of this question,
independent of classification entirely.** The baseline is captured
post-polarity-normalization, pre-other-preprocessing, which is already
upstream of Gemma. Nothing new needs to be built for the tower's
embeddings to exist before Gemma runs — that data point already exists
in this design.

**PROPOSED, for the postprocessing snapshot specifically**: capture it
*after* Gemma's classification completes. This is the safe default —
no new validation required, and it means the snapshot gets tagged with
a trusted bucket label immediately (per the schema's `field_kind`
distinction, a corpus-relative field scoped to a *known-correct*
bucket, not a provisional guess).

**On using the tower to pre-classify before Gemma — this is real, but
it's not a new idea, it's a resurfacing of the Stage 0.5/linear-probe
question from much earlier in this research, set aside (not resolved)
in favor of the encoder-qualification work that became Benchmark 1.**
Picking it back up needs the same evidence discipline, not a shortcut
just because the embeddings will already exist:

- **MEASURED**: DINOv2 bucket-*retrieval* precision (~0.4 at n=24,
  ~0.8 at n=87 — inflated by larger same-bucket candidate pools at the
  larger sample). This is a different claim from classification
  accuracy on a genuinely unlabeled image — retrieval precision
  measures whether same-bucket images rank as near neighbors of each
  other, not whether the single highest-similarity bucket reliably
  matches the correct label.
- **Not yet measured, and required before trusting this for anything
  beyond research**: real classification accuracy against Gemma's own
  decisions, and a calibrated confidence threshold for when "trust the
  tower" would actually be safe.
- **One reframing worth making explicit, since it changes what the
  experiment should even optimize for**: the value case for this idea
  should not be "skip Gemma to save compute" — this project's own
  stated priority (from the very first message of this whole research
  effort) is that runtime reduction is a bonus, not the objective. The
  stronger, better-aligned framing is **cross-checking**: does the
  tower's independent pre-classification *agree* with Gemma's decision?
  Agreement raises confidence in that classification; disagreement is a
  cheap, automatic review flag — this serves the project's actual
  stated priority (fewer manual touches) directly, where "skip Gemma
  for speed" would not.

**What this means practically**: no new pipeline changes are needed to
start collecting the evidence this question needs. Once baseline
embeddings and Gemma's real classification outcomes are both being
logged (which the existing design already produces), the comparison —
does tower-predicted bucket match Gemma's actual decision, and at what
confidence — can be run retrospectively against real data, the same
way the fourth extension's MAE-validation experiment falls out of the
baseline/postprocessing design for free. Recommend treating "tower
cross-checks or pre-classifies" as an explicitly separate, still-
PROPOSED research question, not a decision to make now — and keeping
the postprocessing snapshot's placement (after Gemma) unblocked by it.

---

## Sixth extension: real pilot data (n=11), not synthetic — first MEASURED result for the cross-check question

Real data collected via `benchmark/benchmark2_pilot_isolated.py`, following
the exact archive-immutable/copy-first/per-image-checkpoint pattern
established above: 11 real archival scans (the subset of the Round 1-9
continuity sample with a traceable raw source under
`J:\Screenshots\Knott_Ancestry\Archive Microfilms`, read-only throughout),
each producing its own `data/outputs/benchmark2/image_NNNN/` folder with
`raw_copy`, `metadata_raw.json` (Checkpoint A), `preprocessed`,
`metadata_preprocessed.json` (Checkpoint B), `gemma_reference.json`
(Checkpoint C, real Gemma inference, never written to production bucket
CSVs), and `comparison.json`. Corpus-level `summary.json` rolls up all 11.

**MEASURED — real preprocessing drift**: 0.0001–0.0024, mean 0.0012,
across all 11 images. Deskew angle was exactly 0.000° for every image
(consistent with the already-documented false-zero limitation of
`estimate_deskew_angle` on this microfilm sub-corpus — not a new
finding, a real-world confirmation of one already on record). With no
real rotation applied, the only preprocessing that ran was
`autocontrast`, and the drift this produced is smaller even than Round
1's synthetic autocontrast test (already ~0.0000) — genuine real-world
confirmation of an already-established Benchmark 1 result, not an
independent new data point.

**MEASURED — the cross-check question, answered with real data for the
first time**: DINOv2's nearest-neighbor bucket prediction agreed with
Gemma's real classification decision on **9 of 11 images (82%)**,
identically at both Checkpoint A and Checkpoint B (expected, given how
small the drift between them was).

**The two disagreements are not the same kind of disagreement, and
collapsing them into "2 misses" would lose the more interesting
finding**:

- `oocihm.lac_reel_c10264.767.jpg` — tower predicts `handwritten_ledger`
  at a genuinely confident score (0.761); Gemma classified it
  `printed_document` (confidence 0.95). This is the same image used
  throughout Experiment 3/3.1/3.2/3.3 — direct prior visual inspection
  (the Grad-CAM panels) confirms it's a cursive-handwritten list
  organized in ruled columns ("*These are the names of some of the men
  I have distributed...*"). The tower's classification is at least
  defensible here, plausibly more accurate than Gemma's. **This is
  exactly the kind of disagreement the cross-check design is meant to
  surface for review — not evidence the tower is wrong.**
- `oocihm.lac_reel_c10301.601.jpg` — tower predicts `printed_document`,
  but at a low score (0.15–0.16, far below the other disagreement's
  0.761); Gemma classified it `portrait_photo` (confidence 0.98). The
  tower's own low confidence here is the actual signal — this looks
  like a weak best-of-a-poor-field guess, not a genuine competing claim.
  **This is precisely why a confidence threshold matters for the
  cross-check design**: without one, this disagreement would look
  identical in kind to the first, when the two are not comparable at all.

**Honest scope of this result**: n=11 is a real pilot, not a
statistically powered validation — 82% agreement with one clearly
defensible and one clearly low-confidence disagreement is an
encouraging first result for the cross-check proposal, not a
conclusion. The natural next step is the same one already recommended:
scale this exact protocol to more images before treating the
cross-check idea as validated, now that the pipeline to do so exists
and has been proven correct end-to-end (real archive, real
preprocessing, real Gemma, fully isolated from production state).

**A concrete confirmation of the versioning risk, on the Gemma side
this time, not just the vision towers**: Jon confirmed the classifier
prompt was edited between earlier stages of this research and now
(adding a new `website_screenshot` bucket category — which also
explains why that bucket appeared in the corpus's current bucket CSVs
but not in the samples used throughout Rounds 1–9). Checked directly:
**`ClassificationResult.prompt_version` stayed `"v1"` across this real
content edit** — the version label was not bumped when the prompt
changed. This is the exact failure mode the schema design already
anticipated for encoder checkpoints (a version *label* can go stale;
a *hash* of the actual content can't), now confirmed to apply equally
to Gemma's prompt file, not just vision-tower checkpoints. Fixed going
forward: `gemma_reference.json` now also records `model`,
`prompt_file`, and a `prompt_file_hash` (sha256 of the actual prompt
text) alongside the version label. The existing 11 records were
backfilled with these fields directly from the current config (not by
re-running Gemma — the classification decisions themselves don't
change on a re-run of the same model against the same prompt, only the
recorded provenance was incomplete), with an explicit
`_provenance_note` marking them as backfilled rather than re-inferred.

**A real preserved "before" snapshot exists, which matters for this
specific gap**: this working directory is a branch off Jon's separate,
frozen `...Main` install, which still holds the pre-edit pipeline
(prompt without `website_screenshot`, prior config) intact and
unmodified. Unlike the vision-tower checkpoints (where "the exact
original computation" only exists if explicitly archived per the third
extension), Gemma's pre-edit prompt/config is already safely preserved
elsewhere without any extra effort — meaning a genuine, clean
prompt-version-A-vs-B comparison (re-run these same 11 images' real
classification through Main's original config, compare against this
pilot's current-config results) is available as a real experiment
later, not just a hash-verified "we know it changed" statement. Not
run here; noted as a concrete option if it's ever worth quantifying how
much of the classification shift is attributable to the prompt edit
specifically, versus the case-by-case reasoning already given for the
one clearly defensible disagreement above.

### Full cross-checkpoint comparison, all 11 images (not just the disagreements)

| Image | Deskew° | Drift (A→B) | Tower A | Score A | Tower B | Score B | Gemma | Gemma conf | A=Gemma | B=Gemma |
|---|---|---|---|---|---|---|---|---|---|---|
| c10264.658 | 0.000 | 0.0024 | printed_document | 0.787 | printed_document | 0.784 | printed_document | 0.98 | ✔ | ✔ |
| c10264.730 | 0.000 | 0.0023 | printed_document | 0.889 | printed_document | 0.886 | printed_document | 0.98 | ✔ | ✔ |
| c10264.734 | 0.000 | 0.0019 | printed_document | 0.931 | printed_document | 0.930 | printed_document | 0.98 | ✔ | ✔ |
| c10264.735 | 0.000 | 0.0021 | printed_document | 0.928 | printed_document | 0.930 | printed_document | 0.98 | ✔ | ✔ |
| c10264.767 | 0.000 | 0.0017 | handwritten_ledger | 0.761 | handwritten_ledger | 0.761 | printed_document | 0.95 | ✘ | ✘ |
| c10301.601 | 0.000 | 0.0017 | printed_document | 0.152 | printed_document | 0.160 | portrait_photo | 0.98 | ✘ | ✘ |
| c10414.116 | 0.000 | 0.0003 | printed_document | 0.711 | printed_document | 0.708 | printed_document | 0.98 | ✔ | ✔ |
| c10414.126 | 0.000 | 0.0004 | dense_tabular_rows | 0.705 | dense_tabular_rows | 0.703 | dense_tabular_rows | 0.98 | ✔ | ✔ |
| c10414.129 | 0.000 | 0.0003 | map_land_record | 0.483 | map_land_record | 0.483 | map_land_record | 1.00 | ✔ | ✔ |
| c10610.354 | 0.000 | 0.0003 | dense_tabular_rows | 0.734 | dense_tabular_rows | 0.734 | dense_tabular_rows | 0.95 | ✔ | ✔ |
| t2185.798 | 0.000 | 0.0001 | handwritten_ledger | 0.807 | handwritten_ledger | 0.808 | handwritten_ledger | 0.98 | ✔ | ✔ |

**MEASURED — zero bucket-prediction flips between checkpoints**: the
tower's predicted bucket is identical at A and B for all 11 images —
at this magnitude of real preprocessing drift (0.0001–0.0024), the two
checkpoints are functionally interchangeable for anything downstream
that only consumes the predicted bucket, not the raw score.

**MEASURED — confidence does not cleanly separate agreement from
disagreement, and this matters for any future threshold design**: the
one *defensible* disagreement (`c10264.767`, score 0.761) scores
*higher* than two *correct* agreements (`c10414.129` at 0.483, whose
Gemma confidence was the highest in the set at 1.00; `c10414.126` at
0.705). Only the genuinely low-confidence disagreement (`c10301.601`,
0.15–0.16) is cleanly separable by score alone. A single confidence
threshold could not distinguish "trustworthy agreement" from "the one
disagreement worth reviewing" without also catching real, correct
predictions in the same range — n=11 is too small to calibrate a real
threshold, but the pattern itself (confidence and correctness aren't as
tightly coupled as a first look suggests) should carry into any future
decision about promoting the cross-check idea to an automated signal.
