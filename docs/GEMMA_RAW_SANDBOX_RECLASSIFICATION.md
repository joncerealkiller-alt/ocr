# Gemma raw-image sandbox reclassification (2026-08-04)

**Status**: measurement only. Fully sandboxed - verified by construction
and by a before/after checksum diff of every production file
(`data/pipeline.db`, `data/manifest.csv`, all `data/buckets/*.csv`) that
confirmed zero changes.

## What this tested

Production only ever classifies POST-processed images
(`core/classifier.py`'s `run()` reads `data/working`, always after Stage
3). For the 77 "flagged" images from `benchmark/gained_lost_agreement_
characterization.py` (57 gained-agreement + 20 lost-agreement - every one
of them a case where the TOWER's prediction changed between pre/post),
Gemma's own decision on the RAW pixels had never been measured. This ran
it: copied the 77 hash-verified raw copies into an isolated sandbox
directory outside the repo, loaded the production Gemma model/prompt
(read-only against `config/pipeline.yaml`), and classified each one -
reusing only `build_classifier_loader()` and `loader.classify()`, never
`run()`/`open_bucket_writers()`/`PipelineDatabase` (the only functions in
`core/classifier.py` that write anything).

## [A] Source Evidence

| Comparison | Match rate |
|---|---|
| Gemma-on-raw vs. Gemma-on-postprocessed (production) | 76/77 (98.7%) |
| Gemma-on-raw vs. tower's PRE-processing prediction | 21/77 (27.3%) |
| Gemma-on-raw vs. tower's POST-processing prediction | 56/77 (72.7%) |

By group:

| Group | n | Gemma-on-raw matches tower-pre | Gemma-on-raw matches tower-post |
|---|---|---|---|
| gained agreement | 57 | 1/57 | 56/57 |
| lost agreement | 20 | 20/20 | 0/20 |

The single Gemma flip (raw -> postprocessed) is `Screenshot 2026-06-01
221507.png`, in the gained group: Gemma read it as `mixed_text_image` on
raw pixels, `printed_document` post-processing - the same direction the
tower's own prediction moved for that image (`mixed_text_image` ->
`printed_document`).

## [B] Interpretation

Gemma's classification is essentially stable across preprocessing for
this flagged set (76/77) - in sharp contrast to the vision towers, whose
predictions changed for all 77 of these images by construction (that's
why they were selected). This is a real, measured asymmetry: whatever
about preprocessing moves the tower's embedding-based read around, it
does not similarly move Gemma's read for the same images.

The per-group split is unusually clean and worth reading carefully:

- **Gained-agreement group**: the tower's POST-processing prediction
  agrees with what Gemma says about the RAW image in 56/57 cases. This
  is stronger evidence than the earlier finding (tower-post matches
  Gemma-on-postprocessed) - it shows the tower's post-processing read
  also agrees with an INDEPENDENT judgment made on completely different
  (raw) pixels, not just with Gemma's own postprocessed-image call. The
  tower's pre-processing read was the outlier relative to both.
- **Lost-agreement group**: perfectly clean split - Gemma's raw-image
  read matches the tower's PRE-processing prediction in all 20 cases,
  and matches the tower's POST-processing prediction in zero. Combined
  with Gemma being stable (raw-Gemma = postprocessed-Gemma) for all 20 of
  these, this isolates the "loss" entirely to the tower's own
  post-processing read diverging - not to any ambiguity or instability
  in Gemma's judgment, and not to the raw-image tower reading having been
  a fluke.

## [C] Conclusions

This does not establish that Gemma is "correct" and the tower is
"wrong" (or vice versa) - Gemma is an independent model, not ground
truth, and this project's own tower-consensus-disagreement research has
already found real cases where the tower's call looked more defensible
than Gemma's. What it does establish, as measured fact: for the
57-image gained-agreement group, three independent signals converge
(tower-post, Gemma-on-raw, Gemma-on-postprocessed) - the tower's
post-processing prediction is the one that's broadly corroborated. For
the 20-image lost-agreement group, the tower's post-processing
prediction is the outlier against all three other signals, which stayed
mutually consistent. If a future decision needs to pick which snapshot's
tower reading to trust more by default, this is real, corroborating
evidence in favor of post-processing for the gained group and against it
for the lost group - specific to these 77 images, not a corpus-wide claim.
