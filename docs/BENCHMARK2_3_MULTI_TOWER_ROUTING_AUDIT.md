# Benchmark 2.3 — Multi-Tower Routing Audit

**Status**: first pass complete. Extends Benchmark 2.2's single-tower
(DINOv2 only) cross-validator to all 8 encoders qualified in the frozen
Vision Qualification Battery v1.0 (`docs/VISION_IR_RESEARCH.md`), on
the IDENTICAL stratified 146-image corpus (same `BUCKET_PLAN`, same
deterministic bucket-CSV slicing - directly comparable to that
baseline, not a different sample). Script: `benchmark/benchmark2_3_
multi_tower_routing_audit.py`.

**Real gap this closes**: Benchmark 2.2 only ever persisted the 64
disagreement cases to disk - the other 82 agreement cases' Gemma
confidence/reason were never saved, and only one tower (DINOv2) was
ever run. This run re-classifies fresh with Gemma (146 images, cheap)
so every image has a complete, consistent record - all 8 tower
predictions + Gemma's - not a mix of fresh and reconstructed data.

**GPU sequencing**: all 8 timm tower encoders run on CPU
(`embed_pooled()` never moves anything to `cuda` - confirmed by
reading `vision_encoder_qualification.py` directly), so there's zero
GPU contention computing all 8 before Gemma loads. Gemma loads once at
the end, classifies all 146, releases - no concurrent-residency
question to resolve, unlike the reference-classifier-qualification
work.

## Pairwise agreement matrix (9x9: 8 towers + Gemma)

```
                    dinov2  convnext  naflex_siglip  siglip_fixed  eva02  beit  swin  mae  gemma
dinov2                1.00      0.75           0.67          0.73   0.75  0.77  0.77 0.56   0.56
convnext              0.75      1.00           0.66          0.73   0.77  0.84  0.86 0.64   0.64
naflex_siglip         0.67      0.66           1.00          0.82   0.70  0.67  0.66 0.49   0.55
siglip_fixed          0.73      0.73           0.82          1.00   0.75  0.70  0.72 0.60   0.57
eva02                 0.75      0.77           0.70          0.75   1.00  0.79  0.80 0.57   0.62
beit                  0.77      0.84           0.67          0.70   0.79  1.00  0.89 0.62   0.61
swin                  0.77      0.86           0.66          0.72   0.80  0.89  1.00 0.63   0.60
mae                   0.56      0.64           0.49          0.60   0.57  0.62  0.63 1.00   0.49
gemma                 0.56      0.64           0.55          0.57   0.62  0.61  0.60 0.49   1.00
```

**Reading this**: clear encoder-family clustering, not a flat blob of
similar numbers.
- ConvNeXt/BEiT/Swin cluster tightly (0.84-0.89 mutual agreement) -
  the hierarchical CNN/supervised-lineage encoders.
- NaFlex-SigLIP and fixed-SigLIP cluster together (0.82) - same
  underlying pretraining objective, different resolution handling.
- **MAE is a consistent outlier**, agreeing weakly with every other
  tower AND with Gemma (0.49-0.64) - consistent with this project's
  earlier qualification-round finding that MAE's masked-autoencoder
  objective produces different feature semantics than contrastive/
  supervised pretraining (docs/VISION_IR_RESEARCH.md).
- Gemma's agreement with any single tower (0.49-0.64) is generally
  LOWER than tower-tower agreement - expected: Gemma reasons about
  semantic content + written category instructions, towers do pure
  visual-similarity nearest-neighbor clustering. Different signal,
  which is the whole point of using both.

## Consensus distribution (n=146)

```
unanimous               53   (all 9 voters agree)
majority                73   (>=5/9 agree on one bucket)
split                   19   (multiple buckets w/ real support, no majority)
complete_disagreement    1   (no bucket has more than 1-2 votes)
```

86% of images (unanimous + majority) show real, stable consensus among
the 9 voters - only one image is genuine chaos. This is a good signal
that a consensus-based gate is stable, not just noise dressed up as
agreement.

## Gemma vs. tower-consensus disagreements

**56/146 (38%)** - notably LOWER than Benchmark 2.2's single-tower
(DINOv2-only) disagreement rate of **64/146 (44%)**. The multi-tower
consensus (majority/plurality bucket among the 8 towers) is a cleaner,
less noisy signal than any single tower's opinion alone - some of
DINOv2's individual disagreements with Gemma were DINOv2 being the
outlier while the broader tower ensemble would have agreed with Gemma.
This is the concrete benefit multi-tower auditing was meant to
provide over Benchmark 2.2's single-tower design.

All 56 disagreement cases saved to `data/outputs/benchmark2_3_multi_tower/
consensus_disagreements/disagreement_NNN/` (image + full record.json
including every tower's vote, tower consensus bucket/strength, and a
`human_verdict: null` placeholder for manual review - same pattern as
Benchmark 2.2's disagreement archive).

## Evidence locations

- `data/outputs/benchmark2_3_multi_tower/all_results.json` - full
  per-image record for all 146 images (every tower's bucket+score,
  Gemma's bucket+confidence+reason, consensus category).
- `data/outputs/benchmark2_3_multi_tower/pairwise_agreement.json` -
  the 9x9 matrix above, machine-readable.
- `data/outputs/benchmark2_3_multi_tower/consensus_disagreements/` -
  56 disagreement folders, human-review-ready.
- `data/outputs/benchmark2_3_multi_tower/summary.json` - run metadata.

## Deferred (not part of this first pass)

- Subtype interface (bucket + subtype + properties schema, "unknown"
  default) - flagged as valuable by the c10264.767 taxonomy-boundary
  case found during Reference Classifier Qualification
  (`docs/REFERENCE_CLASSIFIER_QUALIFICATION.md`), not built yet.
- Human-verdict review of the 56 consensus disagreements (same paused
  status as Benchmark 2.2's 64 single-tower disagreements).
- Any confidence-gating threshold decision using this consensus signal
  in production - this remains observational only, nothing written to
  production manifests or bucket CSVs.
