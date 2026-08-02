"""
Cumulative comparison report generator for the Vision Qualification
Battery v1.0 (see benchmark/vision_encoder_qualification.py and
docs/VISION_IR_RESEARCH.md). Regenerate after every completed round -
reads data/outputs/vision_encoder_qualification_log.jsonl directly, no
hand-transcribed tables.

Deliberately does NOT auto-generate the "patterns identified" prose -
that's interpretive judgment (mechanism vs. utility, falsification
tracking, "don't generalize from one encoder"), not something a
heuristic should produce for an N=5-9 comparison. This script produces
the quantitative tables only; the interpretation is written directly
into docs/VISION_IR_RESEARCH.md by whoever reviews each new round,
same discipline as every round so far.

ARCHITECTURE_FACTS below are one-time architecture properties (not
re-measured per run) established during this session's earlier timm
capability-matrix work - extend this dict as new families are
qualified. Everything else in the report comes straight from the JSONL
log.

Usage:
    python -m benchmark.vision_encoder_comparison_report
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_LOG = PROJECT_ROOT / "data" / "outputs" / "vision_encoder_qualification_log.jsonl"
REPORT_PATH = PROJECT_ROOT / "docs" / "VISION_ENCODER_COMPARISON_REPORT.md"

# One-time architecture facts, established via direct inspection during
# this session's capability-matrix work (docs/VISION_IR_RESEARCH.md,
# sixth addendum) - not re-measured per qualification run.
ARCHITECTURE_FACTS = {
    "vit_small_patch14_dinov2.lvd142m": {
        "family": "DINOv2", "training_objective": "self-supervised (no text/class supervision)",
        "native_resolution": False, "dynamic_resolution_support": "opt-in flag, patch-aligned input only",
        "feature_pyramid": "same-resolution depth stack (not true multi-scale)",
    },
    "convnext_tiny.fb_in22k": {
        "family": "ConvNeXt", "training_objective": "supervised classification (ImageNet-22k)",
        "native_resolution": True, "dynamic_resolution_support": "native, no flag needed",
        "feature_pyramid": "true multi-scale (56->28->14->7)",
    },
    "naflexvit_base_patch16_siglip.v2_webli": {
        "family": "SigLIP (NaFlex)", "training_objective": "image-text contrastive (WebLI)",
        "native_resolution": True, "dynamic_resolution_support": "native (NaFlex design point) - but only via the dedicated patchify/patch_coord call path, NOT timm's default create_transform (see Round 4)",
        "feature_pyramid": "same-resolution depth stack (not true multi-scale)",
    },
    "vit_base_patch16_siglip_224.v2_webli": {
        "family": "SigLIP (fixed-res)", "training_objective": "image-text contrastive (WebLI)",
        "native_resolution": False, "dynamic_resolution_support": "opt-in flag, patch-aligned input only",
        "feature_pyramid": "same-resolution depth stack (not true multi-scale)",
    },
    "eva02_base_patch14_224.mim_in22k": {
        "family": "EVA-02", "training_objective": "masked image modeling + CLIP-distillation (mim_in22k)",
        "native_resolution": False, "dynamic_resolution_support": "opt-in flag, patch-aligned input only",
        "feature_pyramid": "same-resolution depth stack (not true multi-scale)",
    },
    "beit_base_patch16_224.in22k_ft_in22k": {
        "family": "BEiT", "training_objective": "masked image modeling (BEiT-style, ImageNet-22k)",
        "native_resolution": False, "dynamic_resolution_support": "NOT implemented in timm's BEiT class",
        "feature_pyramid": "same-resolution depth stack (not true multi-scale)",
    },
    "swin_base_patch4_window7_224.ms_in22k": {
        "family": "Swin", "training_objective": "supervised classification (ImageNet-22k)",
        "native_resolution": False, "dynamic_resolution_support": "NOT supported (window-partitioning constraint)",
        "feature_pyramid": "true multi-scale (56->28->14->7)",
    },
}

# Deliberately NOT a curated subset of "interesting" transforms - Jon's
# direct correction (2026-07-31): this benchmark exists to discover
# future surprises, not confirm current hypotheses. Highlighting only
# the transforms that have already differentiated DINOv2/ConvNeXt/
# NaFlex would systematically under-display the other transforms in
# every future round's report, exactly where a genuinely new family
# might diverge in a way nothing so far has. The full per-transform
# table below is the primary view for every candidate, every transform,
# every round - no filtering.


def load_records():
    records = []
    if not RESULTS_LOG.exists():
        return records
    with open(RESULTS_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    # one record per candidate - keep the latest if a candidate was ever re-run
    by_candidate = {}
    for r in records:
        by_candidate[r["candidate"]] = r
    return list(by_candidate.values())


def build_report(records) -> str:
    lines = []
    lines.append("# Vision Encoder Comparison Report")
    lines.append("")
    lines.append("**Auto-generated from `data/outputs/vision_encoder_qualification_log.jsonl` "
                  f"by `benchmark/vision_encoder_comparison_report.py`. {len(records)} candidates qualified "
                  "under Vision Qualification Battery v1.0 (frozen - transforms/metrics/pass-fail unchanged "
                  "since Round 1). Regenerate after every completed round; do not hand-edit the tables below "
                  "- edit `ARCHITECTURE_FACTS` or the generator script instead. Interpretive analysis lives in "
                  "`docs/VISION_IR_RESEARCH.md`, not here.**")
    lines.append("")

    # --- operational characteristics table ---
    lines.append("## Operational characteristics")
    lines.append("")
    header = ["Candidate", "Family", "Params (M)", "Embed dim", "Patches", "Prefix tokens",
              "Pooling", "CPU ms", "GPU ms", "GPU peak MB", "Native res", "Dynamic res", "Feature pyramid"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for r in records:
        op = r.get("operational_metadata", {})
        facts = ARCHITECTURE_FACTS.get(r["candidate"], {})
        row = [
            r["candidate"],
            facts.get("family", "?"),
            str(op.get("params_millions", "?")),
            str(op.get("embedding_dim", "?")),
            str(op.get("patch_count", "?")),
            str(op.get("num_prefix_tokens", "?")),
            op.get("pooling_method", "?"),
            str(op.get("cpu_latency_ms", "?")),
            str(op.get("gpu_latency_ms", "?")),
            str(op.get("gpu_peak_memory_mb", "?")),
            str(facts.get("native_resolution", "?")),
            facts.get("dynamic_resolution_support", "?"),
            facts.get("feature_pyramid", "?"),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # --- qualification metrics: aggregate summary first, then the FULL
    # per-transform breakdown for every transform, every candidate - no
    # curated subset, per Jon's direct correction above.
    lines.append("## Qualification metrics (Vision Qualification Battery v1.0)")
    lines.append("")
    lines.append("### Aggregate summary (mean across all 10 transforms)")
    lines.append("")
    header2 = ["Candidate", "Mean recall@1", "Mean drift", "Mean Jaccard@5", "Cross-repr. consistency"]
    lines.append("| " + " | ".join(header2) + " |")
    lines.append("|" + "---|" * len(header2))
    for r in records:
        battery = r.get("transform_battery", {})
        all_recalls = [v["recall_at_1"] for v in battery.values() if v.get("recall_at_1") is not None]
        all_drifts = [v["mean_cosine_drift"] for v in battery.values() if v.get("mean_cosine_drift") is not None]
        all_jaccards = [v["mean_neighborhood_jaccard"] for v in battery.values() if v.get("mean_neighborhood_jaccard") is not None]
        mean_recall = sum(all_recalls) / len(all_recalls) if all_recalls else None
        mean_drift = sum(all_drifts) / len(all_drifts) if all_drifts else None
        mean_jaccard = sum(all_jaccards) / len(all_jaccards) if all_jaccards else None
        row = [
            r["candidate"],
            f"{mean_recall:.3f}" if mean_recall is not None else "?",
            f"{mean_drift:.4f}" if mean_drift is not None else "?",
            f"{mean_jaccard:.3f}" if mean_jaccard is not None else "?",
            f"{r.get('cross_representation_consistency_corr', float('nan')):.3f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("**An aggregate mean can hide a single-transform outlier** (this is exactly how "
                  "DINOv2's inversion result would look diluted into a mean) - the full per-transform "
                  "tables below are the primary evidence, not this summary. Read the summary as an "
                  "index into the detail, not a replacement for it.")
    lines.append("")

    all_transforms = sorted({t for r in records for t in r.get("transform_battery", {})})
    for metric_key, metric_label in [
        ("recall_at_1", "Recall@1"),
        ("mean_cosine_drift", "Mean cosine drift"),
        ("mean_neighborhood_jaccard", "Mean neighborhood Jaccard@5"),
    ]:
        lines.append(f"### Full per-transform breakdown — {metric_label}")
        lines.append("")
        header3 = ["Candidate"] + all_transforms
        lines.append("| " + " | ".join(header3) + " |")
        lines.append("|" + "---|" * len(header3))
        for r in records:
            battery = r.get("transform_battery", {})
            row = [r["candidate"]]
            for t in all_transforms:
                v = battery.get(t, {}).get(metric_key)
                row.append(f"{v:.3f}" if isinstance(v, (int, float)) else "?")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("## Observed behavioral notes")
    lines.append("")
    lines.append("(Interpretive - written by hand in `docs/VISION_IR_RESEARCH.md` after each round, "
                  "not auto-generated here. This report is the quantitative substrate for that analysis, "
                  "not a replacement for it.)")
    lines.append("")

    return "\n".join(lines)


def main():
    records = load_records()
    report = build_report(records)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote comparison report for {len(records)} candidates to {REPORT_PATH}")


if __name__ == "__main__":
    main()
