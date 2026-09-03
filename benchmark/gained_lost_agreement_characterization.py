"""
Characterizes the two smallest, most interesting groups from benchmark/
stage1_gemma_stage4_flip_report.py's output: the 57 images where
preprocessing moved the tower's top-bucket prediction TOWARD Gemma's
real decision ("gained agreement"), and the 20 where it moved AWAY
("lost agreement"). For each image in both groups, pulls the bucket
transition (pre -> post -> Gemma) and the physical deltas (deskew,
contrast, blur) for that same image, then compares the two groups
against each other AND against the remaining flipped-but-agreement-
unchanged images (a control group) - to see whether "did the agreement
change" correlates with "how much did the physical pixels change," or
whether it looks more like semantic-encoder noise unrelated to the
size of the preprocessing effect.

Read-only, reuses already-computed CSVs (data/outputs/
stage1_gemma_stage4_flip_report/flip_report.csv, both analysis_report
CSVs) rather than recomputing anything. Measurement/characterization
only - does not decide whether a gain or loss is "real" (per this
project's standing "no ground truth" position, Gemma matching isn't
proof of correctness either).

Usage:
    python -m benchmark.gained_lost_agreement_characterization
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FLIP_REPORT_PATH = PROJECT_ROOT / "data" / "outputs" / "stage1_gemma_stage4_flip_report" / "flip_report.csv"
PHYSICAL_PRE_PATH = PROJECT_ROOT / "data" / "outputs" / "image_analysis" / "analysis_report.csv"
PHYSICAL_POST_PATH = PROJECT_ROOT / "data" / "outputs" / "image_analysis" / "analysis_report_postprocessing.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "gained_lost_agreement_characterization"

PHYSICAL_FIELDS = ["deskew_angle_deg", "page_contrast_p5_p95_spread", "blur_laplacian_var", "noise_residual_std"]


def _load_physical() -> tuple[dict, dict]:
    with open(PHYSICAL_PRE_PATH, newline="", encoding="utf-8") as f:
        pre = {r["file_path"]: r for r in csv.DictReader(f)}
    with open(PHYSICAL_POST_PATH, newline="", encoding="utf-8") as f:
        post = {r["file_path"]: r for r in csv.DictReader(f)}
    return pre, post


def _delta(pre_row: dict, post_row: dict, field: str) -> float | None:
    try:
        return float(post_row[field]) - float(pre_row[field])
    except (TypeError, ValueError, KeyError):
        return None


def _print_group(name: str, rows: list[dict], physical_pre: dict, physical_post: dict) -> list[dict]:
    print(f"\n=== {name} ({len(rows)} images) ===")
    enriched = []
    for r in rows:
        image = r["image"]
        p_pre, p_post = physical_pre.get(image), physical_post.get(image)
        deltas = {}
        if p_pre and p_post:
            for field in PHYSICAL_FIELDS:
                deltas[field] = _delta(p_pre, p_post, field)
        transition = f"{r['tower_top_bucket_pre']} -> {r['tower_top_bucket_post']}  (gemma: {r['gemma_bucket']})"
        print(f"  {Path(image).name:<45} {transition:<70} "
              f"deskew_d={deltas.get('deskew_angle_deg')!s:>8} "
              f"contrast_d={deltas.get('page_contrast_p5_p95_spread')!s:>8} "
              f"blur_d={deltas.get('blur_laplacian_var')!s:>10} "
              f"encoders_flipped={r['n_encoders_flipped']}")
        enriched.append({**r, **{f"delta_{k}": v for k, v in deltas.items()}})
    return enriched


def _group_stats(name: str, rows: list[dict]) -> dict:
    stats = {}
    for field in PHYSICAL_FIELDS:
        vals = [r[f"delta_{field}"] for r in rows if r.get(f"delta_{field}") is not None]
        if not vals:
            continue
        arr = np.array(vals)
        stats[field] = {"n": len(arr), "mean_abs": float(np.abs(arr).mean()), "mean": float(arr.mean())}
    print(f"\n{name} aggregate |delta| means:")
    for field, s in stats.items():
        print(f"  {field:<30} mean_abs={s['mean_abs']:.4f}  mean={s['mean']:+.4f}  n={s['n']}")
    return stats


def main() -> None:
    with open(FLIP_REPORT_PATH, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    physical_pre, physical_post = _load_physical()

    gained = [r for r in all_rows if r["agreement_category"] == "gained_agreement"]
    lost = [r for r in all_rows if r["agreement_category"] == "lost_agreement"]
    control = [r for r in all_rows if r["agreement_category"] in ("agrees_both", "disagrees_both")]

    gained_enriched = _print_group("GAINED agreement (pre disagreed with Gemma, post now agrees)",
                                    gained, physical_pre, physical_post)
    lost_enriched = _print_group("LOST agreement (pre agreed with Gemma, post no longer does)",
                                  lost, physical_pre, physical_post)
    control_enriched = []
    for r in control:
        image = r["image"]
        p_pre, p_post = physical_pre.get(image), physical_post.get(image)
        deltas = {}
        if p_pre and p_post:
            for field in PHYSICAL_FIELDS:
                deltas[field] = _delta(p_pre, p_post, field)
        control_enriched.append({**r, **{f"delta_{k}": v for k, v in deltas.items()}})

    print("\n" + "=" * 80)
    gained_stats = _group_stats("GAINED", gained_enriched)
    lost_stats = _group_stats("LOST", lost_enriched)
    control_stats = _group_stats("CONTROL (agreement unchanged, still flipped)", control_enriched)

    # Bucket-transition tally, both groups
    print("\n=== Bucket transitions (top_bucket_pre -> top_bucket_post) ===")
    for name, rows in [("GAINED", gained), ("LOST", lost)]:
        print(f"\n{name}:")
        transitions = {}
        for r in rows:
            key = f"{r['tower_top_bucket_pre']} -> {r['tower_top_bucket_post']}"
            transitions[key] = transitions.get(key, 0) + 1
        for key, count in sorted(transitions.items(), key=lambda kv: -kv[1]):
            print(f"  {key:<60} {count}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in [("gained", gained_enriched), ("lost", lost_enriched)]:
        path = OUTPUT_DIR / f"{name}_agreement_detail.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n{name} detail written to {path}")

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps({
        "gained_stats": gained_stats, "lost_stats": lost_stats, "control_stats": control_stats,
        "n_gained": len(gained), "n_lost": len(lost), "n_control": len(control),
    }, indent=2), encoding="utf-8")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
