"""
Validates whether diagnostics/row_band_clustering.py's clustered-row-band
signal is real, discriminating evidence - per this project's Hypothesis
-> Experiment -> Benchmark -> Evidence -> Production discipline, this
must be measured across a BALANCED multi-category sample before any
integration decision, not judged from the 2 handwritten_ledger images
that motivated building it.

Hypothesis: substantial_coverage_fraction (how much of the page height
is covered by dense clusters of "row" detections, per row_band_clustering.py)
should be meaningfully higher for dense_tabular_rows and handwritten_ledger
than for the other 6 flat-8 categories, IF this is a real "repeated
record structure" signal and not an artifact specific to the 2 images
already inspected.

Runs the census-bootstrap fine-tuned checkpoint (table/row/header) on
N images per category, all 8 flat-8 categories, computes row_band_signal
per image, and reports per-category statistics plus a simple separability
check (does the dense_tabular_rows/handwritten_ledger group's mean
clearly exceed the other 6 categories' mean, or is it noisy/overlapping).

NOT wired into core/sensor_adapters.py or the Decision Engine - this is
the validation step that decides whether that's worth doing next.

Usage:
    python diagnostics/test_row_band_signal_validity.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from diagnostics.vit_family_benchmark_common import get_split
from diagnostics.error_analysis.common import OLD_WORKING_PREFIX, LEGACY_WORKING_IMAGES_DIR
from diagnostics.row_band_clustering import row_band_signal

CENSUS_CHECKPOINT = (
    PROJECT_ROOT.parent / "genealogy_workspace" / "datasets" / "bootstrap" / "layout_bootstrap_train"
    / "runs" / "census_bootstrap_v4" / "weights" / "best.pt"
)
REPORT_PATH = PROJECT_ROOT / "data" / "logs" / "reviewed" / "row_band_signal_validity_report.txt"

N_PER_CATEGORY = 10
ROW_STRUCTURE_CATEGORIES = {"dense_tabular_rows", "handwritten_ledger"}  # the hypothesis group


def resolve_path(path_str: str) -> str:
    if Path(path_str).exists():
        return path_str
    if path_str.startswith(OLD_WORKING_PREFIX):
        filename = path_str[len(OLD_WORKING_PREFIX):].lstrip("\\/")
        remapped = str(LEGACY_WORKING_IMAGES_DIR / filename)
        if Path(remapped).exists():
            return remapped
    return path_str


def run_census_checkpoint(model, image_path: str) -> list[dict]:
    results = model.predict(image_path, imgsz=1280, conf=0.2, verbose=False)
    result = results[0]
    names = result.names
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            detections.append({
                "class_id": class_id,
                "class_name": names.get(class_id, str(class_id)),
                "confidence": round(float(box.conf.item()), 4),
                "bbox_xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()],
            })
    return detections


def main():
    lines = []
    def log(s=""):
        lines.append(s)
        print(s, flush=True)

    if not CENSUS_CHECKPOINT.exists():
        raise FileNotFoundError(f"{CENSUS_CHECKPOINT} not found.")

    log("Validating the clustered-row-band signal across a balanced 8-category sample "
        f"({N_PER_CATEGORY}/category) - is it real discriminating evidence, or an artifact "
        "of the 2 images that motivated it?\n")
    log(f"Hypothesis group (expected HIGH substantial_coverage_fraction): {sorted(ROW_STRUCTURE_CATEGORIES)}\n")

    from ultralytics import YOLO
    log(f"Loading census-bootstrap checkpoint from {CENSUS_CHECKPOINT}...")
    model = YOLO(str(CENSUS_CHECKPOINT))
    log("Loaded.\n")

    by_cat, train_items, val_items, test_items = get_split()

    results_by_cat: dict[str, list[dict]] = {}
    for category in sorted(by_cat.keys()):
        paths = by_cat.get(category, [])[:N_PER_CATEGORY]
        log(f"Processing {category} ({len(paths)} images)...")
        cat_results = []
        for path in paths:
            resolved = resolve_path(path)
            if not Path(resolved).exists():
                log(f"  SKIP (not found): {path}")
                continue
            with Image.open(resolved) as img:
                h = img.height
            detections = run_census_checkpoint(model, resolved)
            signal = row_band_signal(detections, image_height=h)
            cat_results.append(signal)
        results_by_cat[category] = cat_results

    log(f"\n{'='*100}")
    log("PER-CATEGORY SUMMARY (mean over N images)")
    log(f"{'='*100}")
    log(f"{'category':<20s} {'n':>3s} {'n_bands':>9s} {'n_subst':>9s} {'total_boxes':>12s} "
        f"{'coverage':>10s} {'subst_coverage':>15s}")
    cat_means = {}
    for category, results in results_by_cat.items():
        if not results:
            continue
        n = len(results)
        mean_n_bands = statistics.mean(r["n_bands"] for r in results)
        mean_n_subst = statistics.mean(r["n_substantial_bands"] for r in results)
        mean_total_boxes = statistics.mean(r["total_row_boxes"] for r in results)
        mean_coverage = statistics.mean(r["coverage_fraction"] for r in results)
        mean_subst_coverage = statistics.mean(r["substantial_coverage_fraction"] for r in results)
        cat_means[category] = mean_subst_coverage
        flag = " <- hypothesis group" if category in ROW_STRUCTURE_CATEGORIES else ""
        log(f"{category:<20s} {n:>3d} {mean_n_bands:>9.2f} {mean_n_subst:>9.2f} {mean_total_boxes:>12.1f} "
            f"{mean_coverage:>10.3f} {mean_subst_coverage:>15.3f}{flag}")

    log(f"\n{'='*100}")
    log("SEPARABILITY CHECK")
    log(f"{'='*100}")
    hypothesis_vals = [cat_means[c] for c in ROW_STRUCTURE_CATEGORIES if c in cat_means]
    other_vals = [v for c, v in cat_means.items() if c not in ROW_STRUCTURE_CATEGORIES]
    if hypothesis_vals and other_vals:
        hyp_mean = statistics.mean(hypothesis_vals)
        other_mean = statistics.mean(other_vals)
        other_max = max(other_vals)
        log(f"Hypothesis group (dense_tabular_rows/handwritten_ledger) mean substantial_coverage_fraction: "
            f"{hyp_mean:.3f}")
        log(f"Other 6 categories mean: {other_mean:.3f}  (highest single other category: {other_max:.3f})")
        if hyp_mean > other_max:
            log("\nRESULT: hypothesis group's mean CLEANLY EXCEEDS every other category's mean - "
                "real separability at the category-mean level, worth pursuing further "
                "(e.g. per-image threshold tuning, then a real EvidenceRecord adapter).")
        elif hyp_mean > other_mean:
            log("\nRESULT: hypothesis group's mean is higher on average but OVERLAPS with at least "
                "one other category's mean - a directional signal, not yet a clean discriminator. "
                "Needs per-image distribution inspection (not just category means) before deciding.")
        else:
            log("\nRESULT: no clear separation at the category-mean level - this signal, at least "
                "in this simple form, does NOT look like a valid discriminator on this sample. "
                "Do not integrate without a different feature formulation or larger sample.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
