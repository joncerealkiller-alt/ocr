"""
LoRA Training Set Export — converts ground_truth_log.jsonl (from
ground_truth_labeling_ui.py) into image+target-text training pairs.

Per project scope: the first LoRA should improve cursive recognition,
literal transcription, and uncertainty handling - NOT formatting or
structured output. This export reflects that directly: each target is
a single field's honest value (including "illegible"/"blank"/partial-
with-?), not a full structured multi-field record. Teaching the model
to produce a confident-looking complete record is explicitly the wrong
training signal for this project's goal (finding aid, not source of
truth) - a LoRA trained on cleaned-up/completed targets would actively
un-teach the abstention behavior the pipeline needs.

Usage:
    python export_lora_dataset.py [--log-file PATH] [--out-dir PATH]
        [--min-examples-per-status N]

Outputs to --out-dir (default: data/outputs/lora_dataset/):
    images/<uuid>.png          - one cropped row image per example
    train.jsonl                 - one record per line:
                                   {"image": "images/<uuid>.png",
                                    "column": "Age",
                                    "target": "unclear" | "34" | "3?" | ""}
    dataset_report.txt          - counts per status/column, so it's
                                   visible before training whether the
                                   set is skewed (e.g. almost no
                                   "illegible" examples means the LoRA
                                   won't learn that behavior at all)

Target text format (deliberately NOT the pipe-delimited "value|
confidence" convention used elsewhere in this project): a LoRA teaching
literal transcription should produce literal transcription, not learn
to also emit a confidence tag it wasn't shown a real signal for -
confidence is a downstream structuring-stage concern (see
build_structuring_prompt), not a stage-1 recognition concern. Statuses
map to plain text:
    readable            -> the typed value, as-is
    partially_readable   -> the typed value (already contains any ?)
    illegible            -> the literal string "illegible"
    blank                -> the literal string "blank"
This mirrors ocr_stage1_birthplace_field.txt's DITTO convention -
a recognizable literal token for a real, meaningful non-value, not an
empty string a training pipeline might collapse or drop silently.
"""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from collections import Counter, defaultdict
from pathlib import Path

from core.row_segmentation import load_sidecar, crop_region_from_source, compute_exclude_ranges

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LOG = PROJECT_ROOT / "data" / "outputs" / "ground_truth_log.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "data" / "outputs" / "lora_dataset"

STATUS_TO_TARGET = {
    "illegible": "illegible",
    "blank": "blank",
    # readable / partially_readable use the record's own typed value
}


def _load_records(log_path: Path) -> list[dict]:
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _record_to_target(record: dict) -> str | None:
    """
    Returns the training target text, or None if this record shouldn't
    be included (e.g. 'readable' status with an empty value somehow
    slipped through - the labeling UI shouldn't allow this, but this
    export doesn't trust that invariant blindly at the boundary between
    two separately-run tools).
    """
    status = record.get("status")
    if status in STATUS_TO_TARGET:
        return STATUS_TO_TARGET[status]
    if status in ("readable", "partially_readable"):
        value = (record.get("value") or "").strip()
        return value if value else None
    return None


def _sidecar_cache_get(cache: dict, sidecar_path: str) -> dict | None:
    if sidecar_path not in cache:
        try:
            cache[sidecar_path] = load_sidecar(sidecar_path)
        except Exception as e:
            print(f"WARNING: could not load sidecar {sidecar_path}: {e}")
            cache[sidecar_path] = None
    return cache[sidecar_path]


def export(
    log_path: Path, out_dir: Path, min_examples_per_status: int = 0,
    tight_crop_padding_px: int = 20, tight_crop_padding_pct: float | None = None,
) -> None:
    records = _load_records(log_path)
    if not records:
        print(f"No records found in {log_path}. Nothing to export.")
        return

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    report_path = out_dir / "dataset_report.txt"

    sidecar_cache: dict[str, dict] = {}
    status_counts: Counter = Counter()
    column_counts: defaultdict = defaultdict(Counter)
    skipped_no_target = 0
    skipped_bad_row = 0
    written = 0

    with open(train_path, "w", encoding="utf-8") as train_f:
        for record in records:
            target = _record_to_target(record)
            if target is None:
                skipped_no_target += 1
                continue

            sidecar_path = record.get("sidecar_path")
            sidecar = _sidecar_cache_get(sidecar_cache, sidecar_path)
            if sidecar is None:
                skipped_bad_row += 1
                continue

            row_index = record.get("row_index")
            row = next((r for r in sidecar["rows"] if r["index"] == row_index), None)
            if row is None:
                print(f"WARNING: row {row_index} not found in {sidecar_path} - skipping "
                      f"this example (sidecar may have been regenerated since labeling).")
                skipped_bad_row += 1
                continue

            source_path = sidecar["source_image_path"]
            deskew_angle = sidecar["deskew_angle"]
            # BUG FIX (2026-07-22): this used to read the sidecar's
            # LEGACY top-level mask_keep_ranges/mask_apply_rows fields -
            # which are empty/unused for any sidecar created under the
            # per-column architecture. That meant EVERY exported
            # training image was the same unmasked full row, regardless
            # of which column the label was for - confirmed via a real
            # export where Name's and Age's images for the same row
            # were byte-identical. Now reads the per-column mask that
            # was ACTUALLY used at extraction time, matching
            # core.row_extraction.run_single_column_extraction exactly.
            column_state = sidecar.get("columns", {}).get(record.get("column"))
            if column_state is not None:
                keep_ranges = [tuple(k) for k in column_state.get("mask_keep_ranges", [])]
                apply_rows = column_state.get("mask_apply_rows", True)
            else:
                keep_ranges = [tuple(k) for k in sidecar.get("mask_keep_ranges", [])]
                apply_rows = sidecar.get("mask_apply_rows", False)
            width = sidecar["deskewed_image_size"][0]
            mask_active = bool(keep_ranges) and apply_rows
            row_masks = compute_exclude_ranges(keep_ranges, width) if mask_active else []

            try:
                crop = crop_region_from_source(
                    source_path, row["bbox"], deskew_angle, row_masks,
                    # Tight-crop (2026-07-22): the mask above only PAINTS
                    # outside the kept range white, it doesn't narrow the
                    # image - without this, a masked crop was still the
                    # FULL row width (~3800px) with real content in maybe
                    # 9% of it. Only tightens when a mask is actually
                    # active; a column with no mask exports the full row
                    # as before (nothing to tighten to).
                    tight_crop_keep_ranges=keep_ranges if mask_active else None,
                    tight_crop_padding_px=tight_crop_padding_px,
                    tight_crop_padding_pct=tight_crop_padding_pct,
                )
            except Exception as e:
                print(f"WARNING: could not crop row {row_index} from {source_path}: {e} - skipping.")
                skipped_bad_row += 1
                continue

            if crop.mode != "RGB":
                crop = crop.convert("RGB")

            image_name = f"{uuid.uuid4().hex}.png"
            crop.save(images_dir / image_name)

            column = record.get("column", "")
            train_f.write(json.dumps({
                "image": f"images/{image_name}",
                "column": column,
                "target": target,
                "status": record.get("status"),
                "notes": record.get("notes", ""),
            }, ensure_ascii=False) + "\n")

            status_counts[record.get("status")] += 1
            column_counts[column][record.get("status")] += 1
            written += 1

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"LoRA training set export report\n{'=' * 40}\n\n")
        f.write(f"Source log: {log_path}\n")
        f.write(f"Total records in log: {len(records)}\n")
        f.write(f"Written to training set: {written}\n")
        f.write(f"Skipped (no usable target, e.g. empty 'readable' value): {skipped_no_target}\n")
        f.write(f"Skipped (sidecar/row lookup failed): {skipped_bad_row}\n\n")

        f.write("Status distribution (overall):\n")
        for status, count in status_counts.most_common():
            f.write(f"  {status}: {count}\n")
        f.write("\n")

        f.write("Per-column status breakdown:\n")
        for column in sorted(column_counts.keys()):
            f.write(f"  {column}:\n")
            for status, count in column_counts[column].most_common():
                f.write(f"    {status}: {count}\n")
        f.write("\n")

        # Explicit warning, not just a number buried in the report - a
        # training set with near-zero illegible/blank examples will not
        # teach the LoRA to abstain, regardless of how good the readable
        # examples are. This is the single most important check before
        # training, given the project's stated priority (uncertainty
        # handling over polish).
        f.write("Coverage warnings:\n")
        any_warning = False
        for status in ("illegible", "blank", "partially_readable"):
            count = status_counts.get(status, 0)
            if count < min_examples_per_status:
                f.write(f"  LOW COVERAGE: only {count} '{status}' example(s) "
                        f"(threshold: {min_examples_per_status}). The LoRA is "
                        f"unlikely to learn this behavior reliably without more.\n")
                any_warning = True
        if not any_warning:
            f.write("  None - all tracked statuses meet the minimum threshold.\n")

    print(f"Wrote {written} training examples to {train_path}")
    print(f"Images saved to {images_dir}")
    print(f"Report: {report_path}")
    if status_counts.get("illegible", 0) < min_examples_per_status or \
       status_counts.get("blank", 0) < min_examples_per_status:
        print("\nWARNING: low coverage on illegible/blank examples - see report. "
              "Training now risks a LoRA that still prefers plausible-looking "
              "guesses over honest abstention, which is the opposite of this "
              "project's goal.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", type=str, default=str(DEFAULT_LOG),
                         help=f"Path to ground_truth_log.jsonl. Default: {DEFAULT_LOG}")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT),
                         help=f"Output directory. Default: {DEFAULT_OUT}")
    parser.add_argument("--min-examples-per-status", type=int, default=15,
                         help="Minimum example count for illegible/blank/partial "
                              "statuses before the report flags a coverage warning. "
                              "Default: 15 (arbitrary starting floor, not evidence-"
                              "based - adjust once you have a sense of real class "
                              "balance in a first batch).")
    parser.add_argument("--tight-crop-padding-px", type=int, default=20,
                         help="Fixed pixel padding around each column's kept mask "
                              "range for the exported training image (default: 20). "
                              "Ignored if --tight-crop-padding-pct is given.")
    parser.add_argument("--tight-crop-padding-pct", type=float, default=None,
                         help="Padding as a fraction of the kept range's own width "
                              "(e.g. 0.1 = 10%%), instead of a fixed pixel margin - "
                              "useful when column widths vary a lot across the page. "
                              "Overrides --tight-crop-padding-px if given.")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"ERROR: log file not found: {log_path}")
        return

    export(log_path, Path(args.out_dir), args.min_examples_per_status,
           tight_crop_padding_px=args.tight_crop_padding_px,
           tight_crop_padding_pct=args.tight_crop_padding_pct)


if __name__ == "__main__":
    main()