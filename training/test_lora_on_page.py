"""
Runs a trained LoRA transcription checkpoint against EVERY (row, column)
field of a given sidecar - unlike test_lora_checkpoint.py (which only
evaluates against the ground_truth_log.jsonl-derived held-out val
split), this works on ANY sidecar, including a page that was never
ground-truth labeled at all, by cropping fields directly the same way
export_lora_dataset.py does (same per-column mask, tight-crop, upscale)
so the model sees the same kind of input it was actually trained on.
Uses the SAME TRANSCRIBE_PROMPT train_lora.py trained against - this is
NOT the two-stage extraction pipeline's OCR/structuring prompts, which
this adapter was never trained on.

No ground-truth comparison is possible for an unlabeled page - this
only reports what the model actually predicts, for a human to review
directly.

Usage:
    python training/test_lora_on_page.py \\
        --sidecar data/outputs/row_segmentation/z000017634_dewarped_sidecar.json \\
        --checkpoint data/outputs/lora_checkpoints/epoch_2 \\
        --out data/outputs/lora_checkpoints/epoch_2/inference_z000017634
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sidecar", type=str, required=True,
                         help="Path to a segmentation sidecar JSON (core/row_segmentation.py "
                              "output) - every column must already have a saved mask "
                              "(status: done), same requirement run_two_stage_extraction() has.")
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to a saved LoRA adapter dir, e.g. "
                              "data/outputs/lora_checkpoints/epoch_2")
    parser.add_argument("--model", type=str, default="smolvlm2_2b",
                         help="Must match the base model the checkpoint was trained from.")
    parser.add_argument("--out", type=str, required=True,
                         help="Output directory - writes results.json and results.csv here.")
    parser.add_argument("--tight-crop-padding-px", type=int, default=20,
                         help="Must match export_lora_dataset.py's value for the training "
                              "distribution to actually line up (default: 20).")
    parser.add_argument("--upscale-target-height", type=int, default=160,
                         help="Must match export_lora_dataset.py's value (default: 160).")
    parser.add_argument("--upscale-max-width", type=int, default=4096)
    parser.add_argument("--max-rows", type=int, default=None,
                         help="Limit to the first N rows - useful for a quick spot check.")
    args = parser.parse_args()

    import torch
    from PIL import Image
    from peft import PeftModel

    from core.row_segmentation import load_sidecar, crop_region_from_source, compute_exclude_ranges
    from core.loaders.base_loader import load_model_config
    from core.loader_registry import LOADER_REGISTRY
    from train_lora import TRANSCRIBE_PROMPT

    sidecar_path = Path(args.sidecar)
    if not sidecar_path.exists():
        print(f"ERROR: sidecar not found: {sidecar_path}")
        sys.exit(1)
    sidecar = load_sidecar(sidecar_path)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_dir():
        print(f"ERROR: checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    columns = sidecar.get("column_order") or list(sidecar.get("columns", {}).keys())
    missing_masks = [c for c in columns if not sidecar.get("columns", {}).get(c, {}).get("mask_keep_ranges")]
    if missing_masks:
        print(f"ERROR: these columns have no saved mask in this sidecar: {missing_masks} - "
              f"mask them in ui/row_segmentation_ui.py first.")
        sys.exit(1)

    rows = sidecar["rows"]
    if args.max_rows:
        rows = rows[: args.max_rows]
    print(f"Sidecar: {sidecar_path.name}")
    print(f"Source image: {sidecar['source_image_path']}")
    print(f"Columns: {', '.join(columns)}")
    print(f"Rows: {len(rows)}")
    print(f"Checkpoint: {checkpoint_path}")
    print()

    config = load_model_config(args.model)
    loader_cls = LOADER_REGISTRY.get(config.loader_class)
    if loader_cls is None:
        print(f"ERROR: no loader registered for {config.loader_class!r}.")
        sys.exit(1)

    print(f"Loading base model {args.model!r} ({config.repo_id})...")
    loader = loader_cls(config)
    loader.initialize_model_and_tokenizer()
    model = loader.model
    processor = loader.processor

    # Same bf16 cast train_lora.py applies - see that file's own comment
    # for why "auto" dtype resolution isn't trustworthy on this hardware.
    if next(model.parameters()).dtype == torch.float32:
        model = model.to(torch.bfloat16)
    model_dtype = next(model.parameters()).dtype

    print(f"Loading adapter from {checkpoint_path}...")
    model = PeftModel.from_pretrained(model, str(checkpoint_path))
    model.eval()

    source_path = sidecar["source_image_path"]
    deskew_angle = sidecar["deskew_angle"]
    width = sidecar["deskewed_image_size"][0]

    results = []
    errors = []
    total = len(rows) * len(columns)
    done = 0

    with torch.no_grad():
        for row in rows:
            for column in columns:
                done += 1
                col_state = sidecar["columns"][column]
                keep_ranges = [tuple(k) for k in col_state.get("mask_keep_ranges", [])]
                apply_rows = col_state.get("mask_apply_rows", True)
                mask_active = bool(keep_ranges) and apply_rows
                row_masks = compute_exclude_ranges(keep_ranges, width) if mask_active else []

                try:
                    crop = crop_region_from_source(
                        source_path, row["bbox"], deskew_angle, row_masks,
                        tight_crop_keep_ranges=keep_ranges if mask_active else None,
                        tight_crop_padding_px=args.tight_crop_padding_px,
                        upscale_target_height=args.upscale_target_height,
                        upscale_max_width=args.upscale_max_width,
                    )
                except Exception as e:
                    errors.append({"row_index": row["index"], "column": column, "error": str(e)})
                    continue
                if crop.mode != "RGB":
                    crop = crop.convert("RGB")

                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": crop},
                        {"type": "text", "text": TRANSCRIBE_PROMPT},
                    ],
                }]
                inputs = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt",
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)

                generated = model.generate(**inputs, max_new_tokens=64, do_sample=False)
                trimmed = generated[0][inputs["input_ids"].shape[1]:]
                predicted = processor.decode(trimmed, skip_special_tokens=True).strip()

                results.append({
                    "row_index": row["index"], "column": column, "predicted": predicted,
                })
                if done % 25 == 0 or done == total:
                    print(f"  {done}/{total} fields processed...")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"sidecar": str(sidecar_path), "checkpoint": str(checkpoint_path),
                   "model": args.model, "results": results, "errors": errors}, f,
                  indent=2, ensure_ascii=False)

    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["row_index", "column", "predicted"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\nDone. {len(results)} fields predicted, {len(errors)} error(s).")
    print(f"Results: {out_dir / 'results.json'} and {out_dir / 'results.csv'}")
    if errors:
        print("Errors:")
        for e in errors[:10]:
            print(f"  row {e['row_index']} / {e['column']}: {e['error']}")


if __name__ == "__main__":
    main()
