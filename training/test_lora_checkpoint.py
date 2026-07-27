"""
Standalone tester for a LoRA adapter checkpoint (data/outputs/
lora_checkpoints/epoch_N/) OR the bare base model (omit --checkpoint) -
loads the base model exactly the way train_lora.py does (same loader,
same bf16 cast), optionally applies an adapter, and runs real generation
over the SAME held-out validation split train_lora.py used for that run
(same seed/val_fraction/split mode, reconstructed deterministically via
train_lora.py's own _split_train_val()) - so results reflect genuine
held-out performance, not accidental train-set leakage from picking a
different random split.

Usage:
    python training/test_lora_checkpoint.py --checkpoint data/outputs/lora_checkpoints/epoch_1
    python training/test_lora_checkpoint.py               # base model, no adapter

Reports TWO separate things (2026-07-26, per Jon's direction - exact-
match alone conflates "wrong transcription" with "didn't even attempt
abstention correctly", two different failure modes):

  1. Exact-match accuracy (case/whitespace-normalized) split by status -
     same metric train_lora.py's own end-of-training run_val_accuracy()
     uses, so a number from THIS script is directly comparable to one
     from the log.

  2. Abstention behavior - separate from exact-match. The training
     prompt (TRANSCRIBE_PROMPT below) asks for the literal words "blank"
     /"illegible" on those statuses, but a model that just falls silent
     (empty output) on an illegible/blank example is ALSO getting the
     important part right (not hallucinating a fake value) even if it
     hasn't yet learned the exact expected word - counted here as a
     correct abstention. The mirror-image failure - falling silent on a
     READABLE example, where a real answer exists and abstaining is
     wrong - is counted separately as "incorrectly abstained".
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Windows' console/redirected-file default codepage (cp1252) can't
# represent arbitrary model output - a hallucinated foreign-script
# character or stray BPE token is completely normal, especially from an
# untrained base model with no adapter, and crashed a real run right
# before it printed its summary (2026-07-26). Reconfigured here,
# unconditionally, rather than relying on remembering to set
# PYTHONIOENCODING=utf-8 every time this script is invoked.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Statuses where abstaining (empty output, or the literal expected word)
# IS the desired behavior, vs. statuses where a real transcription is
# expected and falling silent is itself a failure.
ABSTAIN_STATUSES = {"illegible", "blank"}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Path to a saved adapter dir, e.g. data/outputs/lora_checkpoints/"
                              "epoch_1. Omit to test the bare BASE MODEL with no adapter at all "
                              "(useful as a baseline: does the untrained model already produce "
                              "output with this prompt, or is silence a training-induced thing).")
    parser.add_argument("--model", type=str, default="smolvlm2_2b",
                         help="Must match the base model the checkpoint was trained from.")
    parser.add_argument("--dataset-dir", type=str, default="data/outputs/lora_dataset")
    parser.add_argument("--val-fraction", type=float, default=0.15,
                         help="Must match the value used at training time to reconstruct "
                              "the same split.")
    parser.add_argument("--val-split-mode", type=str, choices=["page", "random"], default="page",
                         help="Must match the value used at training time.")
    parser.add_argument("--seed", type=int, default=42,
                         help="Must match the value used at training time.")
    parser.add_argument("--num-examples", type=int, default=None,
                         help="Randomly sample this many val examples instead of testing "
                              "all of them (faster spot-check).")
    parser.add_argument("--show-all", action="store_true",
                         help="Print every prediction, not just mismatches.")
    args = parser.parse_args()

    import torch
    from PIL import Image
    from peft import PeftModel

    from core.loaders.base_loader import load_model_config
    from core.loader_registry import LOADER_REGISTRY
    from train_lora import _load_train_records, _split_train_val, TRANSCRIBE_PROMPT

    print("Prompt used for every example:")
    print(f"  {TRANSCRIBE_PROMPT!r}\n")

    dataset_dir = Path(args.dataset_dir)
    records = _load_train_records(dataset_dir)
    _, val_records = _split_train_val(records, args.val_fraction, args.val_split_mode, args.seed)
    if args.num_examples and len(val_records) > args.num_examples:
        rng = random.Random(args.seed)
        val_records = rng.sample(val_records, args.num_examples)
    print(f"Testing on {len(val_records)} held-out validation example(s).\n")

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

    if args.checkpoint:
        print(f"Loading adapter from {args.checkpoint!r}...")
        model = PeftModel.from_pretrained(model, args.checkpoint)
        label = args.checkpoint
    else:
        print("No --checkpoint given - testing the BARE BASE MODEL (no adapter).")
        label = f"{args.model} (base, no adapter)"
    model.eval()

    by_status: dict[str, dict[str, int]] = {}
    abstain_stats: dict[str, dict[str, int]] = {}  # per-status: total, abstained
    mismatches: list[tuple[str, str, str, str]] = []

    with torch.no_grad():
        for i, rec in enumerate(val_records):
            image_path = dataset_dir / rec["image"]
            if not image_path.exists():
                print(f"WARNING: {image_path} missing, skipping.")
                continue
            image = Image.open(image_path).convert("RGB")

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
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

            expected = rec["target"].strip()
            status = rec.get("status", "unknown")
            column = rec.get("column", "?")

            bucket = by_status.setdefault(status, {"correct": 0, "total": 0})
            bucket["total"] += 1
            correct = predicted.strip().lower() == expected.lower()
            if correct:
                bucket["correct"] += 1
            else:
                mismatches.append((column, status, expected, predicted))

            abstained = predicted == ""
            a_bucket = abstain_stats.setdefault(status, {"total": 0, "abstained": 0})
            a_bucket["total"] += 1
            if abstained:
                a_bucket["abstained"] += 1

            if args.show_all or not correct:
                mark = "OK" if correct else "XX"
                abstain_note = " [abstained]" if abstained else ""
                print(f"[{mark}] ({status:20s} {column:22s}) expected={expected!r} "
                      f"predicted={predicted!r}{abstain_note}")

    total_correct = sum(b["correct"] for b in by_status.values())
    total = sum(b["total"] for b in by_status.values())
    print(f"\n=== Exact-match results: {label} ===")
    print(f"Overall: {total_correct}/{total} ({total_correct/total:.0%})" if total else "No examples tested.")
    for status, counts in sorted(by_status.items()):
        acc = counts["correct"] / counts["total"] if counts["total"] else 0.0
        print(f"  {status}: {counts['correct']}/{counts['total']} ({acc:.0%})")

    print(f"\n=== Abstention behavior: {label} ===")
    print("(empty output on illegible/blank = correct abstention; "
          "empty output on readable/partially_readable = incorrectly silent)")
    for status, counts in sorted(abstain_stats.items()):
        rate = counts["abstained"] / counts["total"] if counts["total"] else 0.0
        if status in ABSTAIN_STATUSES:
            print(f"  {status} (should abstain):      {counts['abstained']}/{counts['total']} "
                  f"abstained ({rate:.0%})")
        else:
            print(f"  {status} (should NOT abstain): {counts['abstained']}/{counts['total']} "
                  f"incorrectly silent ({rate:.0%})")

    if mismatches and not args.show_all:
        print(f"\n{len(mismatches)} mismatch(es) shown above with [XX] "
              f"(pass --show-all to also see correct predictions).")


if __name__ == "__main__":
    main()
