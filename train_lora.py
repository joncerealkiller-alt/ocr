"""
LoRA training script for the single-field transcription/abstention LoRA
described in export_lora_dataset.py's docstring: teach cursive
recognition, literal transcription, and honest illegible/blank
abstention - NOT structured multi-field output.

MODULAR BY DESIGN: the base model is selected via --model, exactly like
run_row_extraction.py --model - it loads that profile's YAML from
config/models/ and dispatches to whatever loader class is registered
for it in core.loader_registry.LOADER_REGISTRY, then reuses THAT
loader's own initialize_model_and_tokenizer() to get the model and
processor. This means:
  - Preprocessing (image handling, chat template shape, dtype casting)
    is IDENTICAL to what's used at real inference time in
    core/row_extraction.py - training on anything else would let the
    LoRA learn a distribution that doesn't match how it's actually used.
  - Swapping the base model is just "--model <a different profile
    name>" - no code change here, AS LONG AS that model's loader
    exposes self.model/self.processor as HF Auto-class objects after
    initialize_model_and_tokenizer() (true for every loader currently
    registered) and its processor supports apply_chat_template() the
    same single-call way SmolVLM2/Qwen3-VL do.

WHAT'S NOT FULLY MODEL-AGNOSTIC, HONESTLY: peft's LoraConfig needs
target_modules - the attention projection layer NAMES - which genuinely
differ across model architectures. The default here
(q_proj/k_proj/v_proj/o_proj) is correct for SmolVLM2 and most Llama-
family language-model backbones, but WILL need --lora-target-modules
overridden for an architecture that names these differently. There's
no way to make this fully automatic without inspecting the specific
model's module tree - inspect via `python -c "from transformers import
AutoModelForImageTextToText; m = AutoModelForImageTextToText.from_pretrained(...); print(m)"`
and look for the attention projection layer names if a new model's
LoRA application fails to find any matching modules.

Usage (defaults tuned for a first run, per Jon's "new to this, get a
clean working run before optimizing" direction):

    python train_lora.py --model smolvlm2_2b \\
        --dataset-dir data/outputs/lora_dataset \\
        --out-dir data/outputs/lora_checkpoints

Train/val split (page-level, not row-level): a random row-level split
would let near-identical handwriting from the same page/enumerator
leak into both train and val, producing a falsely optimistic
validation signal - splits whole PAGES (by source_sidecar, added to
train.jsonl 2026-07-22) into train/val instead, so validation loss
actually reflects generalization to unseen handwriting. Falls back to
a row-level split with a printed warning if source_sidecar is missing
(older exports, or a single-page dataset where a page-level split
isn't meaningful) or if there's only one distinct page.

Requires torch, transformers, peft, pillow - NOT imported at module
level, only inside main(), so this file can still be inspected/
compile-checked on a machine without the training stack installed
(matches the same lazy-import convention used elsewhere in this
pipeline, e.g. row_segmentation_ui.py's Extract button).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path


def _load_train_records(dataset_dir: Path) -> list[dict]:
    train_path = dataset_dir / "train.jsonl"
    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Run export_lora_dataset.py first.")
        sys.exit(1)
    records = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        print(f"ERROR: {train_path} is empty.")
        sys.exit(1)
    return records


def _split_train_val(
    records: list[dict], val_fraction: float, mode: str, seed: int
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)

    if mode == "page":
        pages = sorted({r.get("source_sidecar") for r in records if r.get("source_sidecar")})
        if len(pages) < 2:
            print(
                f"WARNING: page-level split requested but only {len(pages)} distinct "
                f"page(s) found in train.jsonl (missing 'source_sidecar' on older "
                f"exports, or a genuinely single-page dataset) - falling back to a "
                f"row-level split. Validation signal will be WEAKER (handwriting "
                f"from the same page can appear in both train and val)."
            )
            mode = "random"
        else:
            rng.shuffle(pages)
            n_val_pages = max(1, round(len(pages) * val_fraction))
            val_pages = set(pages[:n_val_pages])
            train_recs = [r for r in records if r.get("source_sidecar") not in val_pages]
            val_recs = [r for r in records if r.get("source_sidecar") in val_pages]
            print(f"Page-level split: {len(val_pages)}/{len(pages)} page(s) held out "
                  f"for validation ({len(val_recs)} examples), {len(train_recs)} "
                  f"examples for training.")
            return train_recs, val_recs

    # mode == "random" (either requested directly, or fallen back to above)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction))
    val_recs = shuffled[:n_val]
    train_recs = shuffled[n_val:]
    print(f"Row-level split: {len(val_recs)} examples for validation, "
          f"{len(train_recs)} for training.")
    return train_recs, val_recs


# Deliberately separate from build_row_prompt() in core/row_extraction.py -
# that prompt asks for MULTIPLE named fields in one structured pass; this
# LoRA's whole point (per export_lora_dataset.py's docstring) is a single
# column's literal transcription with honest abstention, a different task
# shape entirely. Teaching the multi-field prompt here would train the
# LoRA on a task it will never actually be asked to do standalone.
TRANSCRIBE_PROMPT = (
    "Transcribe exactly what is written in this image. "
    "If nothing is written, respond with exactly: blank. "
    "If it is written but you genuinely cannot read it, respond with exactly: illegible. "
    "Otherwise, respond with only the text you can read - use ? for any "
    "individual characters you cannot make out."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=str, default="data/outputs/lora_dataset",
                         help="Directory containing train.jsonl + images/ from "
                              "export_lora_dataset.py.")
    parser.add_argument("--model", type=str, required=True,
                         help="Base model profile name (e.g. smolvlm2_2b) - must exist "
                              "in config/models/. Determines both the base weights AND "
                              "which loader class handles preprocessing, matching "
                              "real inference exactly.")
    parser.add_argument("--out-dir", type=str, default="data/outputs/lora_checkpoints",
                         help="Where to save a LoRA adapter checkpoint after each epoch.")
    parser.add_argument("--epochs", type=int, default=4,
                         help="Default: 4. Small dataset, small model - a first run "
                              "shouldn't need more; compare checkpoints across epochs "
                              "afterward rather than guessing the right number upfront.")
    parser.add_argument("--lr", type=float, default=2e-4,
                         help="Default: 2e-4, a standard starting point for LoRA "
                              "(much higher than a full fine-tune's typical 1e-5-1e-4, "
                              "since only a small fraction of parameters are trainable).")
    parser.add_argument("--lora-rank", type=int, default=8,
                         help="Default: 8. Small dataset (hundreds, not tens of "
                              "thousands, of examples) - a higher rank mostly just "
                              "overfits faster without more real signal to justify it.")
    parser.add_argument("--lora-alpha", type=int, default=16,
                         help="Default: 16 (2x rank, a common convention).")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", type=str,
                         default="q_proj,k_proj,v_proj,o_proj",
                         help="Comma-separated attention projection layer names to "
                              "apply LoRA to. Default matches SmolVLM2/Llama-family "
                              "backbones - see this file's module docstring if "
                              "switching to a different model architecture.")
    parser.add_argument("--grad-accum-steps", type=int, default=8,
                         help="Effective batch size = this value (physical batch is "
                              "always 1 example - see module docstring for why: crops "
                              "have different widths per column, and padding a batched "
                              "image tensor across HF processor versions is a real "
                              "source of subtle bugs not worth risking on a first run).")
    parser.add_argument("--val-split-mode", type=str, choices=["page", "random"], default="page",
                         help="Default: page - see module docstring for why a row-level "
                              "split is a weaker validation signal.")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=None,
                         help="Cap the TRAINING set to this many examples (val set "
                              "unaffected) - useful for a quick smoke-test run before "
                              "committing to the full dataset.")
    parser.add_argument("--skip-generation-eval", action="store_true",
                         help="Skip the end-of-training exact-match accuracy check "
                              "(runs model.generate() over the full val set, slower "
                              "than the per-epoch loss-only check). Loss is still "
                              "computed every epoch regardless of this flag.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _load_train_records(dataset_dir)
    if args.max_examples:
        # Cap applied AFTER split (below), to the train side only - see
        # --max-examples help text.
        pass

    train_records, val_records = _split_train_val(
        records, args.val_fraction, args.val_split_mode, args.seed)
    if args.max_examples and len(train_records) > args.max_examples:
        rng = random.Random(args.seed)
        train_records = rng.sample(train_records, args.max_examples)
        print(f"--max-examples: capped training set to {len(train_records)} examples.")

    # --- heavy imports, lazy (see module docstring) ---
    try:
        import torch
        from PIL import Image
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        print(f"ERROR: missing training dependency ({e}). This script requires "
              f"torch, transformers, peft, and pillow installed.")
        sys.exit(1)

    from core.loaders.base_loader import load_model_config
    from core.loader_registry import LOADER_REGISTRY

    config = load_model_config(args.model)
    loader_cls = LOADER_REGISTRY.get(config.loader_class)
    if loader_cls is None:
        print(f"ERROR: no loader registered for {config.loader_class!r}.")
        sys.exit(1)

    print(f"Loading base model {args.model!r} ({config.repo_id}) via {config.loader_class}...")
    loader = loader_cls(config)
    loader.initialize_model_and_tokenizer()
    model = loader.model
    processor = loader.processor
    model.train()

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[m.strip() for m in args.lora_target_modules.split(",") if m.strip()],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    def build_example(record: dict) -> dict | None:
        """
        Loads one image+target pair and tokenizes it as a full user+
        assistant turn (target INCLUDED, add_generation_prompt=False),
        then separately tokenizes just the user turn (add_generation_
        prompt=True) to find where the assistant's response starts -
        everything before that is masked out of the loss (label=-100),
        so the model is only ever supervised on producing the target
        text, not on reproducing the prompt/image tokens back.
        """
        image_path = dataset_dir / record["image"]
        if not image_path.exists():
            print(f"WARNING: {image_path} missing, skipping.")
            return None
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        target = record["target"]

        user_msg = {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": TRANSCRIBE_PROMPT},
            ],
        }
        full_messages = [user_msg, {"role": "assistant", "content": target}]

        prompt_only = processor.apply_chat_template(
            [user_msg], add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        )
        full = processor.apply_chat_template(
            full_messages, add_generation_prompt=False, tokenize=True,
            return_dict=True, return_tensors="pt",
        )

        prompt_len = prompt_only["input_ids"].shape[1]
        labels = full["input_ids"].clone()
        labels[:, :prompt_len] = -100

        full = {k: v.to(model.device) for k, v in full.items()}
        labels = labels.to(model.device)
        dtype = getattr(loader, "_model_dtype", None)
        if dtype is not None and "pixel_values" in full:
            full["pixel_values"] = full["pixel_values"].to(dtype)
        return {**full, "labels": labels}

    def run_val_loss(val_recs: list[dict]) -> float:
        model.eval()
        total_loss, n = 0.0, 0
        with torch.no_grad():
            for rec in val_recs:
                ex = build_example(rec)
                if ex is None:
                    continue
                out = model(**ex)
                total_loss += out.loss.item()
                n += 1
        model.train()
        return total_loss / n if n else float("nan")

    def run_val_accuracy(val_recs: list[dict]) -> dict:
        """
        End-of-training only (see --skip-generation-eval): actually
        GENERATES from each val image and checks exact string match
        (case/whitespace-normalized) against the ground-truth target -
        a far more meaningful signal for THIS task (literal
        transcription + abstention) than perplexity/loss alone, since
        loss can look fine while the model still isn't producing
        usable exact output. Reports accuracy split by status
        (readable/partial/illegible/blank) so it's visible whether
        abstention specifically is being learned, not just masked by
        the majority-class readable examples.
        """
        model.eval()
        by_status = {}
        with torch.no_grad():
            for rec in val_recs:
                image_path = dataset_dir / rec["image"]
                if not image_path.exists():
                    continue
                image = Image.open(image_path)
                if image.mode != "RGB":
                    image = image.convert("RGB")
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
                dtype = getattr(loader, "_model_dtype", None)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                if dtype is not None and "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

                generated = model.generate(**inputs, max_new_tokens=64, do_sample=False)
                trimmed = generated[0][inputs["input_ids"].shape[1]:]
                predicted = processor.decode(trimmed, skip_special_tokens=True).strip()

                expected = rec["target"].strip()
                status = rec.get("status", "unknown")
                bucket = by_status.setdefault(status, {"correct": 0, "total": 0})
                bucket["total"] += 1
                if predicted.strip().lower() == expected.lower():
                    bucket["correct"] += 1
        model.train()
        return by_status

    print(f"\nTraining: {len(train_records)} examples, {args.epochs} epochs, "
          f"effective batch size {args.grad_accum_steps} (grad accumulation).\n")

    rng = random.Random(args.seed)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        order = list(train_records)
        rng.shuffle(order)

        running_loss = 0.0
        step_count = 0
        optimizer.zero_grad()
        for i, rec in enumerate(order):
            ex = build_example(rec)
            if ex is None:
                continue
            out = model(**ex)
            loss = out.loss / args.grad_accum_steps
            loss.backward()
            running_loss += out.loss.item()
            step_count += 1

            if (i + 1) % args.grad_accum_steps == 0 or (i + 1) == len(order):
                optimizer.step()
                optimizer.zero_grad()

        avg_train_loss = running_loss / step_count if step_count else float("nan")
        val_loss = run_val_loss(val_records) if val_records else float("nan")
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch}/{args.epochs}: train_loss={avg_train_loss:.4f} "
              f"val_loss={val_loss:.4f} ({elapsed:.0f}s)")

        checkpoint_dir = out_dir / f"epoch_{epoch}"
        model.save_pretrained(str(checkpoint_dir))
        print(f"  Saved adapter: {checkpoint_dir}")

    if val_records and not args.skip_generation_eval:
        print("\nRunning end-of-training generation accuracy check on validation set...")
        by_status = run_val_accuracy(val_records)
        print("\nValidation exact-match accuracy by status:")
        for status, counts in sorted(by_status.items()):
            acc = counts["correct"] / counts["total"] if counts["total"] else 0.0
            print(f"  {status}: {counts['correct']}/{counts['total']} ({acc:.0%})")

    print(f"\nDone. Adapter checkpoints saved under {out_dir}/epoch_N/. "
          f"Compare val_loss and (if run) the accuracy breakdown across epochs to "
          f"pick the best one rather than assuming the last epoch is best.")


if __name__ == "__main__":
    main()
