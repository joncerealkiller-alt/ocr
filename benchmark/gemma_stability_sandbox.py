"""
Gemma decision-stability audit (2026-08-04) - reclassifies the flagged
disagreement set TWICE, independently, under byte-identical conditions
(same taxonomy, same rendered prompt, same model, same inference
parameters, no prompt modifications, no retries), to determine whether
disagreement cases represent stable-but-difficult decisions or
intrinsically unstable ones.

FULLY ISOLATED SANDBOX: reuses build_classifier_loader() (config/model
loading, read-only) and the loader's own _build_prompt()/
_parse_classification() (the exact production prompt-construction and
output-validation logic, unmodified) - but NEVER calls run(),
open_bucket_writers(), or PipelineDatabase. All output goes to
data/outputs/gemma_stability_sandbox/ only. Verified clean by a
before/after checksum diff of pipeline.db and every bucket CSV, same
verification pattern as benchmark/gemma_raw_sandbox_reclassification.py
earlier this session.

IMPORTANT CORRECTION vs. earlier informal characterization this
session: config/models/gemma.yaml sets do_sample: false (greedy
decoding) - temperature/top_p/top_k are configured but never read
(_run_generate()'s `if self.config.do_sample:` gate never executes).
If this run finds real category instability, it is NOT explained by
sampling temperature - the candidate explanation is GPU floating-point
non-determinism in the forward pass itself (non-deterministic reduction
order in some CUDA kernels without torch.use_deterministic_algorithms),
not stochastic decoding. See docs/GEMMA_STABILITY_AUDIT.md's [F]
section.

Candidate set (union, 683 images): tower<->Gemma disagreement (657,
data/outputs/tower_consensus_audit/20260804T132128Z/disagreements.csv),
mechanically-flagged contradictory-reasoning cases (21,
data/outputs/gemma_reasoning_consistency_audit/contradictory_cases.csv),
human-reviewed cases where the ground-truth label differs from the
CURRENT fresh-run bucket (8), and the 93 checkpoint<->fresh
classification changes from earlier this session.

Pre-reasoning confidence (2026-08-04, engineered for this experiment -
no existing production code exposes this): output_scores=True/
return_dict_in_generate=True added to a LOCAL copy of _run_generate()'s
generate() call (never touches core/loaders/gemma_loader.py) - this
only changes what generate() RETURNS, not what it generates (verified:
same do_sample=False path, same gen_kwargs, same charset logits
processor), so "no prompt modifications" is preserved. Locates the
generation step immediately after "category:" appears in the token-by-
token decode and reports that step's softmax distribution (top-1 prob,
top-2 margin, entropy). Feasibility-tested against 4 real images before
committing to the full run - found a genuine but narrow-range signal
(0.998-1.0 top-1 prob even on reasoning-flagged cases), reported
honestly, not oversold.

Usage:
    python -m benchmark.gemma_stability_sandbox
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "gemma_stability_sandbox"
CANDIDATES_PATH = OUTPUT_DIR / "candidate_set.json"


def build_candidate_set() -> list[str]:
    with open(PROJECT_ROOT / "data" / "outputs" / "tower_consensus_audit" / "20260804T132128Z" / "disagreements.csv",
              newline="", encoding="utf-8") as f:
        tower_disagree = {row["working_path"] for row in csv.DictReader(f)}

    with open(PROJECT_ROOT / "data" / "outputs" / "gemma_reasoning_consistency_audit" / "contradictory_cases.csv",
              newline="", encoding="utf-8") as f:
        contradictory = {row["file_path"] for row in csv.DictReader(f)}

    ALL_BUCKETS = ["dense_tabular_rows", "genealogy_chart", "handwritten_ledger", "map_land_record",
                   "printed_document", "mixed_text_image", "portrait_photo", "website_screenshot",
                   "photo_collage", "casual_photo", "cemetery_photo"]
    fresh_bucket = {}
    for b in ALL_BUCKETS:
        path = PROJECT_ROOT / "data" / "buckets" / f"{b}.csv"
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    fresh_bucket[row["file_path"]] = b

    with open(PROJECT_ROOT / "data" / "outputs" / "reference_pipeline_v3" / "misclassifications.csv",
              newline="", encoding="utf-8") as f:
        human_rows = list(csv.DictReader(f))
    human_disagree = set()
    for row in human_rows:
        fp = row["file_path"]
        label = row.get("correct_category", "").strip()
        if not label or label in ("ignored", "needs_new_bucket", "bad_deskew"):
            continue
        if fresh_bucket.get(fp) and fresh_bucket[fp] != label:
            human_disagree.add(fp)

    with open(PROJECT_ROOT / "data" / "outputs" / "fresh_pass_vs_checkpoint_comparison" / "changed_classifications.csv",
              newline="", encoding="utf-8") as f:
        changed_93_basenames = {row["image"] for row in csv.DictReader(f)}

    union_paths = tower_disagree | contradictory | human_disagree
    union_basenames = {Path(p).name: p for p in union_paths}
    # add the 93-changes set by basename, resolving to a real working_path
    working_dir = PROJECT_ROOT / "data" / "working"
    for name in changed_93_basenames:
        if name not in union_basenames:
            candidate = working_dir / name
            if candidate.exists():
                union_basenames[name] = str(candidate)

    return sorted(union_basenames.values())


@torch.no_grad()
def _generate_with_scores(loader, raw_image: Image.Image) -> tuple[str, dict | None]:
    """Local copy of GemmaLoader._run_generate()'s logic, with
    output_scores=True/return_dict_in_generate=True added - does not
    modify core/loaders/gemma_loader.py. Returns (raw_output_text,
    logit_stats_or_None)."""
    if raw_image.mode != "RGB":
        raw_image = raw_image.convert("RGB")

    system_content = loader.config.extra.get("system_prompt", "")
    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": raw_image},
            {"type": "text", "text": loader._build_prompt(task="classify")},
        ],
    })
    image_token_budget = loader.config.extra.get("image_token_budget", 140)
    text = loader.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=loader.config.reasoning_enabled,
    )
    inputs = loader.processor(
        text=text, images=raw_image, image_seq_length=image_token_budget, return_tensors="pt",
    ).to(loader.model.device)
    input_len = inputs["input_ids"].shape[-1]

    gen_kwargs = dict(max_new_tokens=loader.config.max_new_tokens, do_sample=loader.config.do_sample)
    if loader.config.do_sample:
        gen_kwargs.update(temperature=loader.config.temperature, top_p=loader.config.top_p, top_k=loader.config.top_k)
    if loader.config.repetition_penalty and loader.config.repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = loader.config.repetition_penalty
    if loader.config.no_repeat_ngram_size:
        gen_kwargs["no_repeat_ngram_size"] = loader.config.no_repeat_ngram_size
    loader._maybe_add_charset_logits_processor(gen_kwargs)

    with torch.inference_mode():
        outputs = loader.model.generate(
            **inputs, **gen_kwargs, output_scores=True, return_dict_in_generate=True,
        )

    sequence = outputs.sequences[0][input_len:]
    scores = outputs.scores
    response = loader.processor.decode(sequence, skip_special_tokens=False)
    parsed = loader.processor.parse_response(response)
    final_text = parsed.get("content", parsed) if isinstance(parsed, dict) else parsed
    raw_output = str(final_text).strip()

    logit_stats = None
    running_text = ""
    category_step = None
    for i, tok_id in enumerate(sequence):
        running_text += loader.processor.decode([tok_id], skip_special_tokens=False)
        if "category:" in running_text.lower():
            category_step = i + 1
            break
    if category_step is not None and category_step < len(scores):
        step_scores = scores[category_step][0]
        probs = F.softmax(step_scores.float(), dim=-1)
        top_probs, _ = torch.topk(probs, 2)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum().item()
        logit_stats = {
            "top1_prob": round(top_probs[0].item(), 6),
            "top2_prob": round(top_probs[1].item(), 6),
            "margin": round((top_probs[0] - top_probs[1]).item(), 6),
            "entropy": round(entropy, 6),
        }

    return raw_output, logit_stats


def _classify_one(loader, file_path: str) -> dict:
    with Image.open(file_path) as img:
        raw_image = img.convert("RGB")
    try:
        raw_output, logit_stats = _generate_with_scores(loader, raw_image)
        result = loader._parse_classification(file_path, raw_output)
        return {
            "file_path": file_path,
            "category": result.category.value,
            "confidence": result.confidence,
            "reason": result.reason,
            "error": None,
            "logit_stats": logit_stats,
        }
    except Exception as e:
        return {
            "file_path": file_path, "category": None, "confidence": None,
            "reason": None, "error": f"{type(e).__name__}: {e}", "logit_stats": None,
        }


def run_pass(loader, candidates: list[str], pass_name: str) -> list[dict]:
    results = []
    for i, fp in enumerate(candidates, 1):
        r = _classify_one(loader, fp)
        results.append(r)
        print(f"[{pass_name} {i}/{len(candidates)}] {Path(fp).name} -> {r['category']} "
              f"(conf={r['confidence']}, error={r['error']})")
        if i % 25 == 0:
            with open(OUTPUT_DIR / f"{pass_name}.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
    with open(OUTPUT_DIR / f"{pass_name}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = build_candidate_set()
    print(f"Candidate set: {len(candidates)} images")
    with open(CANDIDATES_PATH, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)

    from core.classifier import load_pipeline_config, build_classifier_loader
    pipeline_cfg = load_pipeline_config()
    loader = build_classifier_loader(pipeline_cfg)

    print("\n=== PASS 1 ===")
    run_pass(loader, candidates, "pass1")

    print("\n=== PASS 2 (independent second observation) ===")
    run_pass(loader, candidates, "pass2")

    print("\nDone. All output under", OUTPUT_DIR)


if __name__ == "__main__":
    main()
