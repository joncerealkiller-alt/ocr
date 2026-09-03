"""
Logit-confidence test at real scale, against the pre-automation, whole-
corpus human-reviewed ground truth (`data/outputs/reference_pipeline_v2/
manifest_final.csv`, 1975 rows, found and confirmed 2026-08-06) - per
Jon's direction: use this larger, real ground truth set instead of the
48-case disagreement-only sample, which had ZERO real `dense_tabular_
rows` positives (only negatives) and skewed toward hard/ambiguous cases.

IMPORTANT TAXONOMY-VERSION CAVEAT (Jon's own point): this manifest
predates subtypes (census/passenger_manifest) being added under
`dense_tabular_rows` - it only has real ground truth at the TOP-LEVEL
category. `dense_tabular_rows` itself is UNCHANGED between old and new
taxonomy (only its subtypes are new), so a gate asking "is this
dense_tabular_rows" (not "is this census specifically") is directly and
validly testable against this old data - this is why this run targets
the top-level category, not GATE_CENSUS_PROMPT/GATE_MANIFEST_PROMPT
(which ask about a distinction this ground truth cannot verify).

GATE_DENSE_TABULAR_PROMPT is built from the REAL, current classifier_
guidance text in config/taxonomy.yaml (core/taxonomy.py's
`Category.classifier_guidance` for `dense_tabular_rows`), not
reworded/guessed - same "reuse the tuned prompt text" discipline as
the rest of this session's gated-prompt work.

Sample: ALL 168 real dense_tabular_rows positives still on disk, plus
200 randomly-sampled negatives (any other real category, excluding the
`uncertain_review` sentinel) also still on disk - the first real test
of this gate mechanism with TRUE POSITIVES available at all, which the
48-case disagreement set never had.

Usage:
    python diagnostics/test_gemma_logit_confidence_old_taxonomy.py
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image

from diagnostics.test_gemma_logit_confidence import (
    load_gemma, build_inputs, find_decision_token_index,
)
from core.taxonomy import load_taxonomy

MANIFEST_PATH = PROJECT_ROOT / "data" / "outputs" / "reference_pipeline_v2" / "manifest_final.csv"
MAX_NEW_TOKENS = 40
N_NEGATIVES = 200
RANDOM_SEED = 42
REPORT_PATH = PROJECT_ROOT / "data" / "logs" / "reviewed" / "logit_confidence_old_taxonomy_report.txt"


def build_gate_prompt() -> str:
    taxonomy = load_taxonomy()
    guidance = taxonomy.get_category("dense_tabular_rows").classifier_guidance
    return f"""Look ONLY at this question: is this image a "dense tabular rows" document?

Definition (this project's real classifier guidance, used as-is): {guidance}

Do not consider any other category - answer only this one question.

Output EXACTLY these two lines, nothing else:
is_dense_tabular_rows: <yes|no>
confidence: <0.0-1.0>"""


def load_sample() -> list[dict]:
    with open(MANIFEST_PATH, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    positives = [r for r in rows if r["category"] == "dense_tabular_rows" and Path(r["file_path"]).exists()]
    negatives_pool = [r for r in rows if r["category"] not in ("dense_tabular_rows", "uncertain_review")
                       and Path(r["file_path"]).exists()]
    rng = random.Random(RANDOM_SEED)
    negatives = rng.sample(negatives_pool, min(N_NEGATIVES, len(negatives_pool)))
    for r in positives:
        r["ground_truth_is_dtr"] = True
    for r in negatives:
        r["ground_truth_is_dtr"] = False
    combined = positives + negatives
    rng.shuffle(combined)
    return combined


def main():
    prompt = build_gate_prompt()
    loader = load_gemma()
    system_content = "You are a strict, non-interpretive archival routing classifier."
    gen_kwargs = dict(do_sample=False, use_cache=True,
                       return_dict_in_generate=True, output_scores=True,
                       max_new_tokens=MAX_NEW_TOKENS)
    loader._maybe_add_charset_logits_processor(gen_kwargs)

    sample = load_sample()
    n_pos = sum(1 for r in sample if r["ground_truth_is_dtr"])
    n_neg = len(sample) - n_pos

    lines = []
    def log(s=""):
        lines.append(s)
        print(s)

    log(f"Gate prompt:\n{prompt}\n")
    log(f"Sample: {len(sample)} images ({n_pos} real dense_tabular_rows positives, "
        f"{n_neg} negatives from other categories)\n")

    results = []
    for i, row in enumerate(sample, 1):
        file_path = row["file_path"]
        p = Path(file_path)
        try:
            image = Image.open(p).convert("RGB")
        except Exception as e:
            log(f"[{i}/{len(sample)}] SKIP {file_path}: {type(e).__name__}: {e}")
            continue
        inputs = build_inputs(loader, image, prompt, system_content)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            out = loader.model.generate(**inputs, **gen_kwargs)
        generated = out.sequences[0][input_len:]
        decision_idx = find_decision_token_index(loader, generated)

        gt = row["ground_truth_is_dtr"]
        gt_str = "yes" if gt else "no"

        if decision_idx is None:
            log(f"[{i}/{len(sample)}] {row['category']:<20s} gt={gt_str:>3s}  NOT FOUND")
            results.append({"category": row["category"], "gt": gt, "found": False})
            del image
            continue

        decision_token_id = generated[decision_idx].item()
        decision_text = loader.processor.decode([decision_token_id], skip_special_tokens=True).strip().lower()
        logits = out.scores[decision_idx][0]
        probs = torch.softmax(logits, dim=-1)
        own_prob = probs[decision_token_id].item()
        topk = torch.topk(logits, k=2)
        logit_gap = (topk.values[0] - topk.values[1]).item()

        predicted = decision_text == "yes"
        correct = predicted == gt
        mark = "OK  " if correct else "WRONG"
        log(f"[{i}/{len(sample)}] {mark} {row['category']:<20s} gt={gt_str:>3s} "
            f"pred={decision_text:>3s}  P={own_prob:.6f}  gap={logit_gap:7.3f}")

        results.append({
            "category": row["category"], "gt": gt, "found": True,
            "decision_text": decision_text, "own_prob": own_prob,
            "logit_gap": logit_gap, "correct": correct,
        })
        del image
        torch.cuda.empty_cache()

        if i % 25 == 0:
            REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    found = [r for r in results if r["found"]]
    correct_cases = [r for r in found if r["correct"]]
    wrong_cases = [r for r in found if not r["correct"]]

    log(f"\n\n=== SUMMARY ({len(found)}/{len(sample)} decision tokens found) ===")
    log(f"Overall accuracy: {len(correct_cases)}/{len(found)} ({100*len(correct_cases)/len(found):.1f}%)")

    tp = sum(1 for r in found if r["gt"] and r["decision_text"] == "yes")
    fn = sum(1 for r in found if r["gt"] and r["decision_text"] == "no")
    tn = sum(1 for r in found if not r["gt"] and r["decision_text"] == "no")
    fp = sum(1 for r in found if not r["gt"] and r["decision_text"] == "yes")
    log(f"True positives (real dtr, said yes): {tp}")
    log(f"False negatives (real dtr, said no): {fn}")
    log(f"True negatives (not dtr, said no): {tn}")
    log(f"False positives (not dtr, said yes): {fp}")

    if correct_cases:
        mean_p_correct = sum(r["own_prob"] for r in correct_cases) / len(correct_cases)
        mean_gap_correct = sum(r["logit_gap"] for r in correct_cases) / len(correct_cases)
        log(f"\nCorrect cases (n={len(correct_cases)}): mean P={mean_p_correct:.6f}  "
            f"mean logit_gap={mean_gap_correct:.3f}")
    if wrong_cases:
        mean_p_wrong = sum(r["own_prob"] for r in wrong_cases) / len(wrong_cases)
        mean_gap_wrong = sum(r["logit_gap"] for r in wrong_cases) / len(wrong_cases)
        log(f"WRONG cases (n={len(wrong_cases)}): mean P={mean_p_wrong:.6f}  "
            f"mean logit_gap={mean_gap_wrong:.3f}")
        log(f"\nWrong-case detail:")
        for r in wrong_cases:
            log(f"  category={r['category']}  gt={r['gt']}  pred={r['decision_text']}  "
                f"P={r['own_prob']:.6f}  gap={r['logit_gap']:.3f}")

    if correct_cases and wrong_cases:
        min_correct_gap = min(r["logit_gap"] for r in correct_cases)
        max_wrong_gap = max(r["logit_gap"] for r in wrong_cases)
        log(f"\nMin correct-case gap: {min_correct_gap:.3f}   Max wrong-case gap: {max_wrong_gap:.3f}")
        log(f"Clean separation (no overlap): {min_correct_gap > max_wrong_gap}")
        # distribution buckets for a real threshold sense
        import statistics
        log(f"\nCorrect-case gap: median={statistics.median(r['logit_gap'] for r in correct_cases):.3f}")
        log(f"Wrong-case gap: median={statistics.median(r['logit_gap'] for r in wrong_cases):.3f}")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nReport written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
