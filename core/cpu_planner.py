"""
Resident CPU planner model - the semantic-judgment half of the hybrid
planner architecture (2026-08-13, Jon's design): deterministic Python
pre-checks handle everything code already knows for certain (internet
toggle, repeated-failure circuit breaker, explicit-command detection -
see core/decision_heuristics.py and core/agent_tools/planner.py's use
of it), and this tiny model handles only the genuinely semantic parts
(which tool fits the question, is the request ambiguous, is a fact
knowable directly). Keeps the GPU dedicated to vision/extraction/final
synthesis (Gemma) - the planner never touches the GPU or
core/model_residency.py at all.

Model choice (confirmed by direct comparison, not assumed): Qwen2.5-0.5B-
Instruct was tried first and rejected - on every single test question
(including trivially unambiguous ones like "what is the capital of
France" and an explicit "search online for X") it picked
genealogy_framework_lookup regardless of catalog order, ruling out
primacy bias as the cause - it's a genuine capability gap at that size,
not a prompt-ordering problem. Qwen2.5-1.5B-Instruct correctly
discriminated all three tools, correctly reproduced the exact
ask_user_clarification "what year was the Canadian census" case this
whole design was built to fix, and correctly used answer_directly for
a trivial arithmetic question - the size step is what bought
reliability, not any prompt trick, so this is a real load-bearing model
choice, not a config default to casually swap later.

Loaded once, kept resident in system RAM for the life of the process -
same "load once, never release" pattern as core/model_residency.py's
Gemma residency, just in system RAM instead of VRAM and with no
borrow/release lifecycle needed (nothing else ever needs this model,
so there is no residency contention to manage).
"""

from __future__ import annotations

import threading
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 100

_lock = threading.Lock()
_tokenizer = None
_model = None


def _ensure_loaded() -> None:
    global _tokenizer, _model
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        print(f"[core.cpu_planner] loading {MODEL_NAME} on CPU (first use, resident for process lifetime)...")
        t0 = time.time()
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32, device_map="cpu")
        _model.eval()
        print(f"[core.cpu_planner] loaded in {time.time() - t0:.1f}s")


def generate_plan(prompt_text: str) -> tuple[str, dict[str, Any]]:
    """
    Runs one greedy-decoded planning completion on the resident CPU
    model. Greedy (do_sample=False), not sampled - this is a decision a
    downstream circuit breaker (planner.py) compares turn-to-turn for
    exact repetition, and planning decisions should be reproducible
    given the same scratchpad state, not vary run to run.

    Returns (raw_text, meta) matching the shape planner.py already
    expects from send_turn_fn's GPU path, so parse_plan_step() and the
    existing status_hub telemetry fields work unchanged.
    """
    _ensure_loaded()
    messages = [{"role": "user", "content": prompt_text}]
    chat_text = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer(chat_text, return_tensors="pt")
    t0 = time.time()
    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0
    generated_tokens = out.shape[1] - inputs["input_ids"].shape[1]
    raw = _tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    meta = {
        "model": MODEL_NAME,
        "device": "cpu",
        "latency_s": elapsed,
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "generated_tokens": int(generated_tokens),
    }
    return raw, meta
