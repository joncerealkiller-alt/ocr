"""
Web research as a bounded skill handoff, not a flat one-shot tool call
(2026-08-13, Jon's design suggestion). Previously web_search was a
single DuckDuckGo call - the "one successful call per planner turn"
circuit breaker in planner.py existed only because the CPU planner
itself was, confirmed live, prone to re-searching with a reworded query
instead of recognizing it already had enough evidence.

This module moves that bounded-refinement judgment to where it
belongs: a narrow, single-purpose sub-agent, invoked ONCE by the CPU
planner/circuit breaker exactly as before, but internally allowed up
to MAX_SEARCHES searches under its own policy: use evidence, not
unsupported model knowledge; refine only when the first result set is
insufficient; separate evidence from inference in the structured
result handed back. Control returns to the normal planner loop with
ONE ToolResultBlock, same shape as any other tool result - nothing
about planner.py's scratchpad/circuit-breaker/final-synthesis plumbing
needed to change.

Refine/synthesize calls run on a DEDICATED text-only research LLM
(core/loaders/text_llm_loader.py, config/models/qwen_research_text.yaml
- quantized Qwen2.5-7B-Instruct), not the chat model that was passed in
before 2026-08-13. Jon's framing: "as this is a research agent only, we
could look at a quantized 8b LLM not VLM, so we're not carrying the
extra bloat the vision tower adds" - these calls are already text-only
(image=None always), so a VLM's vision tower is pure dead weight here.
Confirmed live: InternVL3-8B (a VLM) fixed the digit-fabrication
problem this module's provenance checks were built for, but strained a
16GB card; this quantized text-only 7B matched its quality (~5.5GB
VRAM, faster generation, no vision weights at all). Borrowed via
core.model_residency.residency.borrow() - never held resident
independently, same "one authoritative GPU owner" discipline
extract_fields_tool already follows (core/agent_tools/tools.py).
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from core.agent_tools import research_llm
from core.agent_tools.dispatcher import dispatch
from core.agent_tools.errors import ToolErrorClass
from core.agent_tools.schema import ToolCallBlock, ToolResultBlock
from core.decision_heuristics import extract_years
from core.extraction_parsing import parse_kv_block

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "prompts"

MAX_SEARCHES = 3


def _load_prompt(filename: str, **substitutions: str) -> str:
    text = (PROMPTS_DIR / filename).read_text(encoding="utf-8")
    for key, value in substitutions.items():
        text = text.replace(f"{{{{{key}}}}}", value)
    return text


def _format_results(result: dict) -> str:
    items = result.get("results") or []
    if not items:
        return "(no results)"
    lines = []
    for item in items:
        lines.append(f"- {item.get('title', '')} ({item.get('url', '')}): {item.get('snippet', '')}")
    return "\n".join(lines)


def run_web_research(
    question: str,
    initial_query: str,
) -> ToolResultBlock:
    """
    Runs the bounded search -> refine? -> search -> ... -> synthesize
    loop, up to MAX_SEARCHES real DuckDuckGo calls. Returns a single
    ToolResultBlock (tool_name='web_search', matching what the outer
    planner loop already expects) whose result payload is a structured
    {summary, evidence, inference, sources, searches_performed} dict,
    not raw search hits - the outer final-answer synthesis reads
    evidence/inference distinctly rather than several duplicate raw
    result sets (the exact quality problem that motivated this,
    confirmed live: 3-4 redundant raw web_search results in the
    scratchpad produced a garbled final answer with conflated numbers).

    A search failure (e.g. internet access disabled, network error) on
    the FIRST attempt still surfaces as a failed ToolResultBlock exactly
    as web_search always has - the circuit breaker's existing failed-
    call handling in planner.py needs no changes.
    """
    call_id = str(uuid.uuid4())
    started = time.time()

    all_evidence_blocks: list[str] = []
    searches_performed: list[str] = []
    all_sources: list[dict] = []
    query = initial_query

    # One borrow for the whole function (2026-08-13, efficiency fix) -
    # previously each refine/synthesize call did its own separate
    # borrow/restore cycle (release Gemma -> load this model -> restore
    # Gemma), repeated up to 3x in a single turn. A single `with` here
    # covers every research-LLM call this function makes, so Gemma gets
    # displaced and restored exactly once regardless of how many
    # searches/refinements happen. dispatch()'s own web_search calls
    # inside this block are unaffected - they're non-GPU (network only)
    # and don't care what's currently GPU-resident.
    with research_llm.borrowed() as generate:
        for attempt in range(MAX_SEARCHES):
            call = ToolCallBlock(tool_name="web_search", args={"query": query}, call_id=f"{call_id}-search{attempt}")
            search_result = dispatch(call)
            searches_performed.append(query)

            if not search_result.success:
                if attempt == 0:
                    # First search failed outright (internet off, network
                    # error) - surface exactly as the old single-shot
                    # web_search tool did, no research-agent wrapping to
                    # obscure the real error.
                    return ToolResultBlock(
                        call_id=call_id, tool_name="web_search", success=False,
                        error=search_result.error, latency_s=time.time() - started,
                    )
                # A later refined query failed - fall back to whatever
                # evidence earlier searches already gathered rather than
                # losing it.
                break

            results_text = _format_results(search_result.result)
            all_evidence_blocks.append(f"Search {attempt + 1} (query: {query!r}):\n{results_text}")
            for item in (search_result.result.get("results") or []):
                if item.get("url"):
                    all_sources.append(item)

            if attempt == MAX_SEARCHES - 1:
                break

            refine_prompt = _load_prompt(
                "web_research_refine_v1.txt", QUESTION=question, QUERY=query, RESULTS=results_text,
            )
            refine_raw = generate(refine_prompt)
            try:
                fields = parse_kv_block(refine_raw)
            except ValueError:
                break
            decision = fields.get("decision", "").strip().upper()
            new_query = fields.get("new_query", "").strip()
            if decision != "REFINE" or not new_query:
                break
            query = new_query

        evidence_text = "\n\n".join(all_evidence_blocks) if all_evidence_blocks else "(no evidence gathered)"
        synth_prompt = _load_prompt("web_research_synthesize_v1.txt", QUESTION=question, EVIDENCE=evidence_text)
        synth_raw = generate(synth_prompt)
    try:
        synth_fields = parse_kv_block(synth_raw)
    except ValueError:
        synth_fields = {}

    # Provenance check (2026-08-13, confirmed live: a real turn produced
    # "1926" -> "1916"/a "2016" bracket artifact during this exact
    # synthesis step - digit substitution during paraphrase, not a
    # routing bug). A model asked to "summarize evidence" is free to
    # subtly reword a year; that's not acceptable for a fact this
    # project treats as load-bearing. Deterministic check, not another
    # prompt tweak (matches this project's repeated finding that
    # judgment-call prompt instructions don't reliably hold): every
    # year mentioned in the synthesized evidence must actually appear
    # in the RAW search snippet text. If the model introduced a year
    # that isn't traceable to any raw snippet, the paraphrase is not
    # trustworthy - quarantine it and fall back to the raw evidence
    # text untouched, same "quarantine, don't silently pass through"
    # principle as flag_suspicious_place_count elsewhere in this
    # project. This checks the RESEARCH BUNDLE's own provenance only -
    # it does not touch or fix drift the OUTER final-answer synthesis
    # step might separately introduce on top of even a clean bundle.
    synthesized_evidence = synth_fields.get("evidence", "").strip()
    raw_years = extract_years(evidence_text)
    synth_years = extract_years(synthesized_evidence)
    unverified_years = synth_years - raw_years
    if synthesized_evidence and unverified_years:
        print(f"[core.agent_tools.web_research_agent] synthesized evidence mentioned "
              f"year(s) {sorted(unverified_years)} not present in any raw search "
              f"snippet - quarantining the paraphrase, falling back to raw evidence "
              f"text. Synthesized text was: {synthesized_evidence!r}")
        evidence_for_result = evidence_text
    else:
        evidence_for_result = synthesized_evidence or evidence_text

    # Same provenance check applied to 'inference' (2026-08-13, gap
    # found in review): the field is SUPPOSED to hold reasoning beyond
    # the raw evidence, so it's not wrong for it to add content - but a
    # fabricated YEAR isn't legitimate inference, it's the same
    # corruption bug in a field this check didn't originally cover
    # (confirmed live: a real turn's inference field stated "1876
    # Census"/"1826 census", neither grounded in anything).
    inference_text = synth_fields.get("inference", "").strip() or "none"
    unverified_inference_years = extract_years(inference_text) - raw_years
    if unverified_inference_years:
        print(f"[core.agent_tools.web_research_agent] inference field mentioned "
              f"year(s) {sorted(unverified_inference_years)} not present in any raw "
              f"search snippet - quarantining. Text was: {inference_text!r}")
        inference_text = "none"

    # 'sources' is NEVER taken from the model's own synthesized text
    # (2026-08-13, real bug found live: switching to qwen3vl4b didn't
    # fix digit fidelity, it corrupted the sources field instead -
    # "/1926.html" became ",1912.html", real record IDs like 3005862
    # and 5033008 became 3005962/5033908, and none of that was caught
    # because the year-provenance check only covered evidence/inference,
    # so the garbled URLs flowed straight through even the "safe"
    # fallback path). URLs are exactly the kind of thing a model should
    # never be asked to reproduce from memory/paraphrase at all - they
    # are already available verbatim in `all_sources` from the actual
    # search results, so this is now a pure code join, no model
    # involvement, no provenance check needed because there is nothing
    # to verify against - it IS the raw data.
    sources_text = ", ".join(dict.fromkeys(s["url"] for s in all_sources))

    result_payload = {
        "evidence": evidence_for_result,
        "raw_evidence": evidence_text,
        "inference": inference_text,
        "sources": sources_text,
        "searches_performed": searches_performed,
    }
    return ToolResultBlock(
        call_id=call_id, tool_name="web_search", success=True,
        result=result_payload, latency_s=time.time() - started,
    )
