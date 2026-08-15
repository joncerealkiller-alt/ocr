"""
Modular, deterministic, programmatic scorers for the Model Console
Benchmark tab (core/loader_registry.py's registry refactor's natural
next step - see benchmark/suite_schema.py's module docstring for the
full picture).

Every scorer function has the same signature:

    scorer(raw_output: str, args: dict) -> ScoreResult

and is registered in SCORER_REGISTRY under the name a suite YAML's
`cases[].scorer` field references. score_case() is the one place that
dispatches a case to its scorer, catching any exception the scorer
itself raises (a malformed case's `scorer_args` should produce a
visible "error" result, never crash the whole benchmark run - same
"per-test error handling" discipline as the rest of this project).

Deliberately NOT here: an LLM-as-judge scorer. Per the task's own
direction, this first implementation sticks to objective, programmatic
scoring; a subjective case that can't be scored this way is `scorer:
manual` (status "unscored"), not forced into a fake numeric score.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

# "pass" / "fail" / "error" / "unscored" - the four states every case
# result can be in. "partial" is deliberately not a fifth top-level
# status; scorers that produce a fractional score (e.g. json_field_match
# with 1 of 2 fields correct) still report "fail" (score < 1.0 is not a
# pass) but keep the numeric score/explanation so a reviewer can see
# HOW wrong it was, not just that it was wrong - satisfies "two models
# may receive similar scores for very different reasons."
Status = str


@dataclass
class ScoreResult:
    status: Status          # "pass" | "fail" | "error" | "unscored"
    score: Optional[float]  # 0.0-1.0, or None (error/unscored)
    explanation: str


def _normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace, drop trailing sentence
    punctuation - shared by every exact/contains-style scorer so
    "PONG", "pong.", " Pong \n" all normalize identically. Deliberately
    NOT stripping internal punctuation (e.g. "5:15 PM" needs its colon)
    - only leading/trailing whitespace and a trailing '.'/'!'."""
    t = " ".join(text.strip().split())
    return t.rstrip(".!").strip().lower()


def _extract_json_object(text: str) -> Optional[dict]:
    """Model output is rarely bare JSON (chat models wrap it in prose or
    a ```json fence) - takes the first {...} span and parses that,
    rather than requiring the whole response to be valid JSON. Returns
    None (not a raise) if nothing parses, so callers can produce a
    clean "fail", not the exception each of them would otherwise catch
    identically."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def score_exact_match_normalized(raw_output: str, args: dict) -> ScoreResult:
    expected = args["expected"]
    if _normalize(raw_output) == _normalize(expected):
        return ScoreResult("pass", 1.0, f"normalized output matched expected {expected!r}")
    return ScoreResult(
        "fail", 0.0,
        f"normalized output {_normalize(raw_output)!r} != expected {_normalize(expected)!r}"
    )


def score_exact_match_any(raw_output: str, args: dict) -> ScoreResult:
    """Same as above, but passes if the output matches ANY of
    args['expected_any'] once normalized - for answers with more than
    one acceptable literal form (e.g. "5:15 PM" vs "5:15pm")."""
    normalized_output = _normalize(raw_output)
    expected_any = args["expected_any"]
    for candidate in expected_any:
        if normalized_output == _normalize(candidate):
            return ScoreResult("pass", 1.0, f"matched acceptable answer {candidate!r}")
    return ScoreResult(
        "fail", 0.0,
        f"normalized output {normalized_output!r} matched none of {expected_any!r}"
    )


def score_contains_all(raw_output: str, args: dict) -> ScoreResult:
    keywords = args["keywords"]
    lowered = raw_output.lower()
    missing = [kw for kw in keywords if kw.lower() not in lowered]
    if not missing:
        return ScoreResult("pass", 1.0, f"all required keywords present: {keywords!r}")
    found_fraction = (len(keywords) - len(missing)) / len(keywords)
    return ScoreResult("fail", found_fraction, f"missing keywords: {missing!r}")


def score_contains_any(raw_output: str, args: dict) -> ScoreResult:
    keywords = args["keywords"]
    lowered = raw_output.lower()
    hits = [kw for kw in keywords if kw.lower() in lowered]
    if hits:
        return ScoreResult("pass", 1.0, f"matched keyword(s): {hits!r}")
    return ScoreResult("fail", 0.0, f"none of {keywords!r} found in output")


def score_json_required_keys(raw_output: str, args: dict) -> ScoreResult:
    required_keys = args["required_keys"]
    parsed = _extract_json_object(raw_output)
    if parsed is None:
        return ScoreResult("fail", 0.0, "no valid JSON object found in output")
    missing = [k for k in required_keys if k not in parsed]
    if not missing:
        return ScoreResult("pass", 1.0, f"JSON object present with all required keys {required_keys!r}")
    found_fraction = (len(required_keys) - len(missing)) / len(required_keys)
    return ScoreResult("fail", found_fraction, f"JSON parsed but missing keys: {missing!r}")


def score_json_field_match(raw_output: str, args: dict) -> ScoreResult:
    """Parses a JSON object out of raw_output and checks each of
    args['expected'] (a dict) is present with a value that CONTAINS
    (normalized, case-insensitive substring, not exact-equal) the
    expected value - substring rather than exact match because a model
    correctly extracting "John Test Sample" as "John T. Sample" or
    similar minor formatting variance shouldn't fail a field-accuracy
    check the way a typo'd wrong name should. Partial credit: score is
    the fraction of expected fields that matched."""
    expected: dict = args["expected"]
    parsed = _extract_json_object(raw_output)
    if parsed is None:
        return ScoreResult("fail", 0.0, "no valid JSON object found in output")
    correct, wrong = [], []
    for key, expected_value in expected.items():
        actual_value = str(parsed.get(key, ""))
        if _normalize(expected_value) in _normalize(actual_value):
            correct.append(key)
        else:
            wrong.append(f"{key}: got {actual_value!r}, expected to contain {expected_value!r}")
    score = len(correct) / len(expected) if expected else 0.0
    if not wrong:
        return ScoreResult("pass", score, f"all fields matched: {list(expected)!r}")
    return ScoreResult("fail", score, "; ".join(wrong))


def score_regex_line_format(raw_output: str, args: dict) -> ScoreResult:
    """Checks the output is exactly args['expected_lines'] non-empty
    lines, each matching args['line_pattern'] - a formatting-compliance
    check (e.g. "list exactly three colors, one per line, no
    punctuation") that's about SHAPE, not content correctness."""
    expected_lines = args["expected_lines"]
    pattern = re.compile(args["line_pattern"])
    lines = [line.strip() for line in raw_output.strip().splitlines() if line.strip()]
    if len(lines) != expected_lines:
        return ScoreResult(
            "fail", 0.0,
            f"expected exactly {expected_lines} non-empty lines, got {len(lines)}: {lines!r}"
        )
    bad = [line for line in lines if not pattern.match(line)]
    if bad:
        return ScoreResult(
            "fail", (len(lines) - len(bad)) / len(lines),
            f"{len(bad)} of {len(lines)} lines did not match pattern {args['line_pattern']!r}: {bad!r}"
        )
    return ScoreResult("pass", 1.0, f"{expected_lines} lines, all matching {args['line_pattern']!r}")


def score_abstention_expected(raw_output: str, args: dict) -> ScoreResult:
    """Passes when the output contains at least one of
    args['refusal_keywords'] - for cases where the correct behavior is
    honest abstention (insufficient evidence / no such information),
    not a confident wrong answer. See feedback_abstention_is_a_feature
    project memory: honest hedging beats a fabricated exact match, so
    this scorer rewards the hedge directly instead of treating "the
    model didn't answer" as a missing/failed case."""
    return score_contains_any(raw_output, {"keywords": args["refusal_keywords"]})


def score_manual(raw_output: str, args: dict) -> ScoreResult:
    """No programmatic scoring - the case is inherently subjective
    (e.g. "describe this image in one sentence" has no single correct
    string). Raw output is still fully captured by the runner; this
    scorer only marks the result as not-yet-judged so it isn't silently
    counted as a pass or fail in aggregate stats."""
    return ScoreResult("unscored", None, "no programmatic scorer for this case - review raw output manually")


SCORER_REGISTRY: dict[str, Callable[[str, dict], ScoreResult]] = {
    "exact_match_normalized": score_exact_match_normalized,
    "exact_match_any": score_exact_match_any,
    "contains_all": score_contains_all,
    "contains_any": score_contains_any,
    "json_required_keys": score_json_required_keys,
    "json_field_match": score_json_field_match,
    "regex_line_format": score_regex_line_format,
    "abstention_expected": score_abstention_expected,
    "manual": score_manual,
}


def score_case(scorer_name: str, scorer_args: dict, raw_output: Optional[str]) -> ScoreResult:
    """
    The one dispatch point benchmark/console_runner.py calls. Handles
    the two failure modes a scorer itself must never have to think
    about: raw_output is None (the model call itself errored - there is
    nothing to score) and an unknown/misconfigured scorer name/args
    (a bad suite YAML must produce a visible "error" result, not crash
    the run for every other case in the suite).
    """
    if raw_output is None:
        return ScoreResult("error", None, "no output to score - inference call failed")
    scorer = SCORER_REGISTRY.get(scorer_name)
    if scorer is None:
        return ScoreResult(
            "error", None,
            f"unknown scorer {scorer_name!r} - known scorers: {sorted(SCORER_REGISTRY)}"
        )
    try:
        return scorer(raw_output, scorer_args or {})
    except Exception as e:
        return ScoreResult("error", None, f"scorer raised {type(e).__name__}: {e}")
