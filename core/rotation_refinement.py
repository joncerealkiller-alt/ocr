"""
Shared "validate then refine" utility for any calibrated blob/anchor
detector that can act as a downstream sensor on an upstream deskew
estimate - built 2026-08-08 per Jon's proposal: deskew is a good but
imperfect first estimate, and a later, more constrained detector (the
1906 row-number blob detector, the header-number blob detector, and
potentially table-boundary refinement) already knows roughly where its
features SHOULD land, so a low-confidence result there is real evidence
the deskew angle itself might be off - not just a detector shortfall to
route around.

THE CORE DISTINCTION THIS MODULE ENFORCES, Jon's own framing verbatim:
"best angle in the search" and "good enough to trust" are NOT the same
thing. A local rotation sweep always returns SOME best-scoring angle,
even when the true answer is "nothing here is trustworthy, quarantine
this page" - refine_rotation() never adopts a swept angle just because
it scored highest in the sweep; it must beat the BASELINE by a real,
configurable margin (min_score_margin), or the result reports
quarantine=True and refined_angle falls back to baseline_angle
unchanged. Precedent for why this matters: core/auto_sidecar.py's
_detect_header_number_blobs() (a simpler, density-only blob attempt,
2026-07-27) was tried and found to WORSEN already-accurate pages before
being superseded - "picks something that looks locally better" is a
real, previously-observed failure mode in this exact codebase, not a
hypothetical one.

DELIBERATELY DECOUPLED FROM IMAGE I/O - this module never opens an
image, never rotates one, never calls a specific detector. The caller
supplies `detect_at_delta(delta_deg) -> list[BlobCandidate]` (typically
a closure over a real image + a real detector like core.row_segmentation
.detect_row_number_centers/detect_column_number_centers, re-run against
the image rotated by delta_deg beyond the baseline angle). This keeps
the scoring/search logic testable with plain synthetic candidate lists
(see this module's own tests) and reusable across every calibrated
detector this pattern could apply to, per Jon's direction: "making the
rotation search/scoring a shared utility... that's the right abstraction
point."

SCORING - count alone was explicitly rejected (the same failure mode as
_detect_header_number_blobs() above: optimizing purely for "found more
stuff" invites locking onto noise). Combines three signals, per Jon's
own proposal:
  - count: how many candidates were found, relative to how many are
    expected (a template's own expected_row_count, or similar).
  - spacing regularity: printed numbers are evenly spaced - real
    candidates should show LOW variance in consecutive gaps; noise
    doesn't.
  - mean confidence: whatever per-candidate signal the caller's own
    detector can supply (e.g. detect_column_number_centers()'s width_px,
    or a normalized ink-density/contrast value) - optional, defaults to
    1.0 (no discrimination) if the caller's detector doesn't produce one.

NOT YET WIRED INTO ANY LIVE DETECTOR (2026-08-08) - core.row_segmentation
.detect_row_number_centers()/detect_column_number_centers() and core.
auto_sidecar.py's header-number-anchor machinery are all still research-
only, never called from generate_auto_sidecar()'s live path (see those
functions' own docstrings / core/column_calibration.py's try_load_
anchor_cache() docstring). This module is the reusable scoring/search
piece, ready for whichever detector goes live first to adopt - wiring
it into generate_auto_sidecar() itself is a separate, later step, not
done here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class BlobCandidate:
    """One detected candidate position (e.g. a row-number blob's y-center,
    or a column-number blob's x-center) at a given rotation delta.
    confidence is whatever relative signal the caller's detector can
    supply - width_px, ink density, contrast - normalized or not, this
    module only ever uses it relatively (mean across candidates at one
    angle vs. another), never as an absolute threshold."""
    position: float
    confidence: float = 1.0


@dataclass
class RotationScore:
    delta_deg: float
    candidates: list[BlobCandidate]
    count: int
    spacing_regularity: float   # 1.0 = perfectly even spacing, 0.0 = wildly irregular
    mean_confidence: float
    score: float                # the single combined value used for ranking/margin comparison


@dataclass
class RotationRefinementResult:
    baseline_angle: float
    refined_angle: float        # == baseline_angle unless a swept delta both won AND cleared the margin
    delta_deg: float             # refined_angle - baseline_angle (0.0 if not refined)
    used_refinement: bool        # True only if a swept angle was actually adopted
    quarantine: bool             # True if the baseline was NOT trusted and nothing else cleared the margin either
    baseline_score: RotationScore
    refined_score: RotationScore | None   # None if refinement was never attempted (baseline already trusted)
    evidence: dict = field(default_factory=dict)  # Jon's exact logging schema, see build_evidence()


def score_candidates(
    candidates: list[BlobCandidate], delta_deg: float,
    expected_count: int | None = None,
    count_weight: float = 0.5, spacing_weight: float = 0.35, confidence_weight: float = 0.15,
) -> RotationScore:
    """
    Combines count/spacing/confidence into one score in roughly [0, 1]
    (count and confidence terms are individually clamped to that range;
    the weighted sum stays close to it as long as the weights sum to 1,
    which they do by default - not hard-enforced, a caller who changes
    the weights is responsible for whether that still makes sense).
    Never raises on an empty candidate list - returns an all-zero score,
    the correctly "worst possible, but not exceptional-case" result.
    """
    count = len(candidates)
    if count == 0:
        return RotationScore(delta_deg, candidates, 0, 0.0, 0.0, 0.0)

    if expected_count and expected_count > 0:
        count_term = min(count / expected_count, 1.0)
    else:
        # No expectation supplied - count alone can't be normalized
        # meaningfully, so it contributes nothing rather than an
        # arbitrary raw scale that would dominate the other two terms.
        count_term = 0.0

    if count >= 2:
        positions = sorted(c.position for c in candidates)
        gaps = [b - a for a, b in zip(positions, positions[1:])]
        mean_gap = sum(gaps) / len(gaps)
        if mean_gap > 0:
            variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
            coefficient_of_variation = (variance ** 0.5) / mean_gap
            # CoV of 0 -> regularity 1.0; CoV >= 1.0 (gaps as spread out
            # as their own mean) -> regularity floored at 0.0, not negative.
            spacing_regularity = max(0.0, 1.0 - coefficient_of_variation)
        else:
            spacing_regularity = 0.0
    else:
        # A single candidate has no gap to measure - neither confirms nor
        # denies regularity, scored neutrally rather than 0 (which would
        # unfairly penalize a real page that just has few features) or 1
        # (which would let one lucky blob look as good as a clean run).
        spacing_regularity = 0.5

    mean_confidence = sum(c.confidence for c in candidates) / count
    # confidence is caller-defined and possibly unbounded (e.g. raw
    # width_px) - clamp its contribution to the combined score at 1.0 so
    # one detector's larger confidence scale can't silently dominate the
    # count/spacing terms relative to another detector's smaller scale.
    confidence_term = min(mean_confidence, 1.0)

    score = count_weight * count_term + spacing_weight * spacing_regularity + confidence_weight * confidence_term
    return RotationScore(delta_deg, candidates, count, spacing_regularity, mean_confidence, score)


def build_evidence(result: RotationRefinementResult) -> dict:
    """Jon's exact requested logging schema (2026-08-08): deskew_angle,
    refined_angle, delta, baseline_score, refined_score - kept as a
    separate function (not inlined into refine_rotation()) so a caller
    that wants this shape for its own sidecar/diagnostics dict can call
    it without needing to know RotationRefinementResult's full field
    layout."""
    return {
        "deskew_angle": result.baseline_angle,
        "refined_angle": result.refined_angle,
        "delta": result.delta_deg,
        "baseline_score": result.baseline_score.score,
        "refined_score": result.refined_score.score if result.refined_score else None,
        "used_refinement": result.used_refinement,
        "quarantine": result.quarantine,
    }


def refine_rotation(
    baseline_angle: float,
    detect_at_delta: Callable[[float], list[BlobCandidate]],
    min_count_to_trust_baseline: int,
    expected_count: int | None = None,
    search_range_deg: float = 2.0,
    search_step_deg: float = 0.25,
    min_score_margin: float = 0.15,
    min_absolute_score: float = 0.5,
    log: Callable[[str], None] = lambda _msg: None,
) -> RotationRefinementResult:
    """
    Step 1 - try the baseline angle (delta=0.0) first, always. If it
    finds at least min_count_to_trust_baseline candidates, DONE - no
    search, no risk, this is the common/fast/confident case (matches
    Jon's flowchart: "confidence sufficient? yes -> keep").

    Step 2 - only if the baseline came up short: sweep +/- search_range
    _deg in search_step_deg increments (delta=0.0 already tried, not
    repeated), scoring every candidate set the SAME way as the baseline
    via score_candidates() so the comparison is apples-to-apples.

    Step 3 - TWO conditions must BOTH hold to adopt a swept angle, not
    one:
      (a) it beats the baseline by min_score_margin (relative check -
          "clearly better than what we started with"), AND
      (b) it clears min_absolute_score on its own (absolute check -
          "actually good, not just less-bad-than-a-very-bad-baseline").
    (a) alone is NOT enough - found by this module's own test suite
    (2026-08-08): sweeping ~17 angles against a genuinely bad baseline
    (score ~0.1) gives pure random noise a real chance that AT LEAST ONE
    swept angle clears "baseline + 0.15" purely by the multiple-
    comparisons problem, without ever finding real signal - exactly the
    "best in the search != good enough to trust" failure this module
    exists to prevent, caught by testing against synthetic pure noise
    rather than assumed away. min_absolute_score is the fix: a lucky-
    looking winner that's still objectively mediocre can't pass.

    Ties, narrow wins, or wins that fail the absolute floor all keep the
    baseline and set quarantine=True: real evidence that this page needs
    a human look, not a confident-looking guess.
    """
    baseline_candidates = detect_at_delta(0.0)
    baseline_score = score_candidates(baseline_candidates, 0.0, expected_count)

    if baseline_score.count >= min_count_to_trust_baseline:
        log(f"Baseline angle trusted: {baseline_score.count} candidate(s) "
            f"(>= {min_count_to_trust_baseline} required) - no rotation search needed.")
        result = RotationRefinementResult(
            baseline_angle=baseline_angle, refined_angle=baseline_angle, delta_deg=0.0,
            used_refinement=False, quarantine=False,
            baseline_score=baseline_score, refined_score=None,
        )
        result.evidence = build_evidence(result)
        return result

    log(f"Baseline angle under-confident: {baseline_score.count} candidate(s) "
        f"(< {min_count_to_trust_baseline} required) - sweeping +/-{search_range_deg} deg.")

    deltas = []
    d = search_step_deg
    while d <= search_range_deg + 1e-9:
        deltas.extend([d, -d])
        d += search_step_deg

    swept_scores = [baseline_score]  # baseline included so max() naturally covers "nothing beat it"
    for delta in deltas:
        candidates = detect_at_delta(delta)
        swept_scores.append(score_candidates(candidates, delta, expected_count))

    best = max(swept_scores, key=lambda s: s.score)

    clears_margin = best.score - baseline_score.score >= min_score_margin
    clears_floor = best.score >= min_absolute_score
    if best.delta_deg != 0.0 and clears_margin and clears_floor:
        log(f"Refined angle adopted: delta={best.delta_deg:+.2f} deg, "
            f"score {baseline_score.score:.3f} -> {best.score:.3f} "
            f"(margin {best.score - baseline_score.score:.3f} >= {min_score_margin}).")
        result = RotationRefinementResult(
            baseline_angle=baseline_angle, refined_angle=baseline_angle + best.delta_deg,
            delta_deg=best.delta_deg, used_refinement=True, quarantine=False,
            baseline_score=baseline_score, refined_score=best,
        )
    else:
        reason = [] if best.delta_deg != 0.0 else ["baseline itself was the best-scoring angle"]
        if not clears_margin:
            reason.append(f"margin not cleared ({best.score - baseline_score.score:.3f} < {min_score_margin})")
        if not clears_floor:
            reason.append(f"absolute floor not cleared ({best.score:.3f} < {min_absolute_score})")
        log(f"No swept angle adopted (best score {best.score:.3f} vs. baseline "
            f"{baseline_score.score:.3f}): {'; '.join(reason)} - quarantining rather than guessing further.")
        result = RotationRefinementResult(
            baseline_angle=baseline_angle, refined_angle=baseline_angle, delta_deg=0.0,
            used_refinement=False, quarantine=True,
            baseline_score=baseline_score, refined_score=best,
        )

    result.evidence = build_evidence(result)
    return result
