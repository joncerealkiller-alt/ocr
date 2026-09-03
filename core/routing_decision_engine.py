"""
Routing Decision Engine - evidence-fusion arbitration layer for top-level
document routing. Built 2026-08-07 per Jon's explicit direction to begin
production implementation, using docs/MULTI_SOURCE_VOTING_CLASSIFIER_
PROPOSAL.md's research as the design basis (NOT hard-coded conclusions -
several open questions there are explicitly left open here too).

NAMING COLLISION, addressed on purpose: `core/decision_engine.py`
ALREADY EXISTS and does something different - Stage 2 per docs/
PIPELINE_STAGE_TERMINOLOGY.md, a preprocessing-profile pass-through plus
tower-consensus COMPUTATION/recording that explicitly "does not act on
it, only records it" (see that module's own docstring). THIS module is
later-stage and does the opposite job: it CONSUMES already-recorded
evidence (from Gemma, vision towers, CV sensors, whatever a future
sensor produces) and ARBITRATES a real routing decision - route,
quarantine, or no-decision, with a full audit trail. Named
`routing_decision_engine.py`, not `decision_engine.py`, specifically so
neither module's meaning drifts. Not yet wired into core/manifest_
pipeline.py's Stage 0-6 flow - see the module-level "INTEGRATION STATUS"
note near the bottom of this file for what's built vs. what's next.

CORE PRINCIPLE: this module NEVER runs a sensor. It has zero imports of
transformers/timm/torch/cv2/anything that does inference. It consumes
`EvidenceRecord` objects that something ELSE already produced (a live
adapter, or - today - a replay harness reading already-recorded JSON
from diagnostics/run_gemma_*.py / diagnostics/extract_vit_family_
logits.py). This separation is deliberate and load-bearing: sensor
acquisition can evolve (new towers qualify, Gemma fine-tuning may or may
not become possible, a new CV signal gets built) without this module's
arbitration logic changing at all.

EVIDENCE BOUNDARY TAGS used throughout this file's comments, matching
the proposal doc's discipline:
  [A] measured result / existing implementation
  [B] interpretation supported by measurements
  [C] hypothesis / design choice not yet validated - stated as such,
      not silently promoted to a settled fact

GEMMA PRIORITY IS PROVISIONAL, NOT STRUCTURAL. `gemma_primary` is
TODAY's configured default policy (config/decision_engine.yaml) because
Gemma is currently the single best-measured sensor on the one held-out
comparison that exists (93.7-94.0% depending on prompt variant, vs.
towers' 89.8-92.5% - see the proposal doc's Gemma comparator sections).
That is a CONFIGURATION VALUE, not a code path only Gemma can take -
`vision_primary`, `weighted_fusion`, and `class_specific_authority` are
implemented as equally-real policies, switchable via one YAML field with
zero code changes (see tests/test_decision_engine.py's policy-switch
test, which exists specifically to prove this). If Gemma E2B fine-tuning
turns out to be infeasible on 16GB, or a future vision benchmark
surpasses it, the answer is "change one config value," never "rewrite
this module."

UNCALIBRATED THRESHOLDS: every numeric threshold in DEFAULT_THRESHOLDS
below is explicitly marked uncalibrated in its own comment. Per Jon's
explicit instruction, this module does NOT invent real production
thresholds - the Gemma logit-margin numbers (auto-accept ~20, quarantine
~7) come directly from the earlier logit-confidence experiments
(docs/MULTI_SOURCE_VOTING_CLASSIFIER_PROPOSAL.md's "Logit-based
confidence" sections) but those experiments themselves concluded this is
"real, useful signal... not a strict correctness predictor" at the
sample sizes tested - a DIRECTIONAL starting point, not a validated
production cutoff. Recalibrate against real production outcomes before
trusting the trust-state boundaries at these exact numbers.
"""

from __future__ import annotations

import enum
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "decision_engine.yaml"


class TrustState(str, enum.Enum):
    """Names are flexible per Jon's note; semantics are what matter.
    AUTO_ACCEPT: confident primary evidence, no meaningful contradiction.
    ACCEPT_WITH_CAUTION: a real decision was reached, but with either a
      weak primary signal backed by strong supporting agreement, or a
      confident primary with a WEAK (not strong-family-consensus)
      contradiction - worth a lighter-touch review path, not full
      quarantine.
    QUARANTINE: evidence is weak, contradictory, or a strong opposing
      consensus exists - route to human review, do not guess.
    NO_DECISION: no usable evidence was supplied at all (e.g. the
      configured primary sensor for this policy/category produced no
      record) - distinct from QUARANTINE, which means evidence EXISTS
      and disagrees/is weak, not that evidence is simply absent.
    """
    AUTO_ACCEPT = "auto_accept"
    ACCEPT_WITH_CAUTION = "accept_with_caution"
    QUARANTINE = "quarantine"
    NO_DECISION = "no_decision"


@dataclass
class EvidenceRecord:
    """
    One sensor's observation on one image. Producer-agnostic on purpose -
    a live Gemma call, a replayed JSON prediction, and a future CV
    sensor all produce the same shape. Nothing in this dataclass runs
    inference; it is pure data.

    sensor_name: e.g. "gemma", "siglip", "mobilenetv2", "table_confidence",
      "row_regularity". Must be unique per distinct sensor, not per family.
    sensor_family: grouping key for correlated-sensor discounting (see
      DecisionConfig.family_weight) - e.g. "gemma_semantic", "vit_isotropic",
      "cnn", "classical_cv". Config-driven categorization, not hardcoded
      here; the family name is whatever the evidence producer assigns
      (ideally matching config/decision_engine.yaml's `sensor_families`
      map, but this module does not enforce that - a mismatched family
      name just means that sensor doesn't get grouped/discounted with
      its real relatives, a soft failure, not a crash).
    predicted_class: None means this sensor produced no usable
      prediction for this image (e.g. missing evidence, a failed call) -
      distinct from a real "uncertain_review" prediction, which is a
      real class value.
    raw_score: sensor-native units, NOT normalized - e.g. Gemma's raw
      decision-token logit margin, a tower's top1-top2 logit margin, a
      CV signal's own confidence scale. Meaning is defined by
      `metadata.get("score_kind")` (e.g. "logit_margin", "cosine_margin",
      "softmax_probability", "ordinal_confidence") - the engine looks
      this up in DecisionConfig.score_semantics to know how to interpret
      raw_score's scale, rather than assuming a universal 0-1 range.
    normalized_score: 0.0-1.0 if the producer has already normalized it
      (e.g. softmax top-1 probability) - optional, may be None even when
      raw_score is present.
    uncertainty: an explicit 0 (confident) - 1 (uncertain) estimate if
      the sensor provides one (e.g. normalized entropy) - separate from
      raw_score/normalized_score since [B] per this project's own
      findings, "low margin means uncertainty, not necessarily
      incorrectness" - uncertainty and (in)correctness are related but
      distinct concepts, kept as distinct fields rather than conflated.
    calibrated: True only if normalized_score/uncertainty have been
      checked against real measured accuracy at that score level for
      THIS sensor - False (the default) means "this is a raw signal,
      treat its exact numeric value as directional, not a probability
      you can trust point-for-point." Per-project finding: Gemma's
      self-reported confidence field is explicitly NOT calibrated
      (checked three independent times, see the proposal doc) - always
      calibrated=False for that field specifically.
    reliability_by_class: optional {class_id: {"precision":.., "recall":..,
      "n":..}} - per-sensor, per-class historical performance, e.g. from
      the multi-architecture benchmark's per-category accuracy tables.
      Used by class_specific_authority to judge whether a sensor is
      trustworthy for THIS SPECIFIC predicted class, not just overall.
    metadata: anything else worth keeping for the audit trail - e.g.
      "score_kind", "model_tag", "inference_cost_ms", raw text reason.
    """
    sensor_name: str
    sensor_family: str
    predicted_class: str | None
    raw_score: float | None = None
    normalized_score: float | None = None
    uncertainty: float | None = None
    calibrated: bool = False
    reliability_by_class: dict[str, dict] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "sensor_name": self.sensor_name, "sensor_family": self.sensor_family,
            "predicted_class": self.predicted_class, "raw_score": self.raw_score,
            "normalized_score": self.normalized_score, "uncertainty": self.uncertainty,
            "calibrated": self.calibrated, "reliability_by_class": self.reliability_by_class,
            "metadata": self.metadata,
        }


@dataclass
class DecisionResult:
    """The engine's output for one image. `audit_trail` is the primary
    deliverable per Jon's explicit requirement - a human-readable,
    ordered list of WHY this decision was reached, not just the final
    numbers."""
    image_id: str | None
    selected_class: str | None
    trust_state: TrustState
    policy_used: str
    evidence_for: list[EvidenceRecord]
    evidence_against: list[EvidenceRecord]
    disagreements: list[dict]
    quarantine_reason: str | None
    audit_trail: list[str]

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id, "selected_class": self.selected_class,
            "trust_state": self.trust_state.value, "policy_used": self.policy_used,
            "evidence_for": [e.to_dict() for e in self.evidence_for],
            "evidence_against": [e.to_dict() for e in self.evidence_against],
            "disagreements": self.disagreements,
            "quarantine_reason": self.quarantine_reason,
            "audit_trail": self.audit_trail,
        }


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# UNCALIBRATED. Directional starting points from docs/MULTI_SOURCE_
# VOTING_CLASSIFIER_PROPOSAL.md's logit-confidence experiments, NOT a
# production-validated cutoff - those experiments' own conclusion was
# "real, useful signal... not a strict correctness predictor" at the
# sample sizes tested (n=48-368). Overridable per config/decision_
# engine.yaml without touching this file.
DEFAULT_THRESHOLDS: dict[str, float] = {
    # Gemma raw decision-token logit margin (score_kind="logit_margin").
    # [A] measured: correct-case median gap 30.41 vs wrong-case median
    # 13.56 on the n=368 real-ground-truth run; "most confident correct
    # examples were >20" and "margin around 3-7 corresponds to genuine
    # uncertainty in tested cases" per Jon's summary above.
    "gemma_logit_margin_auto_accept": 20.0,   # UNCALIBRATED
    "gemma_logit_margin_quarantine_below": 7.0,  # UNCALIBRATED
    # Vision tower normalized top-1 softmax probability
    # (score_kind="softmax_probability"). No dedicated calibration
    # experiment run yet for these exact cutoffs - placeholder symmetry
    # with the Gemma numbers above, nothing more.
    "vision_softmax_auto_accept": 0.90,   # UNCALIBRATED
    "vision_softmax_quarantine_below": 0.55,  # UNCALIBRATED
    # Generic ordinal/CV-style confidence (score_kind="ordinal_confidence",
    # e.g. table_confidence) - 0-1 scale, no probabilistic meaning implied.
    "cv_ordinal_auto_accept": 0.80,   # UNCALIBRATED
    "cv_ordinal_quarantine_below": 0.375,  # UNCALIBRATED - this ONE number
    # is the exception: 0.375 is core/image_analysis.py's real, measured
    # table_confidence floor (16 labelled census pages scored min 0.375;
    # 218 microfilm non-tables scored median 0.125). Still marked
    # UNCALIBRATED here because that calibration was for "is there a
    # table," not for "should the decision engine trust this evidence" -
    # a related but not identical question.
}

# score_kind -> which two threshold keys govern it. New score kinds can
# be added here without touching any policy function below.
DEFAULT_SCORE_SEMANTICS: dict[str, dict[str, str]] = {
    "logit_margin": {
        "auto_accept_key": "gemma_logit_margin_auto_accept",
        "quarantine_below_key": "gemma_logit_margin_quarantine_below",
    },
    "softmax_probability": {
        "auto_accept_key": "vision_softmax_auto_accept",
        "quarantine_below_key": "vision_softmax_quarantine_below",
    },
    "ordinal_confidence": {
        "auto_accept_key": "cv_ordinal_auto_accept",
        "quarantine_below_key": "cv_ordinal_quarantine_below",
    },
}


@dataclass
class DecisionConfig:
    default_policy: str = "gemma_primary"
    thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    score_semantics: dict[str, dict[str, str]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_SCORE_SEMANTICS.items()})
    sensor_families: dict[str, list[str]] = field(default_factory=dict)
    # policy-specific knobs
    vision_primary_sensor: str | None = None  # None = fall back to family-weighted vision consensus
    weighted_fusion_family_weight: dict[str, float] = field(default_factory=dict)  # family -> weight, default 1.0
    class_specific_authority: dict[str, dict] = field(default_factory=dict)
    # class_specific_authority[category] = {"primary": sensor_name, "supporting": [sensor_name, ...]}
    class_specific_fallback_policy: str = "weighted_fusion"

    def family_of(self, sensor_name: str) -> str | None:
        for family, members in self.sensor_families.items():
            if sensor_name in members:
                return family
        return None


def load_decision_config(path: Path = DEFAULT_CONFIG_PATH) -> DecisionConfig:
    """Loads config/decision_engine.yaml. Missing file -> pure-default
    config (gemma_primary, DEFAULT_THRESHOLDS) rather than an error, so
    this module works standalone (e.g. in tests) without requiring the
    real config file to exist."""
    if not path.exists():
        return DecisionConfig()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = DecisionConfig()
    cfg.default_policy = raw.get("default_policy", cfg.default_policy)
    cfg.thresholds = {**cfg.thresholds, **(raw.get("thresholds") or {})}
    if raw.get("score_semantics"):
        cfg.score_semantics = {**cfg.score_semantics, **raw["score_semantics"]}
    cfg.sensor_families = raw.get("sensor_families") or {}
    cfg.vision_primary_sensor = raw.get("vision_primary_sensor")
    cfg.weighted_fusion_family_weight = raw.get("weighted_fusion_family_weight") or {}
    cfg.class_specific_authority = raw.get("class_specific_authority") or {}
    cfg.class_specific_fallback_policy = raw.get(
        "class_specific_fallback_policy", cfg.class_specific_fallback_policy)
    return cfg


# ---------------------------------------------------------------------
# Shared scoring helpers
# ---------------------------------------------------------------------

def _accept_thresholds_for(record: EvidenceRecord, config: DecisionConfig) -> tuple[float | None, float | None, float | None]:
    """Returns (value_to_compare, auto_accept_threshold, quarantine_below_threshold)
    for one evidence record, or (None, None, None) if this record's
    score_kind isn't recognized / it has no usable score at all."""
    score_kind = record.metadata.get("score_kind")
    value = record.raw_score if record.raw_score is not None else record.normalized_score
    if value is None or score_kind not in config.score_semantics:
        return None, None, None
    sem = config.score_semantics[score_kind]
    auto_accept = config.thresholds.get(sem["auto_accept_key"])
    quarantine_below = config.thresholds.get(sem["quarantine_below_key"])
    return value, auto_accept, quarantine_below


def _is_confident(record: EvidenceRecord, config: DecisionConfig) -> bool | None:
    """True/False if determinable, None if this record has no score
    semantics the engine understands (treated as neutral, not counted
    either way - an unscored sensor's mere agreement/disagreement on
    predicted_class still matters, just not its confidence)."""
    value, auto_accept, _ = _accept_thresholds_for(record, config)
    if value is None or auto_accept is None:
        return None
    return value >= auto_accept


def _is_weak(record: EvidenceRecord, config: DecisionConfig) -> bool | None:
    value, _, quarantine_below = _accept_thresholds_for(record, config)
    if value is None or quarantine_below is None:
        return None
    return value < quarantine_below


def _group_by_family(records: list[EvidenceRecord], config: DecisionConfig) -> dict[str, list[EvidenceRecord]]:
    groups: dict[str, list[EvidenceRecord]] = defaultdict(list)
    for r in records:
        family = r.sensor_family or config.family_of(r.sensor_name) or r.sensor_name
        groups[family].append(r)
    return dict(groups)


def _family_votes(records: list[EvidenceRecord], config: DecisionConfig) -> Counter:
    """One vote per FAMILY (not per sensor) for whichever class that
    family's members most commonly predicted - the mechanism that stops
    5 correlated ViT-family predictions from outweighing 1 independent
    CV signal, per Jon's explicit requirement."""
    groups = _group_by_family(records, config)
    votes: Counter = Counter()
    for family, members in groups.items():
        preds = [m.predicted_class for m in members if m.predicted_class]
        if not preds:
            continue
        top_class, _ = Counter(preds).most_common(1)[0]
        weight = config.weighted_fusion_family_weight.get(family, 1.0)
        votes[top_class] += weight
    return votes


# ---------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------

def _arbitrate_primary_supporting(
    primary: EvidenceRecord | None,
    supporting: list[EvidenceRecord],
    config: DecisionConfig,
    policy_name: str,
    image_id: str | None,
) -> DecisionResult:
    """
    Shared arbitration core used by gemma_primary, vision_primary, and
    class_specific_authority - all three are really "trust one primary
    sensor, cross-check against supporting evidence, family-weighted"
    with a different choice of WHICH sensor is primary. Implements the
    worked examples in Jon's spec directly:
      - primary confident + supporting agrees (or only weak contradiction)
        -> AUTO_ACCEPT
      - primary confident + supporting has a STRONG, family-weighted
        disagreement -> ACCEPT_WITH_CAUTION
      - primary weak + supporting strongly, consistently agrees with it
        -> ACCEPT_WITH_CAUTION (weak primary rescued by real corroboration)
      - primary weak + supporting disagrees or is itself weak/absent
        -> QUARANTINE
      - primary missing entirely -> NO_DECISION
    """
    audit: list[str] = [f"Policy: {policy_name}"]
    if primary is None or primary.predicted_class is None:
        audit.append("No primary evidence available - NO_DECISION.")
        return DecisionResult(
            image_id=image_id, selected_class=None, trust_state=TrustState.NO_DECISION,
            policy_used=policy_name, evidence_for=[], evidence_against=list(supporting),
            disagreements=[], quarantine_reason="no primary evidence", audit_trail=audit,
        )

    selected_class = primary.predicted_class
    primary_confident = _is_confident(primary, config)
    primary_weak = _is_weak(primary, config)
    value, auto_accept_t, quarantine_t = _accept_thresholds_for(primary, config)
    if value is not None:
        audit.append(f"Primary [{primary.sensor_name}]: {selected_class}, "
                      f"score={value:.3g} (auto_accept>={auto_accept_t}, "
                      f"quarantine<{quarantine_t})")
    else:
        audit.append(f"Primary [{primary.sensor_name}]: {selected_class}, no scored confidence available")

    agreeing = [s for s in supporting if s.predicted_class == selected_class]
    disagreeing = [s for s in supporting if s.predicted_class and s.predicted_class != selected_class]

    family_votes_disagree = _family_votes(disagreeing, config)
    family_votes_agree = _family_votes(agreeing, config)
    strong_disagree_families = [
        s for s in disagreeing if _is_confident(s, config)
    ]
    strong_disagree_family_names = {
        (s.sensor_family or config.family_of(s.sensor_name) or s.sensor_name) for s in strong_disagree_families
    }

    for s in agreeing:
        audit.append(f"  Supporting [{s.sensor_name}] AGREES: {s.predicted_class}")
    for s in disagreeing:
        conf_note = "confident" if _is_confident(s, config) else "weak/unscored"
        audit.append(f"  Supporting [{s.sensor_name}] DISAGREES ({conf_note}): {s.predicted_class}")

    disagreements = [
        {"sensor_name": s.sensor_name, "sensor_family": s.sensor_family, "predicted_class": s.predicted_class}
        for s in disagreeing
    ]

    if primary_confident and not strong_disagree_family_names:
        trust = TrustState.AUTO_ACCEPT
        reason = None
        audit.append("Result: primary confident, no strong contradiction -> AUTO_ACCEPT")
    elif primary_confident and strong_disagree_family_names:
        # more than one INDEPENDENT family confidently disagreeing is treated
        # as a real consensus against the primary, not just noise
        if len(strong_disagree_family_names) >= 2:
            trust = TrustState.QUARANTINE
            reason = (f"primary confident but {len(strong_disagree_family_names)} independent "
                      f"sensor families confidently disagree: {sorted(strong_disagree_family_names)}")
            audit.append(f"Result: {reason} -> QUARANTINE")
        else:
            trust = TrustState.ACCEPT_WITH_CAUTION
            reason = f"primary confident but one family ({sorted(strong_disagree_family_names)}) confidently disagrees"
            audit.append(f"Result: {reason} -> ACCEPT_WITH_CAUTION")
    elif primary_weak or primary_weak is None:
        # primary itself is weak (or unscored, treated conservatively as weak)
        if agreeing and not disagreeing:
            trust = TrustState.ACCEPT_WITH_CAUTION
            reason = "primary weak/unscored, but supporting evidence corroborates with no contradiction"
            audit.append(f"Result: {reason} -> ACCEPT_WITH_CAUTION")
        else:
            trust = TrustState.QUARANTINE
            reason = "primary weak/unscored and supporting evidence is absent, contradictory, or itself weak"
            audit.append(f"Result: {reason} -> QUARANTINE")
    else:
        # primary neither clearly confident nor clearly weak (score exists,
        # between thresholds) - treat as cautious accept if no strong
        # contradiction, else quarantine
        if not strong_disagree_family_names:
            trust = TrustState.ACCEPT_WITH_CAUTION
            reason = "primary in the uncalibrated mid-range, no strong contradiction"
            audit.append(f"Result: {reason} -> ACCEPT_WITH_CAUTION")
        else:
            trust = TrustState.QUARANTINE
            reason = "primary in the uncalibrated mid-range with a confident contradiction"
            audit.append(f"Result: {reason} -> QUARANTINE")

    return DecisionResult(
        image_id=image_id, selected_class=selected_class, trust_state=trust,
        policy_used=policy_name, evidence_for=[primary] + agreeing,
        evidence_against=disagreeing, disagreements=disagreements,
        quarantine_reason=reason if trust in (TrustState.QUARANTINE, TrustState.ACCEPT_WITH_CAUTION) else None,
        audit_trail=audit,
    )


def _policy_gemma_primary(evidence: list[EvidenceRecord], config: DecisionConfig, image_id: str | None) -> DecisionResult:
    primary = next((e for e in evidence if e.sensor_name == "gemma"), None)
    supporting = [e for e in evidence if e.sensor_name != "gemma"]
    return _arbitrate_primary_supporting(primary, supporting, config, "gemma_primary", image_id)


def _policy_vision_primary(evidence: list[EvidenceRecord], config: DecisionConfig, image_id: str | None) -> DecisionResult:
    vision_evidence = [e for e in evidence if e.sensor_name != "gemma"]
    other = [e for e in evidence if e.sensor_name == "gemma"]

    if config.vision_primary_sensor:
        primary = next((e for e in vision_evidence if e.sensor_name == config.vision_primary_sensor), None)
        supporting = [e for e in vision_evidence if e.sensor_name != config.vision_primary_sensor] + other
        if primary is not None:
            return _arbitrate_primary_supporting(primary, supporting, config, "vision_primary", image_id)
        # named primary sensor has no evidence for this image - fall through
        # to family-consensus below rather than NO_DECISION outright

    # no named primary sensor, or it had no evidence: use the family-weighted
    # vision consensus's top class as a synthetic "primary" - a real,
    # distinct sub-path, not a silent fallback to gemma_primary
    votes = _family_votes(vision_evidence, config)
    if not votes:
        return DecisionResult(
            image_id=image_id, selected_class=None, trust_state=TrustState.NO_DECISION,
            policy_used="vision_primary", evidence_for=[], evidence_against=other,
            disagreements=[], quarantine_reason="no vision evidence available",
            audit_trail=["Policy: vision_primary", "No vision evidence at all - NO_DECISION."],
        )
    top_class, top_weight = votes.most_common(1)[0]
    total_weight = sum(votes.values())
    consensus_strength = top_weight / total_weight if total_weight else 0.0
    synthetic_primary = EvidenceRecord(
        sensor_name="vision_family_consensus", sensor_family="vision_consensus",
        predicted_class=top_class, normalized_score=consensus_strength,
        calibrated=False, metadata={"score_kind": "softmax_probability", "family_votes": dict(votes)},
    )
    supporting = other  # gemma (if present) becomes supporting/cross-check evidence
    return _arbitrate_primary_supporting(synthetic_primary, supporting, config, "vision_primary", image_id)


def _policy_weighted_fusion(evidence: list[EvidenceRecord], config: DecisionConfig, image_id: str | None) -> DecisionResult:
    """No single primary - every sensor family gets one weighted vote
    (per _family_votes), highest total wins. Trust state reflects how
    DOMINANT the winning class was relative to the full weighted vote,
    not any one sensor's own confidence score."""
    audit = ["Policy: weighted_fusion"]
    scored = [e for e in evidence if e.predicted_class]
    if not scored:
        return DecisionResult(
            image_id=image_id, selected_class=None, trust_state=TrustState.NO_DECISION,
            policy_used="weighted_fusion", evidence_for=[], evidence_against=[],
            disagreements=[], quarantine_reason="no evidence available",
            audit_trail=audit + ["No evidence at all - NO_DECISION."],
        )
    votes = _family_votes(scored, config)
    ranked = votes.most_common()
    winner, winner_weight = ranked[0]
    total = sum(votes.values())
    dominance = winner_weight / total if total else 0.0
    runner_up_weight = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = winner_weight - runner_up_weight

    audit.append(f"Family-weighted votes: {dict(votes)}")
    audit.append(f"Winner: {winner} (weight={winner_weight:.2f}/{total:.2f}, "
                  f"dominance={dominance:.2f}, margin over runner-up={margin:.2f})")

    for_evidence = [e for e in scored if e.predicted_class == winner]
    against_evidence = [e for e in scored if e.predicted_class != winner]
    disagreements = [
        {"sensor_name": e.sensor_name, "sensor_family": e.sensor_family, "predicted_class": e.predicted_class}
        for e in against_evidence
    ]

    n_families_total = len({e.sensor_family or config.family_of(e.sensor_name) or e.sensor_name for e in scored})
    if dominance >= 0.75 and margin > 0:
        trust = TrustState.AUTO_ACCEPT
        reason = None
        audit.append("Result: dominant family-weighted majority -> AUTO_ACCEPT")
    elif dominance >= 0.5 and margin > 0:
        trust = TrustState.ACCEPT_WITH_CAUTION
        reason = f"winning class has only a modest family-weighted majority (dominance={dominance:.2f})"
        audit.append(f"Result: {reason} -> ACCEPT_WITH_CAUTION")
    else:
        trust = TrustState.QUARANTINE
        reason = f"no clear family-weighted majority (dominance={dominance:.2f} across {n_families_total} families)"
        audit.append(f"Result: {reason} -> QUARANTINE")

    return DecisionResult(
        image_id=image_id, selected_class=winner, trust_state=trust,
        policy_used="weighted_fusion", evidence_for=for_evidence, evidence_against=against_evidence,
        disagreements=disagreements,
        quarantine_reason=reason if trust != TrustState.AUTO_ACCEPT else None,
        audit_trail=audit,
    )


def _policy_class_specific_authority(evidence: list[EvidenceRecord], config: DecisionConfig, image_id: str | None) -> DecisionResult:
    """
    [C] Design choice, not yet validated at scale: bootstrap a candidate
    class via a simple family-weighted plurality vote across ALL
    evidence (same mechanism as weighted_fusion), look up that
    candidate's configured primary/supporting sensors in
    config.class_specific_authority, then re-arbitrate using ONLY that
    category's designated sensors (ignoring the rest) via the same
    primary+supporting logic as gemma_primary/vision_primary. If the
    category has no entry in class_specific_authority, falls back to
    config.class_specific_fallback_policy (default: weighted_fusion)
    with the FULL evidence set, not just the bootstrap subset.

    This is a real simplification worth stating plainly: the bootstrap
    candidate could in principle be one class, while the class-specific
    primary sensor (once consulted) predicts a DIFFERENT class for this
    same image - the final selected_class in that case is whatever the
    designated primary sensor for the BOOTSTRAP candidate's category
    actually said, which may not match the bootstrap candidate itself.
    This is surfaced in the audit trail, not hidden, but is a genuine
    open design question (single-pass vs. iterate to a fixed point) -
    flagged here rather than silently resolved one way.
    """
    audit = ["Policy: class_specific_authority"]
    scored = [e for e in evidence if e.predicted_class]
    if not scored:
        return DecisionResult(
            image_id=image_id, selected_class=None, trust_state=TrustState.NO_DECISION,
            policy_used="class_specific_authority", evidence_for=[], evidence_against=[],
            disagreements=[], quarantine_reason="no evidence available",
            audit_trail=audit + ["No evidence at all - NO_DECISION."],
        )
    votes = _family_votes(scored, config)
    candidate_class, _ = votes.most_common(1)[0]
    audit.append(f"Bootstrap candidate class (family-weighted plurality): {candidate_class}")

    rule = config.class_specific_authority.get(candidate_class)
    if rule is None:
        audit.append(f"No class_specific_authority rule for {candidate_class!r} - "
                      f"falling back to policy={config.class_specific_fallback_policy!r} on full evidence.")
        fallback_fn = _POLICY_REGISTRY.get(config.class_specific_fallback_policy)
        if fallback_fn is None:
            raise ValueError(f"Unknown class_specific_fallback_policy: {config.class_specific_fallback_policy!r}")
        result = fallback_fn(evidence, config, image_id)
        result.audit_trail = audit + result.audit_trail
        result.policy_used = f"class_specific_authority(fallback={config.class_specific_fallback_policy})"
        return result

    primary_name = rule.get("primary")
    supporting_names = set(rule.get("supporting", []))
    primary = next((e for e in evidence if e.sensor_name == primary_name), None)
    supporting = [e for e in evidence if e.sensor_name in supporting_names]
    audit.append(f"Rule for {candidate_class!r}: primary={primary_name!r}, "
                  f"supporting={sorted(supporting_names)}")
    result = _arbitrate_primary_supporting(primary, supporting, config,
                                            "class_specific_authority", image_id)
    result.audit_trail = audit + result.audit_trail
    return result


_POLICY_REGISTRY: dict[str, Any] = {
    "gemma_primary": _policy_gemma_primary,
    "vision_primary": _policy_vision_primary,
    "weighted_fusion": _policy_weighted_fusion,
    "class_specific_authority": _policy_class_specific_authority,
}


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------

def decide(
    evidence: list[EvidenceRecord],
    config: DecisionConfig | None = None,
    policy: str | None = None,
    image_id: str | None = None,
) -> DecisionResult:
    """
    THE single public entry point. `policy` overrides `config.default_
    policy` for this one call (used by tests/replay tooling to compare
    policies on the same evidence without touching the config file);
    normal production callers should leave `policy=None` and drive
    behavior entirely through config/decision_engine.yaml, per Jon's
    explicit "changing the primary authority is configuration-driven"
    requirement.
    """
    config = config or DecisionConfig()
    policy_name = policy or config.default_policy
    fn = _POLICY_REGISTRY.get(policy_name)
    if fn is None:
        raise ValueError(f"Unknown policy: {policy_name!r}. Known policies: {sorted(_POLICY_REGISTRY)}")
    return fn(evidence, config, image_id)


# ---------------------------------------------------------------------
# INTEGRATION STATUS (2026-08-07)
#
# BUILT: the engine core above (EvidenceRecord/DecisionResult/TrustState,
# 4 policies, config loading, full audit trail). Deliberately consumes
# already-recorded evidence only - see diagnostics/replay_decision_
# engine.py for the replay harness that feeds it real recorded JSON from
# the Gemma same-split run and the 7-tower logit extraction, per Jon's
# "recorded evidence JSON -> Decision Engine -> route/quarantine/audit"
# first-milestone instruction.
#
# NOT YET BUILT (explicitly out of scope for this pass, per "do not
# immediately wire every production sensor into live inference"):
#   - live adapters that turn a real Gemma classify() call, a real
#     timm tower forward pass, or a real core/image_analysis.py
#     measurement into an EvidenceRecord in the actual Stage 5 flow
#   - wiring decide()'s output into core/pipeline_db.py or the bucket
#     CSV write path
#   - a QUARANTINE trust_state's effect on the actual pipeline (routing
#     to uncertain_review.csv, a review queue, etc.)
#   - row-count/row-regularity/column-grid EvidenceRecord adapters
#     (core/row_segmentation.py's detectors exist and are real, but no
#     adapter turns their output into an EvidenceRecord yet)
#   - layout-detector (core/layout_detector.py/_v26.py) EvidenceRecord
#     adapter - both are captured but unevaluated against this
#     project's own taxonomy at real scale (see the proposal doc)
#
# These are real, identified next steps, not forgotten - see docs/
# MULTI_SOURCE_VOTING_CLASSIFIER_PROPOSAL.md's Decision Engine section
# for the fuller status/rationale writeup.
# ---------------------------------------------------------------------
