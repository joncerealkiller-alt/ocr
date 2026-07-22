"""
Output schema contract for the genealogy extraction pipeline.

Design principle (from testing across 6+ documents / 11 models):
fabrication happens when models are given open-ended free-text fields
with room to generate plausible-sounding narrative. Bounding every
field to a fixed shape, fixed max length, and a mandatory confidence
tag closes off most of that surface area structurally, rather than
relying on prompt wording to discourage it.

This schema is intentionally strict. Extraction loaders (core/loaders/*)
must produce output that validates against this before it is written
to any bucket CSV or the vector DB. Anything that fails validation goes
to the uncertain/review path, not into the archive.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Classification stage (Step 2 in the pipeline) — separate, smaller contract.
# The classifier NEVER produces archive text. It only routes.
# ---------------------------------------------------------------------------

class DocumentCategory(str, Enum):
    DENSE_TABULAR_ROWS = "dense_tabular_rows"
    HANDWRITTEN_LEDGER = "handwritten_ledger"
    PRINTED_DOCUMENT = "printed_document"
    PORTRAIT_PHOTO = "portrait_photo"
    MAP_LAND_RECORD = "map_land_record"
    MIXED_TEXT_IMAGE = "mixed_text_image"
    GENEALOGY_CHART = "genealogy_chart"
    UNCERTAIN = "uncertain_review"


class ClassificationResult(BaseModel):
    file_path: str
    category: DocumentCategory
    confidence: float = Field(ge=0.0, le=1.0)
    text_density: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    handwriting: Optional[bool] = None
    table_layout: Optional[bool] = None
    faces: Optional[bool] = None
    map_like: Optional[bool] = None
    reason: str = Field(max_length=280)
    model: str
    prompt_version: str

    @field_validator("reason")
    @classmethod
    def reason_not_narrative(cls, v: str) -> str:
        # Guard against the classifier drifting into descriptive prose
        # instead of a short routing justification.
        if len(v.split()) > 60:
            raise ValueError(
                "Classification 'reason' field reads like narrative, not "
                "a short routing justification. Truncate or rephrase."
            )
        return v


# ---------------------------------------------------------------------------
# Extraction stage (Step 4) — per-bucket specialist output.
# Every field below is bounded. Confidence is mandatory per field group,
# not just as an overall score, because the failure mode observed was
# high per-record confidence masking individual fabricated entries.
# ---------------------------------------------------------------------------

class ConfidenceLevel(str, Enum):
    CONFIRMED = "confirmed"       # clearly legible, model asserts direct read
    PARTIAL = "partial"           # some characters/words uncertain
    UNCLEAR = "unclear"           # not legible; placeholder only, no guess
    NOT_PRESENT = "not_present"   # field type does not appear in this document


class PersonalName(BaseModel):
    value: str = Field(max_length=120)
    confidence: ConfidenceLevel
    row_or_position_hint: Optional[str] = Field(default=None, max_length=40)

    @field_validator("value")
    @classmethod
    def no_placeholder_as_value(cls, v: str, info) -> str:
        # If confidence is UNCLEAR/NOT_PRESENT, value should be empty or a
        # bracket placeholder, never a plausible-looking name. This is
        # enforced again at the model level in loaders, but re-checked here.
        return v.strip()


class PlaceName(BaseModel):
    value: str = Field(max_length=160)
    confidence: ConfidenceLevel


class VisibleDate(BaseModel):
    value: str = Field(max_length=60)  # verbatim, not normalized
    confidence: ConfidenceLevel


class ExtractionResult(BaseModel):
    file_path: str
    category: DocumentCategory
    document_type: Optional[str] = Field(default=None, max_length=80)

    personal_names: list[PersonalName] = Field(default_factory=list, max_length=200)
    place_names: list[PlaceName] = Field(default_factory=list, max_length=60)
    visible_dates: list[VisibleDate] = Field(default_factory=list, max_length=20)
    subject_keywords: list[str] = Field(default_factory=list, max_length=25)

    raw_model_output_len: int  # length of what the model actually produced,
                                 # for anomaly detection (see below)
    model: str
    prompt_version: str
    generation_config_hash: str  # ties output to exact repetition_penalty /
                                    # temperature / etc. used, for audit trail

    # Execution mode - lets a slow or degraded run be traced back to its
    # cause after the fact, rather than just seeing "this one took forever"
    # with no explanation. device_map/max_memory reflect what was actually
    # passed to from_pretrained; oom_recovered flags whether this specific
    # record required a retry after a CUDA OOM on a prior attempt.
    device_map: Optional[str] = None
    max_memory: Optional[dict] = None
    vram_headroom_gb: Optional[float] = None
    cpu_offload_limit_gb: Optional[float] = None
    oom_recovered: bool = False

    @field_validator("subject_keywords")
    @classmethod
    def keywords_bounded(cls, v: list[str]) -> list[str]:
        for kw in v:
            if len(kw) > 60:
                raise ValueError(f"Keyword too long, looks like narrative leak: {kw!r}")
        return v

    @field_validator("personal_names")
    @classmethod
    def flag_suspicious_name_count(cls, v: list[PersonalName]) -> list[PersonalName]:
        # Soft structural guard, not a hard block: a document producing
        # >80 CONFIRMED names in one pass matches the fabrication pattern
        # seen repeatedly in testing (InternVL3-8b, Qwen-4b/7b generating
        # long fluent lists). This doesn't reject the record — it's meant
        # to be checked by the anomaly-flagging pass downstream — but it's
        # asserted here so the condition is visible at the schema level.
        confirmed = [n for n in v if n.confidence == ConfidenceLevel.CONFIRMED]
        if len(confirmed) > 80:
            raise ValueError(
                f"{len(confirmed)} CONFIRMED names in a single record exceeds "
                "the sanity ceiling. Route to uncertain_review for manual check "
                "rather than accepting as-is."
            )
        return v


# ---------------------------------------------------------------------------
# Anomaly flags — computed post-hoc across a batch, not per-record.
# See core/anomaly.py (to be built) for the actual detection logic.
# This model just defines what a flag looks like once found.
# ---------------------------------------------------------------------------

class AnomalyFlag(BaseModel):
    file_path: str
    flag_type: str  # e.g. "no_ngram_overlap", "length_outlier", "repeated_entry"
    detail: str = Field(max_length=300)
    severity: str = Field(pattern="^(low|medium|high)$")
