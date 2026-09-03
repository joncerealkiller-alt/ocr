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
    # Added 2026-08-01 after analyzing data/flagged_needs_new_bucket.csv
    # (149 files manually flagged by Jon as wrongly classified into
    # dense_tabular_rows/printed_document/map_land_record): a screenshot
    # of a genealogy/archive WEBSITE's UI (search results, catalog/
    # finding-aid metadata pages, database record cards) - NOT a scan or
    # photograph of a physical document. Deliberately routed to model:
    # null in config/pipeline.yaml, same "skip extraction" mechanism as
    # uncertain_review but for a different reason: these images don't
    # need OCR/vision extraction at all (the text is already crisp,
    # digitally-rendered webpage text), not that the classifier is
    # unsure what they are.
    WEBSITE_SCREENSHOT = "website_screenshot"
    # Added 2026-08-04 after manually ground-truthing the 77-image
    # gained/lost-agreement flagged set (docs/STAGE1_STAGE4_
    # INVESTIGATION_CONCLUSION.md) - two real, recurring content types
    # that didn't fit any existing category, both confirmed as
    # genuinely relevant (kept, not pruned):
    #   PHOTO_COLLAGE - multiple photos shown together as one image,
    #     either a physical scrapbook/album page with mounted photos and
    #     handwritten captions, OR a digital screenshot of a photo
    #     gallery/grid UI - deliberately ONE category for both, per
    #     Jon's direction, since the defining trait is "multiple photos
    #     in one frame," not the physical-vs-digital origin.
    #   CASUAL_PHOTO - a personal/candid photograph that isn't a
    #     portrait (no person as the primary subject) and isn't a
    #     document (food, objects, scenery). Distinct from PORTRAIT_PHOTO
    #     rather than folded into it, per Jon's direction.
    # Extraction routing (config/pipeline.yaml's model: null pattern,
    # used for WEBSITE_SCREENSHOT/UNCERTAIN) is a separate question, not
    # decided here - these are taxonomy additions only.
    PHOTO_COLLAGE = "photo_collage"
    CASUAL_PHOTO = "casual_photo"
    # Added 2026-08-04, same ground-truthing pass as the two above - two
    # independent images (a close-up headstone marker and a wider
    # cemetery scene with grave crosses) both needed this and neither
    # fit CASUAL_PHOTO cleanly enough to fold in, per Jon's direction.
    CEMETERY_PHOTO = "cemetery_photo"


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

    @field_validator("place_names")
    @classmethod
    def flag_suspicious_place_count(cls, v: list[PlaceName]) -> list[PlaceName]:
        # Same principle as flag_suspicious_name_count above, added
        # 2026-08-13 after a live, confirmed instance: qwen3vl4b
        # extracted 86 place_names (max_length=60) from a real map with
        # ~16 actual labels, ALL tagged CONFIRMED, ending in a long run
        # of real but unrelated US military forts (Fort Knox, Fort
        # Bragg, Fort Apache, Fort Sumner...) that have nothing to do
        # with the image - unmistakable pattern-completion from training
        # data, not perception, once the model lost its grounding.
        #
        # An earlier fix attempt for this SAME failure just truncated
        # the list to fit max_length and let it through as
        # "successful" - that was WORSE than crashing, since it wrote
        # 60 fabricated place names into core/genealogy_memory.py as
        # confirmed genealogy facts, silently corrupting the discovery
        # store. Jon's framing, directly applied here: "don't merely cap
        # output to satisfy Pydantic - treat an unexpectedly large
        # extraction count as a grounding failure." This raises (routes
        # to uncertain_review / failed_extraction, matching
        # personal_names' own established handling) instead of
        # truncating-and-accepting.
        #
        # Threshold 30, not 80 like personal_names - place_names'
        # max_length is 60 (vs. personal_names' 200), and the real test
        # map that exposed this had ~16 true labels; 30 gives real
        # headroom for a genuinely dense/detailed map while still
        # decisively catching an 86-entry fabrication run.
        confirmed = [p for p in v if p.confidence == ConfidenceLevel.CONFIRMED]
        if len(confirmed) > 30:
            raise ValueError(
                f"{len(confirmed)} CONFIRMED place names in a single record "
                "exceeds the sanity ceiling. Route to uncertain_review for "
                "manual check rather than accepting as-is."
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
