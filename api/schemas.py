"""
Pydantic response models for the read-only pipeline API (Phase 1, per
docs/UI_MOCKUPS_INTEGRATION_NOTES.md). Field sets mirror core/
pipeline_db.py's real schema exactly - these describe what's already
stored, not a new shape invented for the API. No field here is
computed/derived except PerformanceSnapshot, which is documented as
such on each of its own fields.
"""
from __future__ import annotations

from pydantic import BaseModel


class ImageRecord(BaseModel):
    id: int
    source_path: str
    source_type: str
    page_number: int | None
    working_path: str
    identity_hash: str
    current_hash: str | None
    current_stage: int
    status: str
    bucket: str | None
    classifier_confidence: float | None
    classifier_model: str | None
    processing_profile: str | None
    tower_consensus_category: str | None
    needs_manual_dewarp: int | None
    created_at: str
    updated_at: str


class StageOutputRecord(BaseModel):
    id: int
    image_id: int
    stage: str
    sidecar_path: str | None
    lookup_key: str | None
    sha256: str | None
    status: str
    note: str | None
    created_at: str


class GPUSnapshot(BaseModel):
    available: bool
    # Populated only when available=True - None otherwise, never a
    # fabricated 0. Sourced from `nvidia-smi`, not torch.cuda: querying
    # this process's own CUDA context would report only THIS process's
    # allocation, not real system-wide GPU state (which is what a
    # dashboard actually wants, and what another process - e.g. a
    # running classification batch - is genuinely using).
    utilization_percent: float | None = None
    memory_used_mb: float | None = None
    memory_total_mb: float | None = None
    name: str | None = None


class PerformanceSnapshot(BaseModel):
    timestamp: str
    # A single point-in-time READ, not a live/streaming value - Phase 1
    # is explicitly read-only, no push updates. cpu_percent and the
    # disk_*_bytes_per_sec fields are measured over a short (~0.3s)
    # blocking sample taken during this request, not a since-boot
    # average - see api/main.py's get_performance().
    cpu_percent: float
    ram_used_mb: float
    ram_total_mb: float
    disk_read_bytes_per_sec: float
    disk_write_bytes_per_sec: float
    gpu: GPUSnapshot
    # Derived from stage_outputs event timestamps over a recent window
    # (default 60s) - an approximation from real event history, not an
    # instrumented live counter. None if there were zero events in the
    # window (idle, not zero-cost).
    recent_images_per_sec: float | None
    recent_window_seconds: int
