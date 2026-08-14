"""
Phase 1 of the read-only pipeline API, per docs/UI_MOCKUPS_INTEGRATION_
NOTES.md and Jon's own scope: FastAPI, five GET endpoints, no live
updates (no WebSocket/SSE/polling loop anywhere in this module) - every
response is computed fresh from core/pipeline_db.py at request time and
nothing else. This module never writes to the DB.

Deliberately thin: all real queries live in core/pipeline_db.py (the
project's established "PipelineDatabase owns the schema/queries, callers
never write raw SQL against the DB file" discipline) - this module only
does HTTP plumbing, request validation, and response shaping.

Endpoints:
    GET /images                - list images, filterable (bucket, status,
                                  current_stage), paginated (limit, offset)
    GET /image/{image_id}      - one image's current state
    GET /stage/{image_id}      - that image's full stage_outputs history
                                  (filterable by ?stage=), NOT a lookup by
                                  stage_outputs.id - see docstring below
    GET /logs                  - corpus-wide stage_outputs, most recent
                                  first; this project has no separate
                                  structured-logging system, so this
                                  exposes the real event log that already
                                  exists (stage_outputs) rather than
                                  fabricating a second one
    GET /performance            - a single point-in-time system snapshot
                                  (CPU/RAM/disk sampled over ~0.3s during
                                  the request, GPU via `nvidia-smi`
                                  subprocess, recent throughput derived
                                  from real stage_outputs timestamps) -
                                  NOT a live/streaming value

Usage:
    uvicorn api.main:app --reload --port 8000
    (then GET http://127.0.0.1:8000/docs for the auto-generated Swagger UI)
"""
from __future__ import annotations

import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psutil
from fastapi import FastAPI, HTTPException, Query

from api.schemas import GPUSnapshot, ImageRecord, PerformanceSnapshot, StageOutputRecord
from core.pipeline_db import DEFAULT_DB_PATH, PipelineDatabase

app = FastAPI(
    title="Genealogy Pipeline API (Phase 1 - read-only)",
    description="Read-only view over core/pipeline_db.py. No live updates.",
    version="0.1.0",
)

_db = PipelineDatabase(DEFAULT_DB_PATH)

_RECENT_THROUGHPUT_SAMPLE_LIMIT = 1000  # see get_performance()'s own comment
_PERFORMANCE_SAMPLE_SECONDS = 0.3       # blocking window for CPU%/disk-rate sampling


@app.get("/images", response_model=list[ImageRecord])
def list_images(
    bucket: str | None = None,
    status: str | None = None,
    current_stage: int | None = None,
    limit: int = Query(default=200, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
) -> list[dict]:
    """
    Filters are the exact same three PipelineDatabase.list_images()
    already supports - no new filtering logic invented here. Pagination
    (limit/offset) is applied in this layer, not pushed into
    list_images() itself, since that method's own signature is used
    elsewhere in the project and Phase 1 doesn't need to change it for
    a corpus this size (~1750 images today).
    """
    images = _db.list_images(current_stage=current_stage, status=status, bucket=bucket)
    return images[offset: offset + limit]


@app.get("/image/{image_id}", response_model=ImageRecord)
def get_image(image_id: int) -> dict:
    image = _db.get_image(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail=f"No image with id={image_id}")
    return image


@app.get("/stage/{image_id}", response_model=list[StageOutputRecord])
def get_stage_outputs(image_id: int, stage: str | None = None) -> list[dict]:
    """
    {image_id} is an images.id, not a stage_outputs.id - returns that
    IMAGE's full stage history (optionally narrowed to one stage name),
    matching the mockups' per-image Vision/Routing/OCR tabs, which are
    all "what happened to THIS image at each stage," not a lookup of one
    isolated event row. 404 only when the image itself doesn't exist -
    an image with zero stage_outputs yet (not an error) correctly
    returns an empty list, not a 404.
    """
    if _db.get_image(image_id) is None:
        raise HTTPException(status_code=404, detail=f"No image with id={image_id}")
    return _db.get_stage_outputs(image_id, stage=stage)


@app.get("/logs", response_model=list[StageOutputRecord])
def get_logs(
    stage: str | None = None,
    status: str | None = None,
    limit: int = Query(default=100, ge=1, le=2000),
) -> list[dict]:
    """
    Corpus-wide, most-recent-first. stage_outputs is already a
    structured, timestamped, append-only event log in everything but
    name (image_id, stage, status, note, created_at) - this exposes it
    directly. NOT the same shape as the mockups' human-readable INFO/WARN
    log lines (e.g. "Detected handwriting on row 17") - this project has
    no such logging system yet (every stage script currently print()s to
    stdout, per docs/UI_MOCKUPS_INTEGRATION_NOTES.md), so this is the
    closest real data available, not a fabricated match to the mockup's
    exact copy.
    """
    return _db.list_stage_outputs(stage=stage, status=status, limit=limit)


def _query_gpu() -> GPUSnapshot:
    """
    Shells out to `nvidia-smi` rather than querying torch.cuda in this
    process - torch.cuda.memory_allocated() would report only THIS
    process's own CUDA allocation, not real system-wide GPU state, which
    is what a dashboard wants and what a genuinely running OTHER process
    (e.g. a classification batch) is actually using. Also avoids ever
    creating a CUDA context in the API process at all, which matters
    while a real inference run may be active elsewhere on the same GPU
    (CLAUDE.md's GPU-contention discipline) - this function never
    imports torch.

    Returns available=False (all other fields None) if nvidia-smi isn't
    on PATH, times out, or its output doesn't parse - never fabricates a
    0 in place of "unknown."
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return GPUSnapshot(available=False)
        name, util, mem_used, mem_total = (p.strip() for p in result.stdout.strip().split(",")[:4])
        return GPUSnapshot(
            available=True, name=name,
            utilization_percent=float(util), memory_used_mb=float(mem_used),
            memory_total_mb=float(mem_total),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return GPUSnapshot(available=False)


def _recent_images_per_sec(window_seconds: int) -> float | None:
    """
    Derived from real stage_outputs timestamps, not an instrumented live
    counter - counts events with created_at within the last
    window_seconds, divided by the window. Samples the
    _RECENT_THROUGHPUT_SAMPLE_LIMIT most recent stage_outputs rows
    (generous relative to any realistic per-window event count for this
    project's actual throughput) rather than scanning the whole table.
    Returns None (not 0.0) when there were zero events in the window -
    idle is a different state from "measured and found to be zero."
    """
    recent = _db.list_stage_outputs(limit=_RECENT_THROUGHPUT_SAMPLE_LIMIT)
    if not recent:
        return None
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
    count = 0
    for row in recent:
        try:
            ts = datetime.fromisoformat(row["created_at"])
        except ValueError:
            continue
        if ts >= cutoff:
            count += 1
    return round(count / window_seconds, 3) if count else None


@app.get("/performance", response_model=PerformanceSnapshot)
def get_performance() -> PerformanceSnapshot:
    """
    A single point-in-time read, computed fresh on every request - not a
    live/streaming value (Phase 1 has no WebSocket/SSE anywhere). CPU%
    and disk I/O rate are measured over a short blocking sample taken
    during THIS request (see _PERFORMANCE_SAMPLE_SECONDS), not a
    since-process-start average, so the request has a small fixed
    latency floor (~0.3s) - acceptable for a manually-refreshed
    dashboard, not for anything wanting sub-100ms responses.
    """
    disk_before = psutil.disk_io_counters()
    cpu_percent = psutil.cpu_percent(interval=_PERFORMANCE_SAMPLE_SECONDS)
    disk_after = psutil.disk_io_counters()

    read_rate = (disk_after.read_bytes - disk_before.read_bytes) / _PERFORMANCE_SAMPLE_SECONDS
    write_rate = (disk_after.write_bytes - disk_before.write_bytes) / _PERFORMANCE_SAMPLE_SECONDS

    mem = psutil.virtual_memory()
    window = 60
    return PerformanceSnapshot(
        timestamp=datetime.now(timezone.utc).isoformat(),
        cpu_percent=cpu_percent,
        ram_used_mb=round(mem.used / 1e6, 1),
        ram_total_mb=round(mem.total / 1e6, 1),
        disk_read_bytes_per_sec=round(read_rate, 1),
        disk_write_bytes_per_sec=round(write_rate, 1),
        gpu=_query_gpu(),
        recent_images_per_sec=_recent_images_per_sec(window),
        recent_window_seconds=window,
    )
