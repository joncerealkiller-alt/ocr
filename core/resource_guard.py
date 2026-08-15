"""
Pre-flight system resource check - refuses to START new GPU work
(a transformers load or a vLLM subprocess launch) when host RAM or
free VRAM is already critically low, rather than attempting it and
risking an OOM that could take down the whole WSL VM with no warning.

Built 2026-08-15 as part of an OOM-hardening pass, motivated by two
real incidents the same session: (1) the backend process was found
holding ~8.65GB RSS with nothing resident - bounded now by
core/model_residency.py's _trim_host_memory(), but this guard is the
backstop for whatever that fix doesn't catch; (2) a killed-mid-launch
vLLM subprocess stranded ~13GB of VRAM+RAM with no error anywhere
until a manual `wsl --shutdown` reclaimed it - see
core/vllm_runtime.py's stray-process reconciliation, the other half
of this same hardening pass.

Deliberately conservative, fixed floors - NOT a per-model admission-
control system (that would need real per-model RAM/VRAM cost
estimates this project doesn't have yet, and would be its own,
larger piece of work). This is a blunt backstop against the SYSTEM
running out, checked once before committing to a load, not a
guarantee that specific load will fit.
"""

from __future__ import annotations

import os
import subprocess

# Both overridable via env var for testing (see tests/test_resource_guard.py) -
# production callers should never need to set these.
MIN_AVAILABLE_RAM_MB = float(os.environ.get("GENEALOGY_MIN_AVAILABLE_RAM_MB", "4096"))
MIN_FREE_VRAM_MB = float(os.environ.get("GENEALOGY_MIN_FREE_VRAM_MB", "1024"))


class ResourceExhaustedError(RuntimeError):
    """Raised by check_resources_or_raise() - callers (api/agent_main.py)
    should catch this and return a clean 503, not let it surface as an
    unhandled 500 or (worse) let the caller proceed into an OOM."""


def available_ram_mb() -> float | None:
    """Linux MemAvailable (not MemFree - MemAvailable already accounts
    for reclaimable page cache/buffers, the correct "can I actually
    allocate this much" number), falling back to psutil on a platform
    without /proc/meminfo (e.g. native Windows - see
    core/model_residency.py's _trim_host_memory() docstring for why the
    Windows/local model_console path needed this guard as much as the
    WSL backend did, confirmed live 2026-08-16: a real local model load
    produced a ~10GB transient RSS peak this function couldn't see until
    now). Returns None (not 0) only when NEITHER reader works - caller
    must treat None as "unknown," never as "exhausted.\""""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except (FileNotFoundError, OSError, ValueError, IndexError):
        pass
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 * 1024)
    except Exception:
        pass
    return None


def free_vram_mb() -> float | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip().splitlines()[0])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    return None


def current_rss_mb() -> float | None:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except (FileNotFoundError, OSError, ValueError, IndexError):
        pass
    return None


def log_resource_trend(source: str) -> None:
    """
    One-line RSS + free-VRAM log after every release (called from
    core/model_residency.py and core/vllm_runtime.py) - the visibility
    half of this hardening pass. Not a metrics system, just a plain
    print() matching this project's existing debug-logging convention
    (see CLAUDE.md/project memory: print() is fine until multiple
    instrumentation sources coexist) - lets a growth TREND be read
    directly off the log across many turns instead of needing a human
    to remember to check /debug/rss.
    """
    rss = current_rss_mb()
    vram_free = free_vram_mb()
    print(f"[{source}] post-release resources: "
          f"RSS={rss if rss is not None else '?'}MB, "
          f"free_VRAM={vram_free if vram_free is not None else '?'}MB")


def check_resources_or_raise(context: str = "") -> None:
    """
    Call immediately before committing to real GPU work (a transformers
    load in core/model_residency.py, a vLLM subprocess launch in
    core/vllm_runtime.py) - NOT on the reuse-already-resident fast
    path, since that path doesn't consume any NEW memory. `context` is
    a short label folded into the error message (e.g. the model_name
    about to be loaded) purely for a clearer error, not used for logic.

    Missing readers (RAM or VRAM unavailable, e.g. no /proc/meminfo or
    no nvidia-smi on PATH) are silently skipped, not treated as
    failures - this guard only ever blocks on a POSITIVE confirmation
    that something is low, never on "I couldn't check."
    """
    ram = available_ram_mb()
    if ram is not None and ram < MIN_AVAILABLE_RAM_MB:
        raise ResourceExhaustedError(
            f"Only {ram:.0f}MB system RAM available (floor: "
            f"{MIN_AVAILABLE_RAM_MB:.0f}MB) - refusing to start "
            f"{context or 'new GPU work'}. Release a resident model, "
            f"check for stray processes, or restart the backend."
        )
    vram = free_vram_mb()
    if vram is not None and vram < MIN_FREE_VRAM_MB:
        raise ResourceExhaustedError(
            f"Only {vram:.0f}MB free VRAM (floor: {MIN_FREE_VRAM_MB:.0f}MB) - "
            f"refusing to start {context or 'new GPU work'}. Release a "
            f"resident model first."
        )
