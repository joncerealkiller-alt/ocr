"""
Test for core/model_residency.py's _trim_host_memory() - the glibc
malloc_trim(0) call added to _release_current() after finding live
2026-08-15 that the WSL backend process held ~8.65GB RSS with NO model
resident and GPU VRAM already idle (a separate leak from the earlier
CUDA-allocator VRAM issue - see docs/WSL_COMPUTE_BACKEND_BASELINE.md).

Deliberately does NOT assert exact RSS numbers - real host-memory
reclaim behavior is glibc/allocator-state-dependent and not
deterministic enough to pin in a unit test (the real verification for
this fix is the live 3-cycle test recorded in that doc: RSS growth per
release cycle converged 305MB -> 36.5MB -> 24.3MB rather than growing
unbounded). This test only confirms the function is safe to call -
does not raise, is a no-op on a non-glibc host, doesn't require a GPU
or a loaded model.

No pytest in this environment - plain assert-based, matching this
project's existing tests/test_run_context.py convention.

Usage:
    python tests/test_model_residency_memory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.model_residency import _trim_host_memory

_PASS = 0
_FAIL = 0


def check(condition: bool, description: str) -> None:
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  PASS: {description}")
    else:
        _FAIL += 1
        print(f"  FAIL: {description}")


def test_trim_host_memory_does_not_raise():
    print("\ntest_trim_host_memory_does_not_raise")
    try:
        _trim_host_memory()
        check(True, "_trim_host_memory() completes without raising")
    except Exception as e:
        check(False, f"_trim_host_memory() raised {type(e).__name__}: {e}")


def test_trim_host_memory_is_idempotent():
    """Calling it repeatedly (e.g. release_all() then a manual
    /debug/trim_memory call) must be safe - it's just asking glibc to
    release what it can, never destructive."""
    print("\ntest_trim_host_memory_is_idempotent")
    try:
        _trim_host_memory()
        _trim_host_memory()
        _trim_host_memory()
        check(True, "three consecutive calls all complete without raising")
    except Exception as e:
        check(False, f"repeated calls raised {type(e).__name__}: {e}")


def main():
    test_trim_host_memory_does_not_raise()
    test_trim_host_memory_is_idempotent()

    print(f"\n{'='*60}")
    print(f"RESULTS: {_PASS} passed, {_FAIL} failed")
    print(f"{'='*60}")
    if _FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
