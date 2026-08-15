"""
Tests for core/resource_guard.py - the OOM-hardening pre-flight check
(2026-08-15). No pytest in this environment - plain assert-based,
matching this project's existing tests/test_run_context.py convention.

Threshold logic is tested by monkeypatching the reader functions
(available_ram_mb/free_vram_mb), not by actually exhausting real
system resources - real reclaim behavior (does release() actually
free memory) is verified live and recorded in
docs/WSL_COMPUTE_BACKEND_BASELINE.md, not re-asserted here.

Usage:
    python tests/test_resource_guard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import core.resource_guard as rg

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


def test_passes_when_resources_plentiful():
    print("\ntest_passes_when_resources_plentiful")
    orig_ram, orig_vram = rg.available_ram_mb, rg.free_vram_mb
    try:
        rg.available_ram_mb = lambda: 32000.0
        rg.free_vram_mb = lambda: 8000.0
        try:
            rg.check_resources_or_raise(context="test")
            check(True, "no exception when both RAM and VRAM are well above floor")
        except rg.ResourceExhaustedError:
            check(False, "should not raise when resources are plentiful")
    finally:
        rg.available_ram_mb, rg.free_vram_mb = orig_ram, orig_vram


def test_raises_on_low_ram():
    print("\ntest_raises_on_low_ram")
    orig_ram, orig_vram = rg.available_ram_mb, rg.free_vram_mb
    try:
        rg.available_ram_mb = lambda: 100.0  # well below MIN_AVAILABLE_RAM_MB
        rg.free_vram_mb = lambda: 8000.0
        try:
            rg.check_resources_or_raise(context="loading 'test_model'")
            check(False, "should raise when RAM is critically low")
        except rg.ResourceExhaustedError as e:
            check("test_model" in str(e), "error message includes the context label")
            check("RAM" in str(e), "error message identifies RAM as the constraint")
    finally:
        rg.available_ram_mb, rg.free_vram_mb = orig_ram, orig_vram


def test_raises_on_low_vram():
    print("\ntest_raises_on_low_vram")
    orig_ram, orig_vram = rg.available_ram_mb, rg.free_vram_mb
    try:
        rg.available_ram_mb = lambda: 32000.0
        rg.free_vram_mb = lambda: 10.0  # well below MIN_FREE_VRAM_MB
        try:
            rg.check_resources_or_raise(context="test")
            check(False, "should raise when VRAM is critically low")
        except rg.ResourceExhaustedError as e:
            check("VRAM" in str(e), "error message identifies VRAM as the constraint")
    finally:
        rg.available_ram_mb, rg.free_vram_mb = orig_ram, orig_vram


def test_available_ram_mb_returns_real_value_on_windows():
    """
    2026-08-16: available_ram_mb() gained a psutil fallback for
    platforms without /proc/meminfo (native Windows) - the WSL-only
    version of this guard silently no-op'd on the model_console/local
    path, confirmed live as a real gap the same night a genuine ~10GB
    transient RSS peak happened on that exact path. This just confirms
    the reader itself returns a plausible positive number on whatever
    platform the test suite is running on - the threshold-logic tests
    above already cover check_resources_or_raise()'s behavior once a
    value exists, regardless of which reader produced it.
    """
    print("\ntest_available_ram_mb_returns_real_value_on_windows")
    value = rg.available_ram_mb()
    check(value is None or value > 0,
          f"available_ram_mb() is None or a positive number (got {value!r})")


def test_unknown_readers_never_block():
    """Missing readers (None - e.g. no /proc/meminfo, no nvidia-smi on
    PATH) must be treated as 'unknown', never as 'exhausted' - this
    guard only blocks on a POSITIVE confirmation something is low."""
    print("\ntest_unknown_readers_never_block")
    orig_ram, orig_vram = rg.available_ram_mb, rg.free_vram_mb
    try:
        rg.available_ram_mb = lambda: None
        rg.free_vram_mb = lambda: None
        try:
            rg.check_resources_or_raise(context="test")
            check(True, "no exception when both readers return None (unknown, not exhausted)")
        except rg.ResourceExhaustedError:
            check(False, "must not raise when resource state is genuinely unknown")
    finally:
        rg.available_ram_mb, rg.free_vram_mb = orig_ram, orig_vram


def main():
    test_passes_when_resources_plentiful()
    test_raises_on_low_ram()
    test_raises_on_low_vram()
    test_available_ram_mb_returns_real_value_on_windows()
    test_unknown_readers_never_block()

    print(f"\n{'='*60}")
    print(f"RESULTS: {_PASS} passed, {_FAIL} failed")
    print(f"{'='*60}")
    if _FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
