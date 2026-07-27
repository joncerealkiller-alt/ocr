"""
--debug-model-inputs support: on-disk capture of the EXACT image, prompt,
and output for every model call in a row/two-stage extraction run - not
reconstructed after the fact from bbox coordinates, the actual in-memory
PIL image object handed to loader._run_generate().

Built 2026-07-24 in response to a real diagnostic gap: when a model
returns blank output or describes "only lines," there was no way to
confirm what image it actually received versus what preprocessing was
SUPPOSED to have produced - inferring it from bbox coordinates and
config values requires trusting every preprocessing step ran correctly,
exactly the thing being debugged. This module exists so that question
("what exact image did the model receive?") can be answered by opening
a PNG, not by re-deriving it.

Off by default, zero behavior/output change when disabled - every
public method on DebugModelInputRecorder is a no-op unless constructed
with enabled=True, and DebugItemRecorder likewise no-ops when its
parent recorder is disabled. Callers should construct a recorder
unconditionally and pass it through (no `if debug:` branching needed at
call sites) - see core/row_extraction.py's _extract_region() and
run_two_stage_extraction() for the wiring.

SAVE POINT: model_input.png is captured at the LAST point shared by
every model consumer, immediately before the image is handed to
loader._run_generate() - after bbox crop, column masking, tight-crop,
and upscaling, but before any model-specific internal preprocessing
(the loader's own processor/chat-template image handling, which is
loader-specific and not something this module can observe). This is
implemented via crop_region_from_source()'s debug_stage_callback
parameter (core/row_segmentation.py) - a single instrumented point in
the one function every extraction path already calls, rather than
duplicating capture logic per loader.

A save failure (disk full, permissions, whatever) is logged and
swallowed, never raised - a diagnostic aid must not be able to abort a
real extraction run that would otherwise have succeeded.
"""

from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image


class DebugModelInputRecorder:
    """
    One instance per run. Construct unconditionally; pass enabled=False
    (or just don't pass the CLI flag through) for the normal/default
    case - every method becomes a no-op and no directory is ever
    created, so there is no behavior difference from code that predates
    this feature entirely.
    """

    def __init__(
        self,
        enabled: bool,
        base_dir: str | Path = "data/debug_model_inputs",
        run_id: str | None = None,
    ):
        self.enabled = enabled
        if not enabled:
            return

        # Timestamp-based by default so concurrent/repeated runs never
        # collide and silently overwrite an earlier run's captures.
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.run_dir = Path(base_dir) / self.run_id
        self._run_meta_path = self.run_dir / "run_metadata.json"
        self._run_meta: dict[str, Any] = {
            "run_id": self.run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "items": [],
        }

        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._write_run_metadata()
        except Exception as e:
            # Downgrade to disabled rather than crash the caller - a
            # debug feature failing to initialize must not take down a
            # real extraction run. Loud print so it's not silently lost.
            print(f"[DebugModelInputRecorder] WARNING: could not create "
                  f"{self.run_dir} - debug capture disabled for this run "
                  f"({type(e).__name__}: {e})")
            self.enabled = False

    def _write_run_metadata(self) -> None:
        try:
            self._run_meta_path.write_text(
                json.dumps(self._run_meta, indent=2, default=str), encoding="utf-8"
            )
        except Exception as e:
            print(f"[DebugModelInputRecorder] WARNING: failed to write "
                  f"run_metadata.json: {type(e).__name__}: {e}")

    def new_item(self, item_id: str) -> "DebugItemRecorder":
        """
        item_id becomes the item's subdirectory name under run_dir -
        e.g. "row_0001", "header", or "row_0001/stage1" for two-stage
        extraction's two separate model calls per row (Path handles the
        nested mkdir). Safe to call even when this recorder is
        disabled - returns a no-op recorder.
        """
        if not self.enabled:
            return _NOOP_ITEM
        return DebugItemRecorder(self, item_id)

    def _register_item(self, item_id: str, item_dir: Path) -> None:
        self._run_meta["items"].append(
            {"item_id": item_id, "dir": str(item_dir.relative_to(self.run_dir))}
        )
        self._write_run_metadata()


class DebugItemRecorder:
    """
    Per-model-call capture. Created via DebugModelInputRecorder.new_item().
    Usage:
        item = recorder.new_item(f"row_{row_index:04d}")
        image = crop_region_from_source(..., debug_stage_callback=item.stage_callback())
        item.set_prompt(prompt)
        item.set_meta(source_image_path=..., row_index=..., bbox=..., model=..., ...)
        try:
            raw_output = loader._run_generate(image, prompt)
            item.finalize(raw_output=raw_output)
        except Exception as e:
            item.finalize(error=e)
    """

    def __init__(self, recorder: DebugModelInputRecorder, item_id: str):
        self.recorder = recorder
        self.item_id = item_id
        self.item_dir = recorder.run_dir / item_id
        self.meta: dict[str, Any] = {"item_id": item_id}
        self._stages: dict[str, Image.Image] = {}
        self._start_time = time.monotonic()

    def stage_callback(self) -> Callable[[str, Image.Image], None]:
        """
        Pass the return value straight to crop_region_from_source()'s
        debug_stage_callback= parameter. Copies each stage's image (the
        function that calls this mutates/reassigns its own local
        variable afterward - without a copy, every captured stage would
        end up pointing at whatever the LAST stage produced).
        """
        def _cb(stage_name: str, image: Image.Image) -> None:
            self._stages[stage_name] = image.copy()
        return _cb

    def set_prompt(self, prompt: str) -> None:
        self.meta["prompt"] = prompt

    def set_meta(self, **kwargs) -> None:
        self.meta.update(kwargs)

    def finalize(self, raw_output: str | None = None, error: Optional[BaseException] = None) -> None:
        try:
            self._finalize_unsafe(raw_output, error)
        except Exception as e:
            # Never let a debug-save failure abort or mask the real
            # extraction result - log and move on.
            print(f"[DebugModelInputRecorder] WARNING: failed to save debug "
                  f"item {self.item_id!r}: {type(e).__name__}: {e}")

    def _finalize_unsafe(self, raw_output: str | None, error: Optional[BaseException]) -> None:
        self.item_dir.mkdir(parents=True, exist_ok=True)

        # "bbox_crop" = right after the initial bbox crop (+ deskew),
        # before any masking/tight-crop/upscale - the "original crop
        # before preprocessing" the debug spec asks for. "final" = the
        # exact return value of crop_region_from_source(), i.e. the
        # exact object handed to loader._run_generate() - the
        # "model_input" save point.
        original = self._stages.get("bbox_crop")
        final = self._stages.get("final")

        if original is not None:
            original.convert("RGB").save(self.item_dir / "original_crop.png")
            self.meta["original_crop_dimensions"] = list(original.size)
        if final is not None:
            final.convert("RGB").save(self.item_dir / "model_input.png")
            self.meta["final_crop_dimensions"] = list(final.size)

        if "prompt" in self.meta:
            (self.item_dir / "prompt.txt").write_text(self.meta["prompt"], encoding="utf-8")

        self.meta["runtime_seconds"] = round(time.monotonic() - self._start_time, 3)
        self.meta["empty_output"] = not bool(raw_output and raw_output.strip())

        if error is not None:
            self.meta["exception"] = f"{type(error).__name__}: {error}"
            self.meta["traceback"] = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )
        else:
            self.meta["exception"] = None

        (self.item_dir / "output.txt").write_text(raw_output or "", encoding="utf-8")
        (self.item_dir / "metadata.json").write_text(
            json.dumps(self.meta, indent=2, default=str), encoding="utf-8"
        )

        self.recorder._register_item(self.item_id, self.item_dir)


class _NoopItemRecorder:
    """Returned by DebugModelInputRecorder.new_item() when disabled - every
    call is a no-op so callers never need an `if debug:` guard."""

    def stage_callback(self):
        return None

    def set_prompt(self, prompt: str) -> None:
        pass

    def set_meta(self, **kwargs) -> None:
        pass

    def finalize(self, raw_output: str | None = None, error: Optional[BaseException] = None) -> None:
        pass


_NOOP_ITEM = _NoopItemRecorder()

# Shared disabled instance - callers that weren't given a recorder can
# fall back to this (`debug_recorder or NOOP_RECORDER`) instead of each
# constructing their own throwaway disabled recorder.
NOOP_RECORDER = DebugModelInputRecorder(enabled=False)
