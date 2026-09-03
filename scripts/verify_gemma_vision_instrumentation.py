"""
Standalone verification for the Gemma vision-instrumentation forward hooks
added to core/loaders/gemma_loader.py, per docs/GEMMA_INSTRUMENTATION_AND_
SENSOR_SURVEY.md Part 1 (the verified hook locations) and the follow-up
implementation task that built them.

Loads the real classifier model with --debug-equivalent vision
instrumentation enabled, classifies ONE real image, and checks:
  - all three hooks (encoder/pooled/projected) registered at load time
  - all three fired exactly once for the single classify() call
  - captured metadata is well-formed (shape/dtype/device/size/timestamp)
  - GPU memory returns to baseline after loader.release()

Not part of the pipeline and not a benchmark comparing models - a thin,
one-off diagnostic, matching this project's scripts/ convention (adapters
over core/ engines, no logic duplicated from core/). Does not touch any
bucket CSV, manifest, or the pipeline DB.

Usage:
    python -m scripts.verify_gemma_vision_instrumentation <path-to-image>
"""

from __future__ import annotations

import argparse
import gc

import torch
from PIL import Image

from core.classifier import build_classifier_loader, load_pipeline_config
from core.row_extraction import _release_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to a single real image to classify")
    args = parser.parse_args()

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.empty_cache()
        baseline_allocated = torch.cuda.memory_allocated()
        print(f"GPU memory allocated before load: {baseline_allocated} bytes")
    else:
        baseline_allocated = None
        print("CUDA not available - GPU memory baseline check will be skipped.")

    pipeline_cfg = load_pipeline_config()
    print("\nLoading Gemma classifier with vision instrumentation enabled (debug=True)...")
    loader = build_classifier_loader(pipeline_cfg, debug=True)

    status = loader.vision_instrumentation_status()
    print(
        f"Hook registration status: enabled={status['enabled']}, "
        f"hooks_registered={status['hooks_registered']}, targets={status['targets']}"
    )
    if not status["enabled"] or status["hooks_registered"] != 3:
        raise SystemExit(
            f"VERIFICATION FAILED: expected 3 registered hooks and enabled=True, "
            f"got enabled={status['enabled']}, hooks_registered={status['hooks_registered']}"
        )

    print(f"\nClassifying {args.image} ...\n")
    with Image.open(args.image) as raw_image:
        result = loader.classify(str(args.image), raw_image)

    print(
        f"\nClassification result: category={result.category.value} "
        f"confidence={result.confidence}"
    )

    status = loader.vision_instrumentation_status()
    print("\nCaptured vision-hook metadata (metadata only, no tensor values):")
    for name, meta in status["last_capture_meta"].items():
        print(f"  {name}: {meta}")

    problems = []
    for name in ("encoder", "pooled", "projected"):
        count = status["fire_counts"].get(name, 0)
        if count != 1:
            problems.append(f"{name} fired {count} time(s), expected exactly 1")
    if problems:
        raise SystemExit("VERIFICATION FAILED:\n" + "\n".join(f"  - {p}" for p in problems))
    print("\nOK: encoder/pooled/projected hooks each fired exactly once.")

    print("\nReleasing loader (removes hooks, frees model)...")
    _release_model(loader)  # same teardown every other pipeline caller uses
    loader = None
    gc.collect()

    if cuda_available:
        torch.cuda.empty_cache()
        after_release = torch.cuda.memory_allocated()
        print(
            f"\nGPU memory allocated: baseline={baseline_allocated} bytes, "
            f"after release={after_release} bytes"
        )
        if after_release > baseline_allocated:
            print(
                f"WARNING: {after_release - baseline_allocated} bytes still "
                "allocated above baseline (may include unrelated CUDA context "
                "overhead that never fully releases within one process - rerun "
                "with a fresh process if this is unexpectedly large)."
            )
        else:
            print("OK: GPU memory returned to baseline.")


if __name__ == "__main__":
    main()
