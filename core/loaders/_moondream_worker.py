"""
Standalone worker process for moondream2, run under .venv_moondream (a
separate venv pinned to transformers==4.52.4, --system-site-packages so it
shares the main environment's torch/CUDA install) rather than the main
project venv (transformers 5.12.1).

Why a subprocess instead of just another loader class: moondream2's
trust_remote_code model (hf_moondream.py in the cached snapshot) produces
correct output on transformers 4.52.4 - its own declared version - but
silently degenerates to garbage ("1" followed by hundreds of blank-token
lines) on transformers 5.12.1, which every other loader in this project
runs on. That's a real incompatibility in moondream2's custom
attention/KV-cache code against current transformers internals, not
something fixable by patching from the loader side (a narrower
tied-weights-attribute gap in the same vein WAS patchable - see
MoondreamLoader - but this deeper one isn't). Two transformers versions
can't coexist in one Python process, so moondream2 gets its own venv and
talks to the main pipeline over stdin/stdout instead.

Protocol: newline-delimited JSON on stdin/stdout.
  Request:  {"image_path": str, "question": str, "reasoning": bool}
  Response: {"answer": str} or {"error": str}
A single literal line "READY" is printed to stdout once the model has
finished loading, before the request loop starts - the parent process
blocks on reading that line to know the worker is usable.

Run standalone: .venv_moondream/Scripts/python.exe _moondream_worker.py
<repo_id> <revision>
"""

from __future__ import annotations

import json
import sys


def _block_lora_network_calls(model) -> None:
    """
    See MoondreamLoader's module docstring (core/loaders/moondream_loader.py)
    for the full explanation of what this blocks and why - identical logic,
    duplicated here because this file runs as a standalone script in a
    separate venv and can't import the main package's loader module.
    """
    model_module_name = type(model).__module__
    prefix = model_module_name.rsplit(".", 1)[0]
    lora_module = sys.modules.get(f"{prefix}.lora")
    if lora_module is None:
        print(
            f"[moondream_worker] WARNING: could not locate {prefix}.lora "
            "to disable its network call - LoRA variant download path "
            "was NOT patched.",
            file=sys.stderr,
        )
        return

    def _blocked_cached_variant_path(variant_id, *args, **kwargs):
        raise RuntimeError(
            f"[moondream_worker] Blocked an attempt to download LoRA "
            f"variant {variant_id!r} from the network. This worker runs "
            "moondream2 offline-only; variant/settings kwargs must not be "
            "passed to query()."
        )

    lora_module.cached_variant_path = _blocked_cached_variant_path


def main() -> None:
    if len(sys.argv) != 3:
        print(
            "usage: _moondream_worker.py <repo_id> <revision>",
            file=sys.stderr,
        )
        sys.exit(2)

    repo_id, revision = sys.argv[1], sys.argv[2]

    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        revision=revision,
        trust_remote_code=True,
        torch_dtype="auto",
        token=False,
        local_files_only=True,
    ).to(device).eval()

    _block_lora_network_calls(model)

    print("READY", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            with Image.open(request["image_path"]) as raw_image:
                if raw_image.mode != "RGB":
                    raw_image = raw_image.convert("RGB")
                # No `settings=` kwarg - see _block_lora_network_calls and
                # the module docstring above.
                result = model.query(
                    image=raw_image,
                    question=request["question"],
                    reasoning=bool(request.get("reasoning", False)),
                    stream=False,
                )
            response = {"answer": result["answer"].strip()}
        except Exception as e:
            response = {"error": f"{type(e).__name__}: {e}"}

        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
