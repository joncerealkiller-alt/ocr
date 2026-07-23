"""
Worker script for moondream2, run under .venv_moondream (a separate
venv pinned to transformers==4.52.4, --system-site-packages so it
shares the main environment's torch/CUDA install) rather than the main
project venv (transformers 5.12.1).

Why a subprocess instead of just another loader class: moondream2's
trust_remote_code model (hf_moondream.py in the cached snapshot) produces
correct output on transformers 4.52.4 - its own declared version - but
silently degenerates to garbage ("1" followed by hundreds of blank-token
lines) on transformers 5.12.1, which every other loader in this project
runs on. That's a real incompatibility in moondream2's custom
attention/KV-cache code against current transformers internals, not
something fixable by patching from the loader side. Two transformers
versions can't coexist in one Python process, so moondream2 gets its own
venv and talks to the main pipeline over stdin/stdout instead.

Uses the shared request-loop helper (_subprocess_worker_common.run_worker)
rather than hand-rolling the JSON I/O loop - this file only supplies the
two things actually specific to moondream2: how to load it, and how to
call it. See that module's docstring for the full wire protocol.

Run standalone: .venv_moondream/Scripts/python.exe _moondream_worker.py
<repo_id> <revision>
"""

from __future__ import annotations

import sys

# This worker's own directory isn't necessarily on sys.path when run as
# a standalone script from a different venv/cwd - added explicitly so
# `from _subprocess_worker_common import run_worker` resolves regardless
# of how/where this script is invoked from.
sys.path.insert(0, __file__.rsplit("/", 1)[0].rsplit("\\", 1)[0])

from _subprocess_worker_common import run_worker


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


def _load():
    import torch
    from transformers import AutoModelForCausalLM

    repo_id, revision = sys.argv[1], sys.argv[2]
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
    return model


def _query(model, image, question, extra):
    # No `settings=` kwarg - see _block_lora_network_calls and the
    # module docstring above.
    result = model.query(
        image=image,
        question=question,
        reasoning=bool(extra.get("reasoning", False)),
        stream=False,
    )
    return result["answer"]


if __name__ == "__main__":
    run_worker(_load, _query)
