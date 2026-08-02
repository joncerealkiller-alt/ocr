"""
Worker script for tencent/HunyuanOCR, run under .venv_hunyuan_ocr (a
separate venv pinned to transformers==5.14.1, --system-site-packages so
it shares the main environment's torch/CUDA install) rather than the
main project venv (transformers 5.12.1).

WHY A SUBPROCESS, despite this NOT being a trust_remote_code/custom-code
situation (confirmed 2026-07-28, unlike moondream2 and deepseek-vl2-tiny):
HunyuanOCR ships no .py modeling files and no auto_map at all - it's a
genuinely native transformers architecture (`HunYuanVLForConditionalGeneration`,
lives at transformers.models.hunyuan_vl.modeling_hunyuan_vl), confirmed by
loading it with trust_remote_code omitted entirely. The only reason this
still needs isolation is that native support was only merged into
mainline transformers as of 5.13.0 - this project's main venv is pinned
at 5.12.1 (pixtral/gemma/qwen/etc. all load in-process against that
version). Bumping the shared main transformers install to satisfy one
new candidate model risks silently changing behavior for every other
in-process loader - not worth that risk for a prep/sanity-test model
Jon hasn't evaluated yet. Isolating in its own venv sidesteps the
question entirely, same spirit as the moondream/deepseek-vl2 venvs even
though the underlying reason (newer- not older-than-main version) is
the opposite.

No monkeypatches or ABI workarounds needed here (unlike deepseek-vl2's
worker) - transformers 5.14.1 installs cleanly on Python 3.14 with real
prebuilt wheels throughout, and the model calls through the same
apply_chat_template -> generate -> batch_decode convention every other
in-process loader in this project already uses (near-identical to
PixtralLoader's own _run_generate). The generation call itself needed
repetition_penalty/no_repeat_ngram_size to avoid the same repetition-
loop degeneration documented in pixtral_12b.yaml's own comment - the
model card's bare example (greedy, no penalty) looped ("(4)" repeated
to the token budget) on a real project sample; this project's already-
proven pixtral defaults (repetition_penalty=1.15, no_repeat_ngram_size=5)
fixed it immediately, confirmed on a real sample image.

Uses the shared request-loop helper (_subprocess_worker_common.run_worker)
- see that module's docstring for the wire protocol.

Run standalone: .venv_hunyuan_ocr/Scripts/python.exe _hunyuan_ocr_worker.py
<repo_id> <revision>
"""

from __future__ import annotations

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0].rsplit("\\", 1)[0])

from _subprocess_worker_common import run_worker


def _load():
    import torch
    from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

    repo_id = sys.argv[1]

    processor = AutoProcessor.from_pretrained(repo_id, backend="pil", local_files_only=True)
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        repo_id, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True,
    ).eval()

    return {"model": model, "processor": processor}


def _query(bundle, image, question, extra):
    import torch

    model = bundle["model"]
    processor = bundle["processor"]

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=extra.get("max_new_tokens", 512),
        do_sample=extra.get("do_sample", False),
    )
    if gen_kwargs["do_sample"]:
        gen_kwargs["temperature"] = extra.get("temperature", 0.1)
        gen_kwargs["top_p"] = extra.get("top_p", 0.9)
    if extra.get("repetition_penalty"):
        gen_kwargs["repetition_penalty"] = extra["repetition_penalty"]
    if extra.get("no_repeat_ngram_size"):
        gen_kwargs["no_repeat_ngram_size"] = extra["no_repeat_ngram_size"]

    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)

    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0]


if __name__ == "__main__":
    run_worker(_load, _query)
