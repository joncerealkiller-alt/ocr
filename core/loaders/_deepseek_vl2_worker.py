"""
Worker script for deepseek-ai/deepseek-vl2-tiny, run under
.venv_deepseek_vl2 (a separate venv pinned to transformers==4.38.2,
--system-site-packages so it shares the main environment's torch/CUDA
install) rather than the main project venv (transformers 5.12.1).

WHY A SUBPROCESS (confirmed 2026-07-28, same class of problem as
moondream2 - see moondream_loader.py's docstring): deepseek-vl2-tiny's
config.json declares model_type "deepseek_vl_v2" with no built-in
support in transformers 5.12.1 at all - AutoConfig.from_pretrained()
raises `KeyError: 'deepseek_vl_v2'` outright (confirmed by direct test,
not a guess). Unlike moondream2, this repo doesn't even ship its own
trust_remote_code .py files for dynamic loading - the real "how do I
load this" path is DeepSeek's own deepseek_vl2 Python package (cloned
from https://github.com/deepseek-ai/DeepSeek-VL2, kept at
.venv_deepseek_vl2/DeepSeek-VL2-src/ - NOT pip-installable, not on
PyPI), imported directly as `from deepseek_vl2.models import
DeepseekVLV2ForCausalLM, DeepseekVLV2Processor`. Their requirements.txt
pins transformers==4.38.2, so that's what this venv installs.

TWO FURTHER REAL, CONFIRMED ISSUES beyond the version pin (both worked
around below, not guessed around):

1. tokenizers ABI: transformers 4.38.2 declares "tokenizers>=0.14,<0.19"
   - every version in that range predates Python 3.14 (this project's
   main interpreter) and has no prebuilt wheel for it, so pip falls back
   to compiling tokenizers' Rust extension from source via maturin. That
   freshly-built extension SEGFAULTS on import against Python 3.14's C
   API (confirmed: crashes inside AutoTokenizer.from_pretrained, exit
   code 139, before printing anything). Fix: install a modern tokenizers
   (0.21.4, real cp39-abi3 wheel - forward-compatible, no compile step)
   and loosen the version floor/ceiling string baked into this venv's
   own installed copy of transformers/dependency_versions_table.py
   (a disposable, gitignored venv's vendored copy, not a project file -
   same spirit as this file patching lora.py's network call below).

2. xformers has no working CUDA backend on this GPU at all right now:
   deepseek_vl2's SigLIP vision encoder (siglip_vit.py) unconditionally
   imports `xformers.ops.memory_efficient_attention` with no fallback
   when qk_norm is off (confirmed: this checkpoint's vision_config has
   no qk_norm key, so the xformers branch is the one that actually
   runs). The only installable xformers wheel is a stub built for
   PyTorch 2.10/cu128/Python 3.10 - it explicitly refuses this GPU
   ("requires device with capability <= (9,0) but your GPU has
   capability (12,0) (too new)" - RTX 5060 Ti is Blackwell/compute
   capability 12.0, newer than any current xformers prebuilt kernel).
   Fix below: monkeypatch siglip_vit.Attention.forward to always use
   torch's native scaled_dot_product_attention instead. This is
   correctness-preserving, not an approximation: q_norm/k_norm are
   nn.Identity() whenever qk_norm is off (true for this checkpoint), so
   the only thing that changes is which attention kernel runs the exact
   same math - see the two forward() branches in the upstream file.

Uses the shared request-loop helper (_subprocess_worker_common.run_worker)
like every other worker script - see that module's docstring for the
wire protocol. One extra wrinkle specific to this model: DeepSeek's own
processor/tokenizer setup code calls bare print() (not logging/warnings)
for its special-token debug messages, e.g. "Add pad token = [...]" -
if that lands on real stdout it corrupts the JSON request/response wire
protocol (the parent's readline() would read a debug line instead of
"READY" or a JSON response). _load() below redirects stdout to stderr
for the whole load, restoring it before run_worker() prints "READY".

Run standalone: .venv_deepseek_vl2/Scripts/python.exe _deepseek_vl2_worker.py
<repo_id> <revision>
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/", 1)[0].rsplit("\\", 1)[0])

from _subprocess_worker_common import run_worker

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VENDORED_SRC = PROJECT_ROOT / ".venv_deepseek_vl2" / "DeepSeek-VL2-src"


def _patch_vision_attention_to_native_sdpa() -> None:
    """See module docstring, issue 2. Must run after deepseek_vl2.models
    is imported (patches the class in place) and before any forward
    pass - safe to call once at load time since it replaces the method
    on the class itself, affecting every Attention instance."""
    import torch.nn.functional as F
    from deepseek_vl2.models import siglip_vit

    def _native_sdpa_forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    siglip_vit.Attention.forward = _native_sdpa_forward


def _load():
    import torch
    from transformers import AutoModelForCausalLM

    sys.path.insert(0, str(VENDORED_SRC))

    repo_id = sys.argv[1]

    # Redirect stdout->stderr for the whole load - see module docstring.
    with contextlib.redirect_stdout(sys.stderr):
        from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor

        _patch_vision_attention_to_native_sdpa()

        processor = DeepseekVLV2Processor.from_pretrained(repo_id, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            repo_id, trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()

    return {"model": model, "processor": processor, "tokenizer": processor.tokenizer}


def _query(bundle, image, question, extra):
    import torch

    model = bundle["model"]
    processor = bundle["processor"]
    tokenizer = bundle["tokenizer"]

    conversation = [
        {"role": "<|User|>", "content": f"<image>\n{question}", "images": []},
        {"role": "<|Assistant|>", "content": ""},
    ]
    prepare_inputs = processor(
        conversations=conversation, images=[image], force_batchify=True, system_prompt="",
    ).to(model.device, dtype=torch.bfloat16)

    do_sample = extra.get("do_sample", False)
    gen_kwargs = dict(
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=extra.get("max_new_tokens", 512),
        do_sample=do_sample,
        use_cache=True,
    )
    if do_sample:
        gen_kwargs["temperature"] = extra.get("temperature", 0.1)
        gen_kwargs["top_p"] = extra.get("top_p", 0.9)
    if extra.get("repetition_penalty"):
        gen_kwargs["repetition_penalty"] = extra["repetition_penalty"]

    with torch.no_grad():
        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            input_ids=prepare_inputs.input_ids,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            attention_mask=prepare_inputs.attention_mask,
            **gen_kwargs,
        )

    answer = tokenizer.decode(
        outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=True,
    )
    return answer


if __name__ == "__main__":
    run_worker(_load, _query)
