"""
MiniCPM-V-4.5 GPTQ loader - chat-only, same scope as text_llm_loader.py
(model_console send_turn()/agent paths only, no classify()/extract()
pipeline role). Built 2026-08-13 for the multi-runtime integration
(plan addendum, Phase 4): `openbmb/MiniCPM-V-4_5-GPTQ` is genuine GPTQ
(quant_method "gptq"), which runs on the NORMAL transformers path via
GPTQModel's Marlin kernels - unlike the w4a16-ct/AWQ checkpoints that
need core/vllm_runtime.py's subprocess. So this is a plain BaseLoader
subclass managed by core.model_residency like every other loader, no
new plumbing.

STATUS (2026-08-13): CURRENTLY UNUSED - kept as the fallback path, not
dead code to delete. Installing its dependency stack into venv_backend
failed in practice: gptqmodel 7.3.2 requires transformers>=5.15
(imports create_recurrent_attention_mask) while venv_backend is
deliberately pinned to 5.12.1 to match the Windows production stack,
and the install also downgraded numpy - rolled back entirely.
config/models/minicpm_v_gptq.yaml now runs this checkpoint under
`runtime: vllm` instead (confirmed working, tight KV fit - see the
YAML header). Reactivate this loader only after deciding the venv
question deliberately (upgrade venv_backend's transformers vs. pin an
older gptqmodel vs. a separate serving venv).

Environment requirements if reactivated (proven in ~/venv_test):
`gptqmodel` + `optimum` installed, CUDA Toolkit present (CUDA_HOME set)
for the Marlin kernel JIT - see docs/WSL_COMPUTE_BACKEND_BASELINE.md.
NOT loadable on the Windows side (gptqmodel has no Windows build), so
only reachable via backend="remote" regardless.

The checkpoint's HF-cache config.json carries the required
`"block_name_to_quantize": "llm.model.layers"` fix (applied directly to
the cached blob on 2026-08-13 - GPTQModel's auto-detection fails on
MiniCPM-V's custom `self.llm = Qwen3ForCausalLM(...)` nesting, and
passing the fix via GPTQConfig kwargs is silently ignored when the
checkpoint embeds its own quantization_config). If the cache entry for
this repo is ever cleared, that fix must be reapplied before this
loader works - see the baseline doc's MiniCPM section.

Interface note: MiniCPM-V ships its own `model.chat(msgs=..., tokenizer=
..., ...)` remote-code API (NOT processor/apply_chat_template like
every other VLM here) - an image-bearing user turn puts the PIL image
directly in the content list ([image, text]), per the official model
card usage. sampling=False maps to greedy decoding, matching this
project's default do_sample: false configs.
"""

from __future__ import annotations

from typing import Any

from core.loaders.base_loader import BaseLoader
from core.schema import ClassificationResult, ExtractionResult


class MinicpmVLoader(BaseLoader):
    def initialize_model_and_tokenizer(self) -> tuple[Any, Any, Any]:
        from transformers import AutoModel, AutoTokenizer

        # dtype="auto" + device_map="auto": the exact load call confirmed
        # live in the 2026-08-13 venv_test bring-up (~8.3GB VRAM, Marlin
        # kernel selected + JIT-compiled). trust_remote_code is required
        # - the whole chat interface lives in the checkpoint's own code.
        model = AutoModel.from_pretrained(
            self.config.repo_id,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto",
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(self.config.repo_id, trust_remote_code=True)

        self.model = model
        self.tokenizer = tokenizer
        self.processor = None  # MiniCPM-V's chat() does its own image preprocessing
        return model, tokenizer, None

    def _build_prompt(self, task: str) -> str:
        # Never called - chat-only loader, same stance as TextLLMLoader.
        raise NotImplementedError(
            "MinicpmVLoader has no classify/extract role - it's a chat-only "
            "VLM candidate (see module docstring)."
        )

    def _chat(self, msgs: list[dict[str, Any]]) -> str:
        kwargs: dict[str, Any] = dict(
            image=None,  # images ride inside msgs content lists (official usage)
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=self.config.do_sample,
            max_new_tokens=self.config.max_new_tokens,
        )
        if self.config.do_sample:
            kwargs.update(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )
        system_prompt = self.config.extra.get("system_prompt")
        if system_prompt:
            kwargs["system_prompt"] = system_prompt
        answer = self.model.chat(**kwargs)
        return answer if isinstance(answer, str) else str(answer)

    def _run_generate(self, raw_image: Any, prompt: str) -> str:
        if raw_image is not None:
            content: Any = [raw_image, prompt]
        else:
            content = prompt
        return self._chat([{"role": "user", "content": content}])

    def _run_generate_with_history(
        self, history: list[dict[str, Any]], raw_image: Any, prompt: str,
    ) -> str:
        # history arrives as base_loader.py's documented text-only
        # block-list shape - flatten to the plain strings MiniCPM's own
        # chat() expects (same flattening TextLLMLoader does, for the
        # same reason: only the CURRENT turn may carry an image).
        msgs: list[dict[str, Any]] = [
            {"role": m["role"], "content": "".join(b.get("text", "") for b in m["content"])}
            for m in history
        ]
        if raw_image is not None:
            msgs.append({"role": "user", "content": [raw_image, prompt]})
        else:
            msgs.append({"role": "user", "content": prompt})
        return self._chat(msgs)

    def _parse_classification(self, file_path: str, raw_output: str) -> ClassificationResult:
        raise NotImplementedError(
            "MinicpmVLoader has no classify role - chat-only (see module docstring)."
        )

    def _parse_extraction(self, file_path: str, category: str, raw_output: str) -> ExtractionResult:
        raise NotImplementedError(
            "MinicpmVLoader has no extract role - chat-only (see module docstring)."
        )
