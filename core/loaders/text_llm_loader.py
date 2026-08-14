"""
Pure text causal-LM loader - no vision tower, no processor, no image
handling at all. Built 2026-08-13 for the "research_text" role
(core/loaders/base_loader.py's GenerationConfig.role field already
anticipated this - see its own comment referencing an earlier E4B_QAT
candidate that was ruled out; this is the loader for the model that
replaced it).

Why this exists rather than reusing an existing VLM loader in
text-only mode (every VLM loader here supports text_only_supported):
a VLM's vision tower/projector weights are pure dead weight for a
text-only task and still cost real VRAM even when never invoked. Jon's
own framing: "as this is a research agent only, we could look at a
quantized 8B LLM not VLM, so we're not carrying the extra bloat the
vision tower adds." Confirmed concretely by the InternVL3-8B live test
this same session: correct, faithful digit reproduction (no
fabrication, first try) but "pushing the limits" of a 16GB card and
noticeably slower - a VLM's full weight, spent on a call that never
sees an image.

Deliberately routed through core/model_residency.py's borrow()
mechanism (see core/agent_tools/web_research_agent.py's use of it),
never held resident independently - this project has ONE authoritative
GPU residency owner per process, not two (see model_residency.py's own
module docstring on the OOM risk that discipline prevents).
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from core.loaders.base_loader import BaseLoader, GenerationConfig
from core.schema import ClassificationResult, ExtractionResult


class TextLLMLoader(BaseLoader):
    def initialize_model_and_tokenizer(self) -> tuple[Any, Any, Any]:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.config.repo_id)

        model_kwargs: dict[str, Any] = {}
        max_memory = self.config.build_max_memory_map()
        if max_memory is not None:
            model_kwargs["max_memory"] = max_memory

        # Default True (unlike gemma_loader.py's load_in_8bit, opt-in
        # default False) - the entire reason this loader exists is to
        # run an 8B-class text model in the same VRAM budget a VLM's
        # quantized weights already prove workable at, so an
        # unquantized load defeats the purpose. Still config-driven
        # (extra.load_in_4bit), not hardcoded, for the same reason
        # every other loader keeps this in config: a future smaller
        # research-text model might not need it.
        load_in_4bit = self.config.extra.get("load_in_4bit", True)
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            model_kwargs["dtype"] = "auto"

        if self.config.attn_implementation:
            model_kwargs["attn_implementation"] = self.config.attn_implementation

        model = AutoModelForCausalLM.from_pretrained(
            self.config.repo_id, device_map="auto", **model_kwargs,
        )
        model.eval()
        # core/model_residency.py::acquire() calls
        # initialize_model_and_tokenizer() and discards its return value
        # - every loader is expected to set self.model/self.tokenizer/
        # self.processor itself as a side effect (confirmed against
        # qwen_loader.py's own implementation), the return tuple is only
        # for callers that construct+init a loader standalone.
        self.model = model
        self.tokenizer = tokenizer
        self.processor = None
        return model, tokenizer, None

    def _build_prompt(self, task: str) -> str:
        # Never called - this loader is only used via model_console's
        # send_turn()/send_turn_with_history() path (agent chat/
        # research calls), never via classify()/extract(), which are
        # the only callers of _build_prompt(). Raises loudly rather
        # than silently returning something meaningless, matching this
        # project's "loud error over silent wrong behavior" discipline.
        raise NotImplementedError(
            "TextLLMLoader has no classify/extract role - it's a chat-only "
            "text LLM (see module docstring)."
        )

    def _generate_tail(self, messages: list[dict[str, Any]]) -> str:
        gen_kwargs: dict[str, Any] = dict(
            do_sample=self.config.do_sample,
            temperature=self.config.temperature if self.config.do_sample else None,
            top_p=self.config.top_p if self.config.do_sample else None,
            top_k=self.config.top_k if self.config.do_sample else None,
            repetition_penalty=self.config.repetition_penalty,
            no_repeat_ngram_size=self.config.no_repeat_ngram_size,
            max_new_tokens=self.config.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        self._maybe_add_charset_logits_processor(gen_kwargs)

        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(chat_text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )

    def _run_generate(self, raw_image: Any, prompt: str) -> str:
        if raw_image is not None:
            raise ValueError(
                "TextLLMLoader is text-only - it must never be called with an "
                "image (see module docstring: no vision tower, no processor)."
            )
        return self._generate_tail([{"role": "user", "content": prompt}])

    def _run_generate_with_history(
        self, history: list[dict[str, Any]], raw_image: Any, prompt: str,
    ) -> str:
        if raw_image is not None:
            raise ValueError(
                "TextLLMLoader is text-only - it must never be called with an "
                "image (see module docstring: no vision tower, no processor)."
            )
        # history entries carry content as a list of {"type": "text",
        # "text": "..."} blocks (base_loader.py's own documented shape,
        # designed to be vision-loader-content-shape-compatible) - a
        # pure text LLM's chat template expects a plain string, so
        # flatten here rather than assume the template tolerates the
        # block-list shape (most don't).
        flat_messages = [
            {"role": m["role"], "content": "".join(b.get("text", "") for b in m["content"])}
            for m in history
        ]
        flat_messages.append({"role": "user", "content": prompt})
        return self._generate_tail(flat_messages)

    def _parse_classification(self, file_path: str, raw_output: str) -> ClassificationResult:
        # Never called - see _build_prompt()'s docstring, same reasoning.
        raise NotImplementedError(
            "TextLLMLoader has no classify role - it's a chat-only text LLM."
        )

    def _parse_extraction(self, file_path: str, category: str, raw_output: str) -> ExtractionResult:
        raise NotImplementedError(
            "TextLLMLoader has no extract role - it's a chat-only text LLM."
        )
