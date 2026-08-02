"""
Loader for Meta Chameleon-7B (facebook/chameleon-7b) - encoder-free,
early-fusion architecture: images are tokenized directly into the same
discrete vocabulary as text (VQ tokenizer), not routed through a
separate pretrained vision encoder.

Subclasses GemmaLoader ONLY to inherit its already-generic classifier
key:value parsing (_parse_kv_block/_to_bool/_to_float/_parse_
classification/_build_prompt) - same pattern as Gemma4UnifiedLoader.
Everything about HOW the model is loaded and prompted is completely
different and is overridden here:
  - No chat template exists for this repo (confirmed: no chat_template
    key in tokenizer_config.json, no chat_template.jinja file, and
    transformers' own ChameleonForConditionalGeneration.forward()
    docstring example uses a flat text string with an inline <image>
    placeholder token, not a role-based messages list).
  - ChameleonProcessor takes images=[...] and text="...<image>..." as a
    single flat string - one call, no apply_chat_template step.
"""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from core.loaders.gemma_loader import GemmaLoader


class ChameleonLoader(GemmaLoader):
    def initialize_model_and_tokenizer(self) -> tuple[Any, Any, Any]:
        from transformers import ChameleonProcessor, ChameleonForConditionalGeneration

        # NOTE: no token=False here (unlike GemmaLoader/Gemma4UnifiedLoader) -
        # this repo is gated, so it needs real authentication (whatever
        # token huggingface_hub already has cached) rather than a forced
        # anonymous request, which would fail even with legitimate access.
        processor = ChameleonProcessor.from_pretrained(self.config.repo_id)
        model = ChameleonForConditionalGeneration.from_pretrained(
            self.config.repo_id,
            dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

        self._execution_meta = {
            "device_map": "auto",
            "max_memory": self.config.build_max_memory_map(),
            "vram_headroom_gb": self.config.vram_headroom_gb,
            "cpu_offload_limit_gb": self.config.cpu_offload_limit_gb,
        }
        self._oom_recovered = False

        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        return model, self.tokenizer, processor

    def _run_generate(self, raw_image: Image.Image, prompt: str) -> str:
        if raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")

        # No system role / chat structure - flat text with an inline
        # <image> placeholder, matching transformers' own documented
        # usage example for this model class. Image placeholder first,
        # same "image before text" convention used elsewhere in this
        # project (GemmaLoader's own comment on modality order).
        full_prompt = "<image>\n" + prompt

        inputs = self.processor(
            images=[raw_image], text=full_prompt, return_tensors="pt"
        ).to(self.model.device, torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs = dict(
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.do_sample,
        )
        if self.config.do_sample:
            gen_kwargs.update(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )
        if self.config.repetition_penalty and self.config.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = self.config.repetition_penalty
        if self.config.no_repeat_ngram_size:
            gen_kwargs["no_repeat_ngram_size"] = self.config.no_repeat_ngram_size
        self._maybe_add_charset_logits_processor(gen_kwargs)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        return self.processor.batch_decode(
            outputs[:, input_len:], skip_special_tokens=True
        )[0].strip()
