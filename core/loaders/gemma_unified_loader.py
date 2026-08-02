"""
Loader for the Gemma 4 12B Unified architecture (encoder-free: raw
image patches are projected directly into the LLM's embedding space,
no separate vision encoder - see the model card's "Unified" section).

Subclasses GemmaLoader and overrides ONLY initialize_model_and_tokenizer
- everything else (prompt building, chat template, reasoning toggle,
image token budget, generation, response parsing) is identical to the
E2B loader and confirmed by the model card to work the same way for
this architecture too. The one real difference: this model class is
Gemma4UnifiedForConditionalGeneration, loaded via AutoModelForMultimodalLM
per the model's own README, not AutoModelForCausalLM.
"""

from __future__ import annotations

from typing import Any

from core.loaders.gemma_loader import GemmaLoader


class Gemma4UnifiedLoader(GemmaLoader):
    def initialize_model_and_tokenizer(self) -> tuple[Any, Any, Any]:
        from transformers import AutoProcessor, AutoModelForMultimodalLM

        processor = AutoProcessor.from_pretrained(
            self.config.repo_id,
            token=False,
        )
        model = AutoModelForMultimodalLM.from_pretrained(
            self.config.repo_id,
            dtype="auto",
            device_map="auto",
            token=False,
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
