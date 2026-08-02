"""
Nanonets-OCR2-3B loader - extraction-stage CANDIDATE, prep/Candidate-
Qualification only (2026-07-28).

Thin subclass of QwenLoader (core/loaders/qwen_loader.py), NOT a
standalone loader - QwenLoader is left completely untouched. This is a
fine-tune of Qwen/Qwen2.5-VL-3B-Instruct (same architecture,
Qwen2_5_VLForConditionalGeneration, no custom code) and would otherwise
just need a new YAML profile reusing QwenLoader directly - but this
specific checkpoint has a real, confirmed config export bug that
QwenLoader's plain from_pretrained() call doesn't work around, so a
one-method override is needed.

THE BUG (confirmed 2026-07-28, not guessed): the official base model's
config.json sets "tie_word_embeddings": true at the TOP LEVEL. Nanonets'
fine-tuned re-export restructured the config into a nested "text_config"
sub-object and only set tie_word_embeddings there - this project's
installed transformers (5.12.1) doesn't look in that nested location for
Qwen2_5_VLForConditionalGeneration, so it defaults to untied. The
checkpoint's own safetensors files only contain model.embed_tokens.weight
(no separate lm_head.weight - correct and expected for a genuinely tied-
embedding model), so loading via the plain path prints a "lm_head.weight
MISSING... newly initialized" warning and silently gives the model a
RANDOM, untrained output layer - confirmed via direct test: raw output
was complete incoherent gibberish ("FWochen sub literalDismissasonry
mistakes COVID..."), not just degraded quality.

THE FIX: load the config explicitly first, force tie_word_embeddings=True
at the top level (matching what the checkpoint's own genuinely-trained
weights expect), then pass that corrected config into from_pretrained().
Confirmed via direct test: with this fix, the LOAD REPORT warning
disappears entirely and the model produces coherent, accurate, well-
grounded output (correctly read a real historical passenger manifest's
ship name, date, port, and column structure as a faithful markdown/HTML
table - the strongest raw output quality seen across every candidate
checked this session).

License note: not resolved here - see config/models/nanonets_ocr2_3b.yaml.
"""

from __future__ import annotations

from typing import Any

from core.loaders.qwen_loader import QwenLoader


class NanonetsOcr2Loader(QwenLoader):
    def initialize_model_and_tokenizer(self) -> tuple[Any, Any, Any]:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoConfig

        processor_kwargs = dict(token=False)
        if self.config.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.config.min_pixels
        if self.config.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.config.max_pixels

        processor = AutoProcessor.from_pretrained(
            self.config.repo_id, **processor_kwargs
        )

        # The actual fix - see module docstring. Everything else below
        # mirrors QwenLoader.initialize_model_and_tokenizer() exactly.
        model_config = AutoConfig.from_pretrained(self.config.repo_id, token=False)
        model_config.tie_word_embeddings = True

        max_memory = self.config.build_max_memory_map()

        model_kwargs = dict(
            config=model_config,
            torch_dtype="auto",
            device_map="auto",
            token=False,
        )
        if max_memory is not None:
            model_kwargs["max_memory"] = max_memory

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.repo_id, **model_kwargs
        ).eval()

        if hasattr(model, "hf_device_map"):
            device_counts = {}
            for layer, device in model.hf_device_map.items():
                device_counts[str(device)] = device_counts.get(str(device), 0) + 1
            print(f"[NanonetsOcr2Loader] Device placement: {device_counts}")

        self._execution_meta = {
            "device_map": model_kwargs.get("device_map"),
            "max_memory": max_memory,
            "vram_headroom_gb": self.config.vram_headroom_gb,
            "cpu_offload_limit_gb": self.config.cpu_offload_limit_gb,
        }
        self._oom_recovered = False

        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        return model, self.tokenizer, processor

    # _build_prompt, _run_generate, _parse_classification, _parse_extraction
    # all inherited unchanged from QwenLoader - only weight loading differs.
