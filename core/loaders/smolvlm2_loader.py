"""
SmolVLM2 loader — extraction stage candidate.

Uses AutoModelForImageTextToText + SmolVLMForConditionalGeneration
(resolved automatically via the Auto class from config.json's declared
architecture). Official pattern per HuggingFaceTB/SmolVLM2-2.2B-Instruct
model card:
  - AutoProcessor.from_pretrained()
  - apply_chat_template(tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt") - same single-call shape as
    Qwen3-VL, unlike the older Idefics3/SmolVLM-Base two-step pattern
    (apply_chat_template returning a string, then a separate processor()
    call) - do not confuse the two SmolVLM generations' loading patterns.
  - Inputs need an explicit dtype cast to match the model's loaded dtype,
    not just device - the card's example chains
    .to(model.device, dtype=torch.bfloat16) on the processor output,
    unlike Qwen3-VL which only needed .to(model.device).

UNCONFIRMED, watch on first real test: the model card's local-image
examples use {"type": "image", "path": "/path/to/file.jpg"} (a string)
in some SmolVLM2 documentation, not a raw PIL Image object. This loader
passes a PIL Image object directly under the "image" key instead,
matching Qwen3VLLoader's proven-working pattern, since both are modern
Auto-class VLMs from the same transformers generation and this is
expected to work the same way - but this hasn't been directly confirmed
for SmolVLM2 specifically. If image loading fails or produces garbage
output, this is the first thing to check: switch to writing the PIL
Image to a temp file and passing {"type": "image", "path": temp_path}
instead.

Config note: HuggingFaceTB's own generation_config.json for this
checkpoint carries no sampling defaults beyond token IDs - but the
card's own example code explicitly passes do_sample=False, which is
a real authored signal (unlike some other cards where greedy-vs-sampling
was never demonstrated either way). Starting from that.
"""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from core.loaders.base_loader import BaseLoader
from core.schema import (
    ExtractionResult, PersonalName, PlaceName, VisibleDate, DocumentCategory,
)
from core.extraction_parsing import parse_kv_block, parse_pipe_entries, parse_keyword_list, REQUIRED_KEYS


class SmolVLM2Loader(BaseLoader):
    def initialize_model_and_tokenizer(self) -> tuple[Any, Any, Any]:
        from transformers import AutoProcessor, AutoModelForImageTextToText

        processor = AutoProcessor.from_pretrained(self.config.repo_id, token=False)

        max_memory = self.config.build_max_memory_map()

        model_kwargs = dict(
            torch_dtype="auto",
            device_map="auto",
            token=False,
        )
        if self.config.attn_implementation:
            # Optional, not hardcoded - see GenerationConfig.attn_implementation
            # docstring in base_loader.py. The card's own example hardcodes
            # flash_attention_2, but that's not guaranteed installed here.
            model_kwargs["attn_implementation"] = self.config.attn_implementation
        if max_memory is not None:
            model_kwargs["max_memory"] = max_memory

        model = AutoModelForImageTextToText.from_pretrained(
            self.config.repo_id, **model_kwargs
        ).eval()

        if hasattr(model, "hf_device_map"):
            device_counts = {}
            for layer, device in model.hf_device_map.items():
                device_counts[str(device)] = device_counts.get(str(device), 0) + 1
            print(f"[SmolVLM2Loader] Device placement: {device_counts}")
            print(f"[SmolVLM2Loader] torch.cuda.is_available(): {torch.cuda.is_available()}")

        # Needed for the explicit input dtype-cast in _run_generate - the
        # card's example casts inputs to torch.bfloat16 specifically, but
        # hardcoding that would be wrong if torch_dtype="auto" resolved to
        # something else on this hardware. Track what actually loaded.
        self._model_dtype = next(model.parameters()).dtype

        self._execution_meta = {
            "device_map": model_kwargs.get("device_map"),
            "max_memory": max_memory,
            "vram_headroom_gb": self.config.vram_headroom_gb,
            "cpu_offload_limit_gb": self.config.cpu_offload_limit_gb,
        }
        self._oom_recovered = False

        self.model = model
        self.tokenizer = processor.tokenizer
        self.processor = processor
        return model, self.tokenizer, processor

    def _build_prompt(self, task: str) -> str:
        if task != "extract":
            raise NotImplementedError(
                "SmolVLM2Loader is not assigned a classification role. "
                "Classification is handled by GemmaLoader per pipeline.yaml."
            )
        return self.config.prompt_text

    def _run_generate(self, raw_image: Any, prompt: str) -> str:
        if not isinstance(raw_image, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(raw_image)}")

        if raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": raw_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        # Card's example casts to both device AND dtype here - pixel_values
        # from the processor default to float32, which mismatches a
        # bfloat16-loaded model unless cast explicitly. Using the model's
        # actual loaded dtype (tracked at init) rather than hardcoding
        # bfloat16, in case torch_dtype="auto" resolved differently.
        inputs = inputs.to(self.model.device, dtype=self._model_dtype)

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
        if self.config.stop_string:
            gen_kwargs["stop_strings"] = [self.config.stop_string]
            gen_kwargs["tokenizer"] = self.processor.tokenizer
        # presence_penalty deliberately omitted here unless/until confirmed
        # supported for this architecture - Qwen3-VL's card advertised it
        # and it still turned out unsupported by generate() on that model;
        # no reason to assume it's supported here without evidence either.

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        # Trim to only the new tokens (exclude prompt) - same pattern as
        # every other loader in this codebase. An earlier version of this
        # method decoded the full sequence and split on the literal
        # "Assistant:" marker from the chat template instead; switched to
        # trim-by-length because it's simpler, doesn't depend on a text
        # marker surviving decode intact, and is completely indifferent
        # to what's actually in the generated content (a string-split
        # approach would break if an extracted field ever happened to
        # contain the word "Assistant", however unlikely).
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        decoded = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True
        )
        result_text = decoded[0].strip() if decoded else ""

        if self.config.stop_string and result_text.endswith(self.config.stop_string):
            result_text = result_text[: -len(self.config.stop_string)].rstrip()

        return result_text

    def _parse_classification(self, file_path: str, raw_output: str):
        raise NotImplementedError(
            "SmolVLM2Loader is not assigned a classification role. "
            "Classification is handled by GemmaLoader per pipeline.yaml."
        )

    def _parse_extraction(self, file_path: str, category: str, raw_output: str) -> ExtractionResult:
        fields = parse_kv_block(raw_output)

        missing = REQUIRED_KEYS - fields.keys()
        if missing:
            raise ValueError(
                f"Extraction output missing required fields {missing} for "
                f"{file_path}. Raw output: {raw_output[:200]!r}."
            )

        personal_names = [
            PersonalName(value=v[:120], confidence=c)
            for v, c in parse_pipe_entries(fields.get("personal_names", ""))
        ]
        place_names = [
            PlaceName(value=v[:160], confidence=c)
            for v, c in parse_pipe_entries(fields.get("place_names", ""))
        ]
        visible_dates = [
            VisibleDate(value=v[:60], confidence=c)
            for v, c in parse_pipe_entries(fields.get("visible_dates", ""))
        ]
        keywords_raw = fields.get("subject_keywords", "")
        subject_keywords = parse_keyword_list(keywords_raw)

        return ExtractionResult(
            file_path=file_path,
            category=DocumentCategory(category),
            document_type=fields.get("document_type", "")[:80] or None,
            personal_names=personal_names,
            place_names=place_names,
            visible_dates=visible_dates,
            subject_keywords=subject_keywords,
            raw_model_output_len=len(raw_output),
            model=self.config.model_name,
            prompt_version=self.config.prompt_version,
            generation_config_hash=self.config.content_hash(),
            device_map=self._execution_meta.get("device_map"),
            max_memory=self._execution_meta.get("max_memory"),
            vram_headroom_gb=self._execution_meta.get("vram_headroom_gb"),
            cpu_offload_limit_gb=self._execution_meta.get("cpu_offload_limit_gb"),
            oom_recovered=getattr(self, "_oom_recovered", False),
        )
