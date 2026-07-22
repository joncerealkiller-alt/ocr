"""
InternVL3 loader — extraction candidate, both 2B and 8B share this one
loader class (identical architecture/calling convention, only repo_id
and resource footprint differ between sizes).

Confirmed native transformers support (OpenGVLab/InternVL3-{2B,8B}-hf
repos specifically - NOT the newer InternVL3.5 series, which requires
trust_remote_code=True and is a different generation from what was
requested 2026-07-12). AutoModelForImageTextToText resolves correctly,
no trust_remote_code needed for InternVL3 proper.

Calling convention matches Qwen3-VL/SmolVLM2/Granite/GLM-OCR's single-
call pattern exactly: apply_chat_template(tokenize=True, return_dict=
True, return_tensors="pt", add_generation_prompt=True), confirmed
directly from the official model card examples for both sizes.

No author-provided sampling defaults found in generation_config.json
research for this model - same situation as SmolVLM2/Granite/olmOCR.
Defaulting to greedy, consistent with how every other under-specified
model was handled this session.
"""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from core.loaders.base_loader import BaseLoader
from core.schema import (
    ExtractionResult, PersonalName, PlaceName, VisibleDate, DocumentCategory,
)
from core.extraction_parsing import (
    parse_kv_block, parse_pipe_entries, parse_keyword_list, REQUIRED_KEYS,
)


class InternVLLoader(BaseLoader):
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
            print(f"[InternVLLoader] Device placement: {device_counts}")
            print(f"[InternVLLoader] torch.cuda.is_available(): {torch.cuda.is_available()}")

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
                "InternVLLoader is not assigned a classification role. "
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
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

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
        if self.config.num_beams and self.config.num_beams > 1:
            gen_kwargs["num_beams"] = self.config.num_beams
            gen_kwargs["do_sample"] = False
            if self.config.length_penalty != 1.0:
                gen_kwargs["length_penalty"] = self.config.length_penalty
            if self.config.early_stopping:
                gen_kwargs["early_stopping"] = self.config.early_stopping
        if self.config.stop_string:
            gen_kwargs["stop_strings"] = [self.config.stop_string]
            gen_kwargs["tokenizer"] = self.processor.tokenizer

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        result_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True
        )[0].strip()

        if self.config.stop_string and result_text.endswith(self.config.stop_string):
            result_text = result_text[: -len(self.config.stop_string)].rstrip()

        return result_text

    def _parse_classification(self, file_path: str, raw_output: str):
        raise NotImplementedError(
            "InternVLLoader is not assigned a classification role. "
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
