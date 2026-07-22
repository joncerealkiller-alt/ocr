"""
Pixtral-12B loader — stage-1 extraction candidate (2026-07-14, per Jon's
voice-relayed suggestion, independently verified before building).

VERIFICATION NOTES (2026-07-14): the original suggestion specified "Q4
GGUF" at ~9-10GB - the VRAM figure checked out via a real HF discussion
report (~10GB in 4-bit), but GGUF/llama.cpp is the wrong ecosystem for
this project (nothing here uses llama.cpp - every loader is standard
transformers). The actual confirmed-working path uses bitsandbytes
4-bit quantization instead, same approach already proven throughout
this project, not GGUF:
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        "mistral-community/pixtral-12b", quantization_config=quantization_config
    )
No trust_remote_code needed - LlavaForConditionalGeneration is a
standard, built-in transformers class.

repo_id is mistral-community/pixtral-12b specifically, NOT the official
mistralai/Pixtral-12B-2409 - the official release needs vLLM/its own
serving stack (confirmed via a real GitHub issue: "we don't have any
transformers implementation of pixtral" for the official weights). The
mistral-community conversion is what actually loads through standard
transformers, which is what this project uses everywhere else.

UNCONFIRMED, watch on first real test: message/chat-template format
assumed to follow the same modern content-list pattern already proven
for GraniteVisionLoader (content: [{"type":"image","image":...},
{"type":"text","text":...}]) via apply_chat_template - Pixtral is
LLaVA-architecture-based like Granite Vision, so this is a reasonable
default, not independently confirmed for Pixtral's specific processor.
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


class PixtralLoader(BaseLoader):
    def initialize_model_and_tokenizer(self) -> tuple[Any, Any, Any]:
        from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

        processor = AutoProcessor.from_pretrained(self.config.repo_id, token=False)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        max_memory = self.config.build_max_memory_map()

        model_kwargs = dict(
            device_map="auto",
            quantization_config=quantization_config,
            token=False,
        )
        if self.config.attn_implementation:
            model_kwargs["attn_implementation"] = self.config.attn_implementation
        if max_memory is not None:
            model_kwargs["max_memory"] = max_memory

        model = LlavaForConditionalGeneration.from_pretrained(
            self.config.repo_id, **model_kwargs
        ).eval()

        if hasattr(model, "hf_device_map"):
            device_counts = {}
            for layer, device in model.hf_device_map.items():
                device_counts[str(device)] = device_counts.get(str(device), 0) + 1
            print(f"[PixtralLoader] Device placement: {device_counts}")
            print(f"[PixtralLoader] torch.cuda.is_available(): {torch.cuda.is_available()}")

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
                "PixtralLoader is not assigned a classification role. "
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
        inputs = inputs.to(self.model.device)

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

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

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
            "PixtralLoader is not assigned a classification role. "
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
