"""
Granite Vision 3.2-2b loader — extraction stage candidate.

Standard architecture: LlavaNextForConditionalGeneration (confirmed via
config.json's architectures field), resolved automatically through
AutoModelForVision2Seq per the official model card. No trust_remote_code,
no custom modeling files - unlike granite-vision-4.1-4b, which was
deliberately skipped for this reason (see project history 2026-07-11).

Official pattern (ibm-granite/granite-vision-3.2-2b model card):
  - AutoProcessor.from_pretrained() / AutoModelForVision2Seq.from_pretrained()
  - apply_chat_template(tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt") - single-call shape, same as
    Qwen3-VL and SmolVLM2.

CORRECTION (2026-07-11, first real test): the card's own example uses
AutoModelForVision2Seq, but that class has been deprecated/removed in
newer transformers releases (confirmed via a real ImportError on the
installed 4.57 dev build) in favor of the unified
AutoModelForImageTextToText - the same class Qwen3VLLoader and
SmolVLM2Loader already use. Switched to that instead; config.json's
declared architecture (LlavaNextForConditionalGeneration) resolves
correctly through either Auto class, so this is a clean fix, not a
workaround.

Deliberate deviations from the card's literal example (consistent with
every other loader in this project, not oversights):
  - device_map="auto" + max_memory, not the card's simple .to(device) -
    keeps OOM-recovery behavior uniform across all loaders.
  - Trim-by-token-length before decoding, not the card's full-sequence
    single decode() call - more robust, matches every other loader.
  - torch_dtype="auto", not the card's unspecified (implicit float32)
    dtype.

Resolution: LLaVA-NeXT "AnyRes" dynamic tiling (image_grid_pinpoints in
config.json) supports up to 3840px on the long edge - meaningfully
higher ceiling than SmolVLM2's ~1536px, good for dense document text.

No author-provided sampling defaults exist in generation_config.json
(token IDs only, "_from_model_config": true) - defaulting to greedy,
consistent with how every other under-specified model was handled
tonight.

UNCONFIRMED, watch on first real test: PIL Image object passed directly
under the "image" content key, matching Qwen3VLLoader's proven pattern.
Mildly de-risked (not proven) by the card's own example passing a local
file path under an "image"/"url"-style key rather than a real remote
URL - suggests this processor's image-loading utility is flexible about
input type, but not confirmed identical for a raw PIL object specifically.
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


class GraniteVisionLoader(BaseLoader):
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
            print(f"[GraniteVisionLoader] Device placement: {device_counts}")
            print(f"[GraniteVisionLoader] torch.cuda.is_available(): {torch.cuda.is_available()}")

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
                "GraniteVisionLoader is not assigned a classification role. "
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
        # presence_penalty deliberately omitted - unconfirmed for this
        # architecture, same reasoning as SmolVLM2Loader.

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
            "GraniteVisionLoader is not assigned a classification role. "
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
