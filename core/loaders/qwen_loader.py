"""
Qwen2.5-VL loader — extraction stage.

Rewritten to match the official model card pattern exactly:
  - Qwen2_5_VLForConditionalGeneration (specific class), not the
    generic AutoModelForImageTextToText previously used
  - qwen_vl_utils.process_vision_info() handles image loading/prep,
    replacing the earlier manual <|im_start|>/<|vis_start|> string
    construction, which was a guess at the format rather than the
    documented one
  - apply_chat_template(tokenize=False) followed by a separate
    processor(text=[text], images=image_inputs, ...) call, matching
    the two-step flow in the official usage example

Requires the qwen-vl-utils package: pip install qwen-vl-utils

Output parsing (pipe-delimited value|confidence format) and schema
validation are unchanged from the earlier version - only the model
loading and generation mechanics were corrected to match documented
behavior instead of an inferred one.
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


class QwenLoader(BaseLoader):
    def initialize_model_and_tokenizer(self) -> tuple[Any, Any, Any]:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        processor_kwargs = dict(token=False)
        if self.config.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.config.min_pixels
        if self.config.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.config.max_pixels

        processor = AutoProcessor.from_pretrained(
            self.config.repo_id, **processor_kwargs
        )

        max_memory = self.config.build_max_memory_map()

        model_kwargs = dict(
            torch_dtype="auto",
            device_map="auto",
            token=False,
        )
        if max_memory is not None:
            # Reserves vram_headroom_gb for context-window growth during
            # long extraction runs, and gives device_map a CPU fallback
            # target so a single oversized image doesn't crash the whole
            # batch with an OOM - it offloads instead, slower but survives.
            model_kwargs["max_memory"] = max_memory

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.repo_id, **model_kwargs
        ).eval()

        # Diagnostic: confirm actual device placement. If this shows mostly
        # or entirely "cpu", device_map="auto" decided to offload despite
        # GPU being available - check CUDA visibility and max_memory
        # values printed below before assuming it's a real capacity issue.
        if hasattr(model, "hf_device_map"):
            device_counts = {}
            for layer, device in model.hf_device_map.items():
                device_counts[str(device)] = device_counts.get(str(device), 0) + 1
            print(f"[QwenLoader] Device placement: {device_counts}")
            print(f"[QwenLoader] max_memory passed to from_pretrained: {max_memory}")
            import torch as _torch
            print(f"[QwenLoader] torch.cuda.is_available(): {_torch.cuda.is_available()}")

        # Recorded here so every ExtractionResult from this loader instance
        # can report the execution mode it actually ran under, per the
        # traceability requirement - a slow/degraded run should be
        # explainable after the fact, not just "this one took forever."
        self._execution_meta = {
            "device_map": model_kwargs.get("device_map"),
            "max_memory": max_memory,
            "vram_headroom_gb": self.config.vram_headroom_gb,
            "cpu_offload_limit_gb": self.config.cpu_offload_limit_gb,
        }
        self._oom_recovered = False

        self.model = model
        self.processor = processor
        return model, None, processor

    def _build_prompt(self, task: str) -> str:
        if task != "extract":
            raise NotImplementedError(
                "QwenLoader is currently scoped to extraction only."
            )
        if not self.config.prompt_text:
            raise ValueError(
                "config.prompt_text is empty — load it from the bucket's "
                "prompt_file before calling extract()."
            )
        return self.config.prompt_text

    def _run_generate(self, raw_image: Image.Image, prompt: str) -> str:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as e:
            raise ImportError(
                "qwen_vl_utils is required for QwenLoader. Install with: "
                "pip install qwen-vl-utils"
            ) from e

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

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
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

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        # Trim the input prompt tokens off the front, per official example.
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0].strip()

    def _parse_classification(self, file_path: str, raw_output: str):
        raise NotImplementedError("QwenLoader is not assigned a classification role.")

    # -- parsing -------------------------------------------------------
    # Uses core/extraction_parsing.py's shared implementation, not a
    # local copy. This loader previously had its own older duplicate of
    # _parse_kv_block/_parse_pipe_entries, predating the duplicate-block
    # degeneration guard added during Qwen3-VL debugging (2026-07-10) -
    # meaning it had the same silent-data-loss vulnerability (a
    # degenerate repeated block could overwrite good data and still
    # validate cleanly) that was found and fixed elsewhere, just never
    # backported here since this loader wasn't part of active testing
    # after the session's first few minutes. Migrating to the shared
    # module closes that gap, plus the subject_keywords comma/semicolon
    # delimiter bug found via Granite testing (2026-07-11) - both fixed
    # once, here, rather than needing a fourth separate patch.

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
            oom_recovered=self._oom_recovered,
        )
