"""
LFM2-VL-1.6B loader - extraction-stage CANDIDATE, prep/sanity-test only
(2026-07-28). NOT wired into config/pipeline.yaml - registering this
loader only makes it available to the same testing tools every other
model profile already uses (model_assessment.py, ad-hoc diagnostics
scripts), it does not activate it for any real bucket.

Genuinely native transformers architecture (Lfm2VlForConditionalGeneration,
model_type "lfm2_vl") - no trust_remote_code, no custom .py files, no
auto_map. Confirmed to load in-process against this project's main venv
(transformers 5.12.1 already satisfies the model's declared >=4.57
requirement) - the ONLY candidate this round that didn't need a separate
venv at all. Same apply_chat_template -> generate -> batch_decode calling
convention as PixtralLoader.

Smoke test (2026-07-28, 30807_A000676-00099(1)_dewarped.jpg, printed
manifest): load ~12s, ~3GB VRAM for weights (2.4B params bf16), ~6.3GB
peak during generation. Needed the same repetition_penalty=1.15/
no_repeat_ngram_size=5 defaults as every other model in this project's
roster (the card's own bare example passes neither).

Output quality: got the real document type ("Passenger List") and the
real sailing date (5 December 1920, corroborated independently by
HunyuanOCR's own read of the same image) right, and transcribed real
column headers off the manifest - genuinely grounded, not a blind guess.
But then confidently invented a long tail of fields that do not exist on
this document at all (Ticket Price, Boarding Time, Confidentiality
Notice, Document Control #, etc.) instead of stopping - fabrication, not
an "unclear" hedge. Also never populated personal_names:/place_names: at
all, so schema parsing fails on this prompt as-is. Needs more testing
before trusting it - worth trying a stricter stop condition (see
qwen3vl2b.yaml's stop_string approach, which fixed a similar unbounded-
field problem) and/or a lower max_new_tokens ceiling to catch the
fabrication before it runs on.
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


class Lfm2VlLoader(BaseLoader):
    def initialize_model_and_tokenizer(self) -> tuple[Any, Any, Any]:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(self.config.repo_id, token=False)

        max_memory = self.config.build_max_memory_map()

        model_kwargs = dict(
            device_map="auto",
            dtype=torch.bfloat16,
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
            print(f"[Lfm2VlLoader] Device placement: {device_counts}")

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
                "Lfm2VlLoader is not assigned a classification role."
            )
        return self.config.prompt_text

    @staticmethod
    def _build_user_message(raw_image: Any, prompt: str) -> dict:
        user_content = []
        if raw_image is not None:
            user_content.append({"type": "image", "image": raw_image})
        user_content.append({"type": "text", "text": prompt})
        return {"role": "user", "content": user_content}

    def _run_generate(self, raw_image: Any, prompt: str) -> str:
        """
        raw_image=None path added 2026-08-10 for model_console (see
        base_loader.py's GenerationConfig.text_only_supported) - the
        model card states plainly that images aren't technically
        required, LFM2-VL can process text-only inputs. Confirmed
        2026-08-10 via real generation calls - see
        config/models/lfm2_vl_1_6b.yaml's text_only_supported flag.
        """
        if raw_image is not None and not isinstance(raw_image, Image.Image):
            raise TypeError(f"Expected PIL Image or None, got {type(raw_image)}")
        if raw_image is not None and raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")
        conversation = [self._build_user_message(raw_image, prompt)]
        return self._generate_from_messages(conversation)

    def _run_generate_with_history(self, history: list[dict], raw_image: Any, prompt: str) -> str:
        """
        2026-08-11, model_console conversation context (see base_loader.py's
        _run_generate_with_history docstring). No system message support
        exists in this loader today - `history` is inserted directly
        before the current turn with no system entry.
        """
        if raw_image is not None and not isinstance(raw_image, Image.Image):
            raise TypeError(f"Expected PIL Image or None, got {type(raw_image)}")
        if raw_image is not None and raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")
        conversation = list(history) + [self._build_user_message(raw_image, prompt)]
        return self._generate_from_messages(conversation)

    def _generate_from_messages(self, conversation: list[dict]) -> str:
        inputs = self.processor.apply_chat_template(
            conversation,
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
        self._maybe_add_charset_logits_processor(gen_kwargs)

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        gen = generated_ids[:, inputs["input_ids"].shape[1]:]
        decoded = self.processor.batch_decode(gen, skip_special_tokens=True)
        result_text = decoded[0].strip() if decoded else ""

        if self.config.stop_string and result_text.endswith(self.config.stop_string):
            result_text = result_text[: -len(self.config.stop_string)].rstrip()

        return result_text

    def _parse_classification(self, file_path: str, raw_output: str):
        raise NotImplementedError(
            "Lfm2VlLoader is not assigned a classification role."
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
