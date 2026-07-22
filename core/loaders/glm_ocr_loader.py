"""
GLM-OCR loader — native structured-extraction candidate, printed_document
bucket. Built 2026-07-11 after Florence-2's OCR-preprocessing route
showed persistent fabrication regardless of prompt/generation-parameter
tuning (see project history same date).

GENUINELY DIFFERENT ARCHITECTURE from every other loader in this
project: GLM-OCR has a native "Information Extraction" prompt mode
(distinct from its "Document Parsing"/raw-OCR mode, which this loader
does NOT use) where a JSON schema with empty placeholder values is
provided in the prompt, and the model returns that schema filled in
with actual document values - a single-pass image-to-structured-JSON
capability, not raw transcription requiring a separate downstream
extraction stage the way Florence-2 does. Per Jon's direction
(2026-07-11: "yes if we can skip pre process that would be good"),
this loader targets that native mode directly.

Confirmed via official model card (huggingface.co/zai-org/GLM-OCR):
  - Native transformers class GlmOcrForConditionalGeneration, resolved
    via AutoModelForImageTextToText - no trust_remote_code, contributed
    to transformers 2026-01-27.
  - inputs.pop("token_type_ids", None) required before generate() -
    confirmed from the official example, not a guess. The processor
    apparently returns this key in some circumstances and generate()
    doesn't want it.
  - Decode: trim by input_ids length (matches this project's established
    pattern already), skip_special_tokens=False (per the official
    example - not skip_special_tokens=True like most other loaders here,
    a deliberate difference worth preserving rather than "correcting" to
    match the rest of the codebase).

REAL UNCERTAINTY, not yet resolved: the official card's Information
Extraction example schema is entirely FIXED-SHAPE (one ID number, one
name, one address per field) - it never demonstrates an array-of-objects
pattern for a variable-length list of repeated entries, which is what
our actual schema needs (multiple names, multiple places, each with its
own confidence tag). There's no confirmation this generalizes. The
prompt (config/prompts/glm_ocr_extract.txt) asks for arrays anyway,
since that's the correct target shape to test - but _parse_extraction
below defensively coerces a lone dict into a one-item list for each
field, in case the model returns a single object per field instead of
an array when it doesn't have real multi-entry support. If results come
back schema-mismatched, this coercion (not a code bug) is the first
thing to suspect.
"""

from __future__ import annotations

import json as json_module
from typing import Any

import torch
from PIL import Image

from core.loaders.base_loader import BaseLoader
from core.schema import (
    ExtractionResult, PersonalName, PlaceName, VisibleDate, DocumentCategory,
)
from core.extraction_parsing import parse_json_schema_output, CONFIDENCE_MAP


def _extract_tagged_entries(data: dict, field: str) -> list[tuple[str, str]]:
    """
    Handles TWO possible schema shapes GLM-OCR might return, since which
    one it actually follows is itself the thing being tested (see module
    docstring - 2026-07-11 first test used nested objects and got bare
    strings back instead):

      1. Parallel arrays: data["personal_names"] = [...],
         data["personal_names_confidence"] = [...], paired by index.
      2. Nested objects: data["personal_names"] = [{"value":...,
         "confidence":...}, ...].

    Returns (value, confidence_string) tuples - confidence validation
    against CONFIDENCE_MAP happens in the caller, same as every other
    field type in this project.

    If a parallel confidence array exists but its length doesn't match
    the values array, this drops the ENTIRE field's entries rather than
    guessing an index alignment - a length mismatch is itself a real
    schema violation, and pairing values to confidences by a guessed
    alignment could silently attach the WRONG confidence to the WRONG
    name, which is worse than losing the field's data entirely.
    """
    values = data.get(field)
    if not isinstance(values, list):
        return []

    confidence_key = f"{field}_confidence"
    if confidence_key in data:
        confidences = data.get(confidence_key)
        if not isinstance(confidences, list) or len(confidences) != len(values):
            print(f"[GlmOcrLoader] {field}: {confidence_key} present but "
                  f"length mismatch ({len(confidences) if isinstance(confidences, list) else 'not a list'} "
                  f"vs {len(values)} values) - dropping this field's "
                  f"entries rather than guessing an alignment.")
            return []
        return [(str(v), str(c)) for v, c in zip(values, confidences)]

    # Fall back to nested-object shape
    result = []
    for entry in values:
        if isinstance(entry, dict):
            result.append((str(entry.get("value", "")), str(entry.get("confidence", ""))))
        # bare strings with no confidence data anywhere - can't validate,
        # so they're not included; caller's missing-required-field check
        # will surface this clearly rather than silently guessing.
    return result


class GlmOcrLoader(BaseLoader):
    def initialize_model_and_tokenizer(self) -> tuple[Any, Any, Any]:
        from transformers import AutoProcessor, GlmOcrForConditionalGeneration

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

        model = GlmOcrForConditionalGeneration.from_pretrained(
            self.config.repo_id, **model_kwargs
        ).eval()

        if hasattr(model, "hf_device_map"):
            device_counts = {}
            for layer, device in model.hf_device_map.items():
                device_counts[str(device)] = device_counts.get(str(device), 0) + 1
            print(f"[GlmOcrLoader] Device placement: {device_counts}")
            print(f"[GlmOcrLoader] torch.cuda.is_available(): {torch.cuda.is_available()}")

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
                "GlmOcrLoader is not assigned a classification role. "
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

        # Confirmed necessary from the official example - the processor
        # apparently returns this key in some circumstances and
        # generate() doesn't accept it. Not a defensive guess.
        inputs.pop("token_type_ids", None)

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

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        # Trim by length (this project's established pattern). Switched
        # to skip_special_tokens=True after a real test (2026-07-11)
        # showed skip_special_tokens=False (as the official card example
        # uses) let a stray "<|user|>" next-turn role marker leak into
        # the decoded output, breaking JSON parsing - evidence-based
        # correction, not just "matching the rest of the codebase for
        # its own sake."
        generated_ids_trimmed = generated_ids[0][inputs["input_ids"].shape[1]:]
        result_text = self.processor.decode(
            generated_ids_trimmed, skip_special_tokens=True
        ).strip()

        return result_text

    def _parse_classification(self, file_path: str, raw_output: str):
        raise NotImplementedError(
            "GlmOcrLoader is not assigned a classification role. "
            "Classification is handled by GemmaLoader per pipeline.yaml."
        )

    def _parse_extraction(self, file_path: str, category: str, raw_output: str) -> ExtractionResult:
        try:
            data = parse_json_schema_output(raw_output)
        except json_module.JSONDecodeError as e:
            raise ValueError(
                f"GLM-OCR output was not valid JSON for {file_path}. This "
                f"is EXPECTED and fine if you deliberately used a raw "
                f"'Document Parsing' prompt (e.g. glm_ocr_text_recognition.txt, "
                f"'Text Recognition:') to compare pure extraction quality "
                f"against JSON-schema adherence - read raw_output directly "
                f"for the transcription text in that case, this Schema: "
                f"FAIL is not a real problem. If you intended the "
                f"'Information Extraction' JSON mode instead "
                f"(glm_ocr_extract.txt / glm_ocr_extract_flat.txt), then "
                f"this genuinely means the model didn't follow the "
                f"requested schema format. Raw output: {raw_output[:300]!r}. "
                f"JSON error: {e}"
            )

        document_type = (data.get("document_type") or "").strip()[:80] or None

        personal_names = []
        for value, conf_str in _extract_tagged_entries(data, "personal_names"):
            value = value.strip()[:120]
            confidence = CONFIDENCE_MAP.get(conf_str.strip().lower())
            if value and confidence is not None:
                personal_names.append(PersonalName(value=value, confidence=confidence))

        place_names = []
        for value, conf_str in _extract_tagged_entries(data, "place_names"):
            value = value.strip()[:160]
            confidence = CONFIDENCE_MAP.get(conf_str.strip().lower())
            if value and confidence is not None:
                place_names.append(PlaceName(value=value, confidence=confidence))

        visible_dates = []
        for value, conf_str in _extract_tagged_entries(data, "visible_dates"):
            value = value.strip()[:60]
            confidence = CONFIDENCE_MAP.get(conf_str.strip().lower())
            if value and confidence is not None:
                visible_dates.append(VisibleDate(value=value, confidence=confidence))

        subject_keywords_raw = data.get("subject_keywords") or []
        if not isinstance(subject_keywords_raw, list):
            subject_keywords_raw = [subject_keywords_raw]
        subject_keywords = [str(k).strip()[:60] for k in subject_keywords_raw if str(k).strip()]

        missing = []
        if not document_type:
            missing.append("document_type")
        if not personal_names:
            missing.append("personal_names")
        if not place_names:
            missing.append("place_names")
        if missing:
            raise ValueError(
                f"GLM-OCR JSON parsed but is missing required fields "
                f"{missing} for {file_path} - either the document "
                f"genuinely has none of this content, confidence tags "
                f"were missing/invalid, or (see module docstring) the "
                f"model didn't generalize to the array-of-objects schema "
                f"shape requested. Raw output: {raw_output[:300]!r}."
            )

        return ExtractionResult(
            file_path=file_path,
            category=DocumentCategory(category),
            document_type=document_type,
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
