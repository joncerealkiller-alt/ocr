"""
Granite Vision 4.1-4B loader - extraction-stage CANDIDATE, prep/sanity-
test only (2026-07-28).

This exact model was already looked at once before and deliberately
skipped (see granite_vision_loader.py's own docstring: "unlike
granite-vision-4.1-4b, which was deliberately skipped for this reason
[trust_remote_code/custom modeling files] - see project history
2026-07-11"). Re-checked now because IBM's own model card states native
transformers support as of >=5.8.0 - confirmed true against this
project's current main venv (transformers 5.12.1): AutoConfig.from_
pretrained() resolves Granite4VisionForConditionalGeneration with zero
flags, no trust_remote_code, no custom code executed. Whatever blocked
this in July no longer applies - the project's own transformers version
moved past the point where it mattered, not a re-evaluation of the model
itself.

Standard architecture, same calling convention as GraniteVisionLoader
(the 3.2-2b sibling already in this project) and PixtralLoader - single-
call apply_chat_template -> generate -> batch_decode, no venv needed,
runs fully in-process.

Smoke test (2026-07-28, 3 real images): load ~6-14s, ~7.6GB VRAM for
weights (4B params bf16), ~9.6GB peak during generation - the heaviest
candidate checked this round but still comfortable on this machine's
16GB card.

This is a CANDIDATE QUALIFICATION smoke test (loads/VRAM/catastrophic-
failure check), not a Production Evaluation - see models.md's methodology
note. The output-quality notes below came from a whole-page prompt against
a whole dense document image - NOT how this project's real pipeline
handles census/manifest documents (see core/row_extraction.py: tight
per-FIELD crops, one column of one row per call, built specifically
because whole-page extraction already caused a proven fabrication failure
with olmOCR before this candidate existed). This model has NOT been
through a real Production Evaluation (core/row_extraction.py) yet - what
follows is real output, accurately transcribed, but describes behavior at
whole-page scale (an already-known, not-model-specific overload effect),
not a settled verdict on real per-field accuracy. Also: the
JSON-wrapping noted below is not actually a pipeline problem per Jon -
granite-vision-2b does the same thing and the real per-field parser
already handles it; only this smoke test's own whole-page harness
(calling parse_kv_block directly) doesn't tolerate that shape.

Output quality at whole-page scale was the worst of every candidate
checked this round. On the printed manifest, it produced dense, real,
grounded content (real period names/UK addresses, correct sailing date/
ports, corroborated by HunyuanOCR's and LFM2-VL's own reads of the same
image) - but wrapped in broken JSON syntax instead of the plain
key:value block the prompt asked for, so schema parsing fails on format
alone. On the OTHER TWO samples (1911 census, handwritten manifest), it
did something much worse than a format problem: it FABRICATED an entire
39-person fictional family (every single name surnamed "Janssen") on the
census page, pairing each invented name with an increasingly elaborate,
novel made-up excuse for why it supposedly "couldn't read" a name it had
just confidently produced ("handwritten illegibility high", "field
contains scribbles rather than text", "region designated as 'Not
Applicable'" - none of these are the project's real confidence
vocabulary). On the handwritten manifest, it listed ~46 alphabetically-
patterned common surnames (many prefixed "William") as "confirmed"
personal_names, and EVERY COUNTRY AND CANADIAN PROVINCE ON EARTH as
"confirmed" place_names (Switzerland, Japan, Egypt, Persia, Fiji,
Labrador, ...) - not plausible content for any single real document.
This is the exact "confident invented list" pattern config/pipeline.yaml's
own no_ngram_overlap anomaly check describes catching (its own comment
cites InternVL3-8b generating alphabetically-ordered common surnames as
the motivating example) - and on the third sample it happened to use
valid confidence vocabulary throughout, so it PASSED schema validation
and would have silently reached the real pipeline's output looking like
a legitimate, fully-confirmed record. Not recommended without a hard
look at whether this failure mode is fixable (stricter prompt, lower
max_new_tokens, or an anomaly check run before trusting output at all) -
worse than a formatting problem, this is the fabrication failure mode
this project's own design philosophy treats as the most dangerous kind.
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


class GraniteVision41Loader(BaseLoader):
    def initialize_model_and_tokenizer(self) -> tuple[Any, Any, Any]:
        from transformers import AutoProcessor, AutoModelForImageTextToText

        processor = AutoProcessor.from_pretrained(self.config.repo_id, token=False)

        max_memory = self.config.build_max_memory_map()

        model_kwargs = dict(
            dtype=torch.bfloat16,
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
            print(f"[GraniteVision41Loader] Device placement: {device_counts}")

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
                "GraniteVision41Loader is not assigned a classification role."
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
            "GraniteVision41Loader is not assigned a classification role."
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
