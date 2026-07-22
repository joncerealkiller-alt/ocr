"""
Florence-2 loader — OCR preprocessing candidate, printed_document bucket
only, per 2026-07-11 decision to test a two-stage (OCR-then-structure)
architecture as an alternative to asking one VLM to read AND structure
in a single pass.

FUNDAMENTALLY DIFFERENT ROLE from every other loader in this codebase:
Florence-2 is not a chat-following VLM. It's a prompt-based multi-task
model where the "prompt" is one of a small fixed set of task tokens
(<OCR>, <OCR_WITH_REGION>, <CAPTION>, <OD>, etc.), not a natural-language
instruction. It cannot follow the document_type:/personal_names:/etc.
extraction format every other loader targets. This loader's job is ONLY
to produce clean transcribed text (or region-tagged text) for a second,
separate extraction stage to structure - not to be tested against
ExtractionResult schema itself.

Uses the NATIVE transformers integration (Florence2ForConditionalGeneration,
contributed to transformers 2025-08-20), not the original microsoft/
repo's custom trust_remote_code=True modeling files. Confirmed via
official docs (huggingface.co/docs/transformers/model_doc/florence2):
repo_id should be florence-community/Florence-2-large, not
microsoft/Florence-2-large - same architecture/weights lineage, but the
former loads through a real transformers class.

Calling convention is the TRADITIONAL task-token style, not the chat-
template style also documented for this native integration:
  processor(text=task_token, images=image, return_tensors="pt")
  model.generate(**inputs, max_new_tokens=..., num_beams=...)
  processor.batch_decode(generated_ids, skip_special_tokens=False)
  processor.post_process_generation(generated_text, task=task_token,
                                     image_size=image.size)

skip_special_tokens=False is deliberate, not a bug - post_process_
generation() needs the location/special tokens preserved to correctly
decode bounding-box coordinates for <OCR_WITH_REGION>. Confirmed from
the official model card example, not assumed.

Official example uses num_beams=3 (beam search) rather than the do_sample
greedy/sampling convention every other loader in this project uses -
see GenerationConfig.num_beams in base_loader.py.
"""

from __future__ import annotations

import json
from typing import Any

import torch
from PIL import Image

from core.loaders.base_loader import BaseLoader


class FlorenceLoader(BaseLoader):
    def initialize_model_and_tokenizer(self) -> tuple[Any, Any, Any]:
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        processor = AutoProcessor.from_pretrained(self.config.repo_id, token=False)

        max_memory = self.config.build_max_memory_map()

        model_kwargs = dict(
            torch_dtype="auto",
            device_map="auto",
            token=False,
        )
        if max_memory is not None:
            model_kwargs["max_memory"] = max_memory

        model = Florence2ForConditionalGeneration.from_pretrained(
            self.config.repo_id, **model_kwargs
        ).eval()

        if hasattr(model, "hf_device_map"):
            device_counts = {}
            for layer, device in model.hf_device_map.items():
                device_counts[str(device)] = device_counts.get(str(device), 0) + 1
            print(f"[FlorenceLoader] Device placement: {device_counts}")
            print(f"[FlorenceLoader] torch.cuda.is_available(): {torch.cuda.is_available()}")

        # Needed to cast pixel_values to match the model's loaded dtype -
        # same reasoning as SmolVLM2Loader. Tracked rather than hardcoded
        # in case torch_dtype="auto" resolves differently on this hardware.
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
        # task param is BaseLoader's classify/extract role concept - not
        # applicable here. The actual "prompt" is config.prompt_text,
        # expected to be a literal Florence task token (e.g. "<OCR>"),
        # supplied via a prompt file containing just that token, or the
        # assessment tool's prompt override box. Not validated against
        # the known task-token list here - an invalid token will simply
        # fail visibly at the processor/model call, which is an
        # acceptable way to surface a typo.
        return self.config.prompt_text.strip()

    def _run_generate(self, raw_image: Any, prompt: str) -> str:
        if not isinstance(raw_image, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(raw_image)}")

        if raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")

        task_token = prompt.strip()

        inputs = self.processor(
            text=task_token, images=raw_image, return_tensors="pt"
        ).to(self.model.device, self._model_dtype)

        gen_kwargs = dict(max_new_tokens=self.config.max_new_tokens)
        if self.config.num_beams and self.config.num_beams > 1:
            gen_kwargs["num_beams"] = self.config.num_beams
            gen_kwargs["do_sample"] = False
            # Beam-search-specific - meaningless (and rejected by some
            # generate() versions) outside beam search, only added here.
            if self.config.length_penalty != 1.0:
                gen_kwargs["length_penalty"] = self.config.length_penalty
            if self.config.early_stopping:
                gen_kwargs["early_stopping"] = self.config.early_stopping
        else:
            gen_kwargs["do_sample"] = self.config.do_sample

        # Previously NOT wired despite existing as config fields - real
        # gap, not a new feature. These were the single most effective
        # lever against fabrication/drift for every other loader tested
        # this session (Qwen, SmolVLM2, Granite), but had never actually
        # been tested for Florence-2 before this fix (2026-07-11).
        if self.config.repetition_penalty and self.config.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = self.config.repetition_penalty
        if self.config.no_repeat_ngram_size:
            gen_kwargs["no_repeat_ngram_size"] = self.config.no_repeat_ngram_size

        # Diagnostic - added after length_penalty=0.6 and length_penalty=0.1
        # both produced byte-identical output to a run without it, which
        # static code review couldn't explain. This makes it directly
        # visible whether a given setting actually reached generate(),
        # rather than continuing to guess from output differences alone.
        print(f"[FlorenceLoader] gen_kwargs passed to generate(): {gen_kwargs}")

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        # skip_special_tokens=False is deliberate here - see module
        # docstring. post_process_generation needs the special/location
        # tokens intact to decode bounding boxes correctly for region-
        # aware tasks like <OCR_WITH_REGION>.
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed = self.processor.post_process_generation(
            generated_text, task=task_token, image_size=raw_image.size
        )

        # parsed is a dict (e.g. {"<OCR>": "raw text..."} or
        # {"<OCR_WITH_REGION>": {"quad_boxes": [...], "labels": [...]}}).
        # Serialized to JSON rather than extracting only the plain text,
        # so region/bounding-box data (if present) is preserved and
        # visible in the assessment tool's raw output pane, not silently
        # discarded - that spatial data is the whole point of testing
        # <OCR_WITH_REGION> specifically (see module docstring: this was
        # explored as a possible mechanical fix for "ignore the stamp/
        # instructional text" from earlier prompt-only attempts).
        return json.dumps(parsed, ensure_ascii=False, indent=2)

    def _parse_classification(self, file_path: str, raw_output: str):
        raise NotImplementedError(
            "FlorenceLoader is an OCR preprocessing stage, not a "
            "classification role. Read raw_output directly - it's the "
            "actual artifact this loader produces, not a schema-checked "
            "result. See module docstring."
        )

    def _parse_extraction(self, file_path: str, category: str, raw_output: str):
        raise NotImplementedError(
            "FlorenceLoader cannot produce confidence-tagged structured "
            "extraction - it's an OCR preprocessing stage feeding a "
            "SEPARATE downstream extraction step, not a final extractor "
            "itself. This is expected to show as Schema: FAIL in the "
            "assessment tool - read the raw_output pane directly to "
            "evaluate OCR quality, that's the actual test here. See "
            "module docstring for the two-stage architecture this is "
            "part of."
        )
