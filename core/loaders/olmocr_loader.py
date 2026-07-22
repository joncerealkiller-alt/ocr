"""
olmOCR loader — OCR/transcription candidate, printed_document bucket.
Built 2026-07-11 after a blind AllenAI playground test
(playground.allenai.org/model/olmocr-2-7b-1025) produced the strongest
raw transcription result of the entire session on the Knott marriage
certificate - see GROUND_TRUTH_Knott_marriage_cert.md for the field-by-
field comparison. Only 3 specific errors (Brantford->Brampton,
Lawlor->Lawton, Gough->Yonge, all plausible common-word-override
misreads) against an otherwise complete, correctly-structured,
non-fabricated transcription - no repetition loops, no category
contamination, no invented content.

GENUINELY DIFFERENT from every other loader in this project in one
important way: this is allenai/olmOCR-2-7B-1025-FP8, a real fine-tune of
Qwen2.5-VL-7B-Instruct (270k pages supervised, plus GRPO reinforcement
learning that specifically rewarded the model for passing verifiable
correctness checks - table structure, math fidelity, clean termination).
The prompt it was RL-trained against is a specific structured YAML
format built by a function in AllenAI's own `olmocr` PyPI package, not
free-text instructions we can vary. This loader has a REAL, NEW
dependency on that package (`pip install olmocr`) and imports its
`build_no_anchoring_v4_yaml_prompt()` directly, rather than
reverse-engineering an equivalent prompt by hand - the exact prompt
structure appears load-bearing for the quality seen in testing, and
hand-rolling it risks silently diverging from what the model was
actually trained against.

"no_anchoring" (as opposed to the toolkit's anchored variant, which
injects extracted PDF text-layer metadata) is the correct choice for
our use case, not a fallback - confirmed via AllenAI's own source
(olmocr/data/buildsilver.py): anchoring is for born-digital PDFs with
an extractable text layer; our input is a rendered image with no such
layer, exactly the case the no-anchoring prompt variant is for.

Model expects images resized so the LONGEST dimension is 1288px - a
hard requirement from the official model card, not a testable
preprocessing option like the profiles in core/image_preprocessing.py.
Applied unconditionally here, not exposed as a toggle.

_build_prompt() ALWAYS returns build_no_anchoring_v4_yaml_prompt()'s
output, ignoring whatever prompt file is selected in the assessment
tool's UI - deliberate, not a bug. The trained prompt IS the point of
using this model; varying it the way we iterate prompts for every other
loader would work against the model's actual training.

SCOPING DECISION, matching FlorenceLoader: this is treated as a
high-quality OCR/transcription stage, NOT a confidence-tagged extractor
matching our ExtractionResult schema. The playground output is
excellent, well-labeled, clean text - but produces no confidence tags
of its own. Rather than inventing a fragile mechanism to guess
confidence levels the model never stated, _parse_extraction raises a
clear NotImplementedError (Schema: FAIL is EXPECTED, same as Florence)
and the real signal is in the raw_output pane - the YAML front-matter
metadata plus the clean document transcription.

Calling convention mirrors QwenLoader exactly (same Qwen2.5-VL base
architecture, proven working in this codebase already): two-step
apply_chat_template(tokenize=False) + processor() call via
qwen_vl_utils.process_vision_info, not the newer single-call pattern
used by Qwen3-VL/SmolVLM2/Granite/GLM-OCR.
"""

from __future__ import annotations

import re
from typing import Any

import torch
import yaml as yaml_module
from PIL import Image

from core.loaders.base_loader import BaseLoader

TARGET_LONGEST_DIM = 1288  # hard model requirement, not configurable


class OlmOcrLoader(BaseLoader):
    def initialize_model_and_tokenizer(self) -> tuple[Any, Any, Any]:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        processor_repo = self.config.processor_repo_id or self.config.repo_id
        processor = AutoProcessor.from_pretrained(processor_repo, token=False)

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

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.repo_id, **model_kwargs
        ).eval()

        if hasattr(model, "hf_device_map"):
            device_counts = {}
            for layer, device in model.hf_device_map.items():
                device_counts[str(device)] = device_counts.get(str(device), 0) + 1
            print(f"[OlmOcrLoader] Device placement: {device_counts}")
            print(f"[OlmOcrLoader] torch.cuda.is_available(): {torch.cuda.is_available()}")
            print(f"[OlmOcrLoader] Processor loaded from: {processor_repo}")

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
        # Deliberately ignores config.prompt_text / whatever prompt file
        # is selected in the UI - see module docstring. The RL-trained
        # prompt structure is the point of this model.
        try:
            from olmocr.prompts import build_no_anchoring_v4_yaml_prompt
        except ImportError as e:
            raise ImportError(
                "The 'olmocr' package is required for OlmOcrLoader (used "
                "for its RL-trained prompt structure, not general PDF "
                "tooling we need here). Install with: pip install olmocr"
            ) from e
        return build_no_anchoring_v4_yaml_prompt()

    def _run_generate(self, raw_image: Any, prompt: str) -> str:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as e:
            raise ImportError(
                "qwen_vl_utils is required (same dependency QwenLoader "
                "already needs). Install with: pip install qwen-vl-utils"
            ) from e

        if not isinstance(raw_image, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(raw_image)}")

        if raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")

        # Hard model requirement, not a testable preprocessing toggle -
        # resize so the longest dimension is exactly TARGET_LONGEST_DIM.
        longest = max(raw_image.width, raw_image.height)
        if longest != TARGET_LONGEST_DIM:
            scale = TARGET_LONGEST_DIM / longest
            new_size = (round(raw_image.width * scale), round(raw_image.height * scale))
            raw_image = raw_image.resize(new_size, Image.LANCZOS)

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

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        result_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return result_text

    def _parse_classification(self, file_path: str, raw_output: str):
        raise NotImplementedError(
            "OlmOcrLoader is not assigned a classification role. "
            "Classification is handled by GemmaLoader per pipeline.yaml."
        )

    def _parse_extraction(self, file_path: str, category: str, raw_output: str):
        # See module docstring "SCOPING DECISION" - this is a
        # high-quality OCR/transcription stage, not a confidence-tagged
        # extractor. Schema: FAIL here is EXPECTED, same pattern as
        # FlorenceLoader - the real signal is in raw_output.
        metadata = self._try_parse_front_matter(raw_output)
        metadata_note = f" Parsed metadata: {metadata}." if metadata else ""
        raise NotImplementedError(
            "OlmOcrLoader produces high-quality transcribed text with "
            "YAML front-matter metadata (language, rotation, table/"
            "diagram flags), not confidence-tagged structured fields "
            "matching our schema. This Schema: FAIL is EXPECTED - read "
            "raw_output directly for the transcription."
            f"{metadata_note} If you want structured fields from this "
            "text, that requires a separate downstream extraction pass "
            "(same two-stage architecture explored for Florence-2), not "
            "something this loader does natively."
        )

    @staticmethod
    def _try_parse_front_matter(raw_output: str) -> dict:
        """
        Best-effort extraction of YAML front matter, if present, purely
        to surface it in the diagnostic message above - not load-bearing
        for anything else. olmOCR's raw output format wasn't directly
        confirmed before building this (only the AllenAI playground's
        rendered display was seen, which may reformat the model's
        literal output) - this defensively returns {} rather than
        raising if no recognizable ---delimited block is found, so a
        format mismatch here doesn't block reading the actual
        transcription text.
        """
        match = re.search(r"^---\s*\n(.*?)\n---\s*\n", raw_output, re.DOTALL)
        if not match:
            return {}
        try:
            parsed = yaml_module.safe_load(match.group(1))
            return parsed if isinstance(parsed, dict) else {}
        except yaml_module.YAMLError:
            return {}
