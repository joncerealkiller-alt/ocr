"""
Gemma loader — currently used only for the classification stage.

NOTE: model_id/exact HF repo, processor class, and reasoning-toggle
mechanism below are placeholders pending confirmation of the exact
Gemma vision-capable checkpoint you're using (the LM Studio testing
referenced "google/gemma-4-e2b" — confirm the equivalent HF repo id
and whether it ships as AutoModelForImageTextToText or a
model-specific class before running this for real).

Parsing strategy: key:value lines rather than JSON. Smaller models
were observed throughout testing to wrap JSON in markdown fences,
add trailing commentary after the closing brace, or emit malformed
JSON under any output-length pressure. A line-based key:value parser
degrades more predictably and makes partial-failure detection
(missing required key) simpler than a JSON parse try/except.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from core.loaders.base_loader import BaseLoader, GenerationConfig
from core.schema import ClassificationResult, ExtractionResult, DocumentCategory


REQUIRED_CLASSIFY_KEYS = {"category", "confidence", "reason"}


class GemmaLoader(BaseLoader):
    def initialize_model_and_tokenizer(self) -> tuple[Any, Any, Any]:
        # Correction from earlier placeholder: Gemma-4 uses
        # AutoModelForCausalLM per the actual model card, not
        # AutoModelForImageTextToText or AutoModelForMultimodalLM as
        # previously guessed. Also note the -it (instruction-tuned)
        # suffix on the repo id - the base (non-it) checkpoint is not
        # what you want for prompted extraction/classification tasks.
        from transformers import AutoProcessor, AutoModelForCausalLM

        processor = AutoProcessor.from_pretrained(
            self.config.repo_id,
            token=False,
        )

        # Config-driven, not hardcoded to any one model: extra.load_in_8bit
        # (default off) lets a specific config opt into bitsandbytes 8-bit
        # loading - e.g. config/models/gemma_e4b.yaml, where the raw bf16
        # weights (16.02GB) exceed this card's total VRAM (15.93GB) on
        # their own. dtype="auto" is omitted when quantizing: bitsandbytes
        # manages compute precision for the quantized layers itself, and
        # passing both together is redundant/conflicting.
        load_in_8bit = self.config.extra.get("load_in_8bit", False)
        model_kwargs = dict(device_map="auto", token=False)
        if load_in_8bit:
            from transformers import BitsAndBytesConfig
            # Real bug caught before trusting any E2B-vs-E4B result (2026-08-02):
            # quantizing the WHOLE model - including vision_tower/embed_vision
            # (confirmed real module names via modeling_gemma4.py, not guessed) -
            # produced coherent-but-wrong output ("the image is largely blank",
            # low confidence) on images the same model reads correctly at bf16.
            # This is a known failure mode for naive int8 quantization of
            # multimodal models: the vision/projector path is more precision-
            # sensitive than the text decoder. Skipping these modules keeps
            # them in full precision while still 8-bit quantizing the LLM
            # backbone - the actual VRAM-saving target.
            # lm_head added after a second real bug: transformers normally
            # auto-detects and skips tied-weight output heads like lm_head
            # on its own, but supplying an explicit llm_int8_skip_modules
            # list REPLACES that default detection rather than extending
            # it - without lm_head listed explicitly here, generation
            # crashed with AttributeError: 'Parameter' object has no
            # attribute 'CB' (lm_head left as a bare, non-bitsandbytes-
            # wrapped Parameter that the quantized forward path still
            # tried to call).
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=["vision_tower", "embed_vision", "audio_tower", "embed_audio", "lm_head"],
            )
        else:
            model_kwargs["dtype"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            self.config.repo_id,
            **model_kwargs,
        ).eval()

        self._execution_meta = {
            "device_map": "auto",
            "max_memory": self.config.build_max_memory_map(),
            "vram_headroom_gb": self.config.vram_headroom_gb,
            "cpu_offload_limit_gb": self.config.cpu_offload_limit_gb,
        }
        self._oom_recovered = False

        self.model = model
        self.processor = processor
        # Previously left as the BaseLoader default (None) - real gap
        # found 2026-07-24 wiring restrict_output_charset support: every
        # OTHER loader in this project sets self.tokenizer = processor.
        # tokenizer, and _maybe_add_charset_logits_processor() (base_
        # loader.py) needs it. Brought in line rather than special-cased.
        self.tokenizer = processor.tokenizer
        return model, self.tokenizer, processor

    def _build_prompt(self, task: str) -> str:
        if task != "classify":
            raise NotImplementedError(
                "GemmaLoader is currently scoped to classification only. "
                "Extraction role not yet assigned pending re-testing."
            )

        base_prompt = self.config.prompt_text
        if not base_prompt:
            raise ValueError(
                "config.prompt_text is empty — load it from "
                "config/prompts/classifier_classify_v1.txt before calling classify()."
            )
        return base_prompt

    def _run_generate(self, raw_image: Image.Image, prompt: str) -> str:
        if raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")

        # System prompt is config-driven, NOT hardcoded here (2026-07-30
        # fix, per Jon's direction, after a real bug: this used to be a
        # hardcoded classifier persona applied to EVERY call through this
        # loader regardless of task - harmless for gemma.yaml's actual
        # classification role, but silently injected into gemma_extract's
        # extraction calls too whenever a caller invokes _run_generate()
        # directly (e.g. benchmark/prompt_sweep.py, which bypasses
        # _build_prompt()'s task gate by design), making an OCR/structuring
        # call behave like "I am a strict router" instead. See docs/
        # CODE_MAP.md's "Architectural principle: config owns behavior,
        # loaders own mechanics" for the full story. Each model config now
        # sets its OWN extra.system_prompt (gemma.yaml keeps the classifier
        # wording, gemma_extract.yaml gets its own extraction-appropriate
        # one) - DELIBERATELY no hardcoded fallback here: a config that
        # doesn't set one gets no system message at all, not an
        # accidentally-inherited personality from whichever config
        # happened to define one first.
        #
        # Reasoning toggle: confirmed mechanism is a <|think|> token at
        # the START of the system prompt, not a generate() kwarg (see
        # gemma.yaml's reasoning_toggle_mechanism comment). Applied here
        # regardless of whether a system_prompt is configured, so the
        # toggle still works even for a config with none set.
        system_content = self.config.extra.get("system_prompt", "")
        if self.config.reasoning_enabled:
            system_content = "<|think|>" + system_content

        # Modality order matters per the model card: image content must
        # precede text content in the message, not follow it (this is
        # the reverse of how the Qwen/InternVL loaders in this project
        # structure their content lists - don't copy that pattern here).
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": raw_image},
                {"type": "text", "text": prompt},
            ],
        })

        # Image token budget: classification doesn't need fine-grained
        # detail (no OCR happening at this stage), so use a low budget
        # per the model card's own guidance ("lower budgets for
        # classification"). This is a real speed lever specific to
        # Gemma-4 that the other loaders don't have an equivalent of.
        image_token_budget = self.config.extra.get("image_token_budget", 140)

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.config.reasoning_enabled,
        )
        inputs = self.processor(
            text=text,
            images=raw_image,
            image_seq_length=image_token_budget,
            return_tensors="pt",
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        # Deliberate deviation from the model card's recommended sampling
        # config (temperature=1.0, top_p=0.95, top_k=64): those defaults
        # are tuned for open-ended generation quality, not for the
        # deterministic, low-fabrication-risk behavior this pipeline
        # needs. do_sample stays governed by config/models/gemma.yaml
        # (currently false) rather than silently adopting the model
        # card's suggested values. Revisit only if greedy decoding
        # underperforms in actual testing.
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
        self._maybe_add_charset_logits_processor(gen_kwargs)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        response = self.processor.decode(
            outputs[0][input_len:], skip_special_tokens=False
        )

        # processor.parse_response strips/separates the <|channel>thought
        # block per the model card, when thinking was enabled. When
        # thinking is disabled the channel tags may still be emitted
        # (empty thought block) except for E2B/E4B variants per the card -
        # parse_response handles both cases, so always route through it
        # rather than hand-rolling a strip.
        parsed = self.processor.parse_response(response)
        final_text = parsed.get("content", parsed) if isinstance(parsed, dict) else parsed
        return str(final_text).strip()

    # -- parsing -----------------------------------------------------------

    @staticmethod
    def _parse_kv_block(raw_output: str) -> dict[str, str]:
        """
        Parses 'key: value' lines. Ignores anything before the first
        recognized key (strips leaked reasoning/preamble) and stops at
        the first blank line after keys begin, so trailing commentary
        doesn't get absorbed into a field value.

        Tolerates an optional leading "-"/"*" markdown bullet marker
        before the key (2026-07-25, real bug found via a live
        classification run: Gemma formatted its ENTIRE response as a
        bullet list - "- category: dense_tabular_rows", "- confidence:
        1.0", etc. - and the original whitespace-only prefix regex
        rejected every single line, not just the ones that ended up
        "missing": all 8 fields were present and readable, but zero
        matched, so every field silently failed to parse and the
        image was routed to uncertain_review with a "missing required
        fields" error that named only the 3 REQUIRED keys, masking
        that this was a 100% parse failure, not a partial one).
        """
        result: dict[str, str] = {}
        lines = raw_output.splitlines()
        started = False
        for line in lines:
            match = re.match(r"^\s*[-*]?\s*([a-zA-Z_]+)\s*:\s*(.*)$", line)
            if match:
                key, value = match.group(1).lower(), match.group(2).strip()
                result[key] = value
                started = True
            elif started and line.strip() == "":
                break
        return result

    @staticmethod
    def _to_bool(value: str) -> bool | None:
        v = value.strip().lower()
        if v in ("true", "yes", "1"):
            return True
        if v in ("false", "no", "0"):
            return False
        return None

    @staticmethod
    def _to_float(value: str) -> float | None:
        try:
            return float(value.strip())
        except (ValueError, AttributeError):
            return None

    def _parse_classification(self, file_path: str, raw_output: str) -> ClassificationResult:
        fields = self._parse_kv_block(raw_output)

        missing = REQUIRED_CLASSIFY_KEYS - fields.keys()
        if missing:
            raise ValueError(
                f"Classifier output missing required fields {missing} for "
                f"{file_path}. Raw output: {raw_output[:200]!r}. "
                "This file should be routed to uncertain_review by the caller."
            )

        try:
            category = DocumentCategory(fields["category"].strip())
        except ValueError as e:
            raise ValueError(
                f"Classifier returned unrecognized category "
                f"{fields['category']!r} for {file_path}."
            ) from e

        confidence = self._to_float(fields["confidence"])
        if confidence is None:
            raise ValueError(f"Could not parse confidence from {fields['confidence']!r}")

        return ClassificationResult(
            file_path=file_path,
            category=category,
            confidence=confidence,
            text_density=self._to_float(fields.get("text_density", "")),
            handwriting=self._to_bool(fields.get("handwriting", "")),
            table_layout=self._to_bool(fields.get("table_layout", "")),
            faces=self._to_bool(fields.get("faces", "")),
            map_like=self._to_bool(fields.get("map_like", "")),
            reason=fields["reason"][:280],
            model=self.config.model_name,
            prompt_version=self.config.prompt_version,
        )

    def _parse_extraction(self, file_path: str, category: str, raw_output: str) -> ExtractionResult:
        raise NotImplementedError(
            "GemmaLoader is not currently assigned an extraction role. "
            "See config/pipeline.yaml — extraction buckets are routed "
            "to other models pending re-testing."
        )
