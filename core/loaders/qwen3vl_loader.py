"""
Qwen3-VL loader — extraction stage.

Uses Qwen3VLForConditionalGeneration (Qwen3 series, not Qwen2.5).
Follows the official model card pattern:
  - Qwen3VLForConditionalGeneration (specific class for Qwen3)
  - AutoProcessor.from_pretrained()
  - apply_chat_template() with add_generation_prompt=True
  - Image passed as local PIL Image object (not URL)
  - Single-image assessment context (batch_decode trimmed to single output)

Generation defaults from model card VL hyperparameters:
  - do_sample: False (greedy decoding)
  - top_p: 0.8, top_k: 20, temperature: 0.7
  - repetition_penalty: 1.0
  - presence_penalty: conditionally passed if set (TBD via other instance)

Requires transformers >= 4.57.0 (or dev build):
  pip install git+https://github.com/huggingface/transformers
"""

from __future__ import annotations

import re
from typing import Any

import torch
from PIL import Image

from core.loaders.base_loader import BaseLoader
from core.schema import (
    ExtractionResult, PersonalName, PlaceName, VisibleDate,
    ConfidenceLevel, DocumentCategory,
)
from core.extraction_parsing import parse_kv_block, parse_pipe_entries, parse_keyword_list, REQUIRED_KEYS


class Qwen3VLLoader(BaseLoader):
    def initialize_model_and_tokenizer(self) -> tuple[Any, Any, Any]:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

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
        if self.config.attn_implementation:
            model_kwargs["attn_implementation"] = self.config.attn_implementation
        if max_memory is not None:
            # Reserves vram_headroom_gb for context-window growth during
            # long extraction runs, and gives device_map a CPU fallback
            # target so a single oversized image doesn't crash the whole
            # batch with an OOM - it offloads instead, slower but survives.
            model_kwargs["max_memory"] = max_memory

        model = Qwen3VLForConditionalGeneration.from_pretrained(
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
            print(f"[Qwen3VLLoader] Device placement: {device_counts}")
            print(f"[Qwen3VLLoader] max_memory passed to from_pretrained: {max_memory}")
            print(f"[Qwen3VLLoader] torch.cuda.is_available(): {torch.cuda.is_available()}")

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
        self.tokenizer = processor.tokenizer
        self.processor = processor
        return model, self.tokenizer, processor

    def _build_prompt(self, task: str) -> str:
        """
        task is "classify" or "extract". Qwen3-VL does not support a
        runtime reasoning toggle - "Thinking" is a separate checkpoint
        (repo_id), not a parameter or system-prompt switch. So
        reasoning_toggle_mechanism="checkpoint" (set when repo_id points
        at a -Thinking checkpoint) is expected and not warned on; anything
        else other than "unsupported"/"" gets a warning since this loader
        has no other mechanism to act on it.
        """
        if self.config.reasoning_toggle_mechanism not in ("unsupported", "checkpoint", ""):
            print(f"[Qwen3VLLoader] Warning: reasoning_toggle_mechanism is "
                  f"{self.config.reasoning_toggle_mechanism!r}, which this "
                  f"loader doesn't know how to act on. Ignoring.")
        return self.config.prompt_text

    # Hardcoded, not a config/prompt-file setting: this is a Qwen3VL-specific
    # mitigation for one confirmed failure mode (subject_keywords drift),
    # not a general pipeline prompt convention yet. If this pattern proves
    # out, promoting it to a proper prompts/*.txt file is worth doing later.
    _KEYWORD_ONLY_PROMPT = (
        "Look at the same document image again. Output EXACTLY one line, "
        "in this format, and nothing else:\n\n"
        "subject_keywords: <keyword>, <keyword>, ...\n\n"
        "Rules:\n"
        "  - Maximum 10 keywords.\n"
        "  - Each keyword must name something directly visible or named in "
        "this specific document (a place, an event type, a record type, "
        "a named institution) - not a general archival/technical/legal "
        "category that could apply to any document of this kind.\n"
        "  - Do not include generic domain terms unless that exact phrase "
        "appears on the document itself.\n"
        "  - Output nothing before or after this one line."
    )

    _SUBJECT_KEYWORDS_MARKER = re.compile(
        r"^\s*subject_keywords\s*:", re.IGNORECASE | re.MULTILINE
    )

    _THINK_CLOSE_TAG = "</think>"

    def _strip_thinking(self, text: str) -> str:
        """
        Thinking checkpoints emit a reasoning trace wrapped through a
        closing </think> tag before the actual answer. If we don't strip
        this, _parse_kv_block would either fail outright or - worse -
        silently pick up spurious "key: value"-shaped lines from inside
        the reasoning text itself, since models often restate structure
        while thinking through a problem.
        """
        if not self.config.reasoning_enabled:
            return text
        idx = text.rfind(self._THINK_CLOSE_TAG)
        if idx == -1:
            print(f"[Qwen3VLLoader] reasoning_enabled=True but no "
                  f"{self._THINK_CLOSE_TAG!r} found in output - likely "
                  f"truncated mid-reasoning (max_new_tokens too small for "
                  f"the reasoning phase to complete). Returning raw text "
                  f"as-is; downstream parsing will likely fail on it.")
            return text.strip()
        return text[idx + len(self._THINK_CLOSE_TAG):].strip()

    def _run_generate(self, raw_image: Any, prompt: str) -> str:
        """
        For extraction prompts (detected by presence of a subject_keywords:
        field marker), splits generation into two separate model.generate()
        calls rather than one:

          1. Core fields (document_type/visible_dates/personal_names/
             place_names) at the normal max_new_tokens budget - these have
             been consistently clean across testing.
          2. subject_keywords alone, under a small dedicated token ceiling
             (config.keywords_max_new_tokens).

        This exists because subject_keywords showed unbounded drift that
        scaled with whatever token budget was available (44/128/282
        keywords at 250/512/1024 tokens) regardless of prompt wording,
        repetition_penalty, no_repeat_ngram_size, or an explicit stop_string
        instruction - none of which gave it a reason to stop on its own.
        An external per-field token ceiling is the only thing that reliably
        bounded it in testing. The two outputs are merged into one raw_output
        string so _parse_kv_block/_parse_extraction downstream are unchanged.

        Non-extraction prompts (no subject_keywords marker) fall through to
        a single generate() call, unaffected by any of this.
        """
        if not isinstance(raw_image, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(raw_image)}")

        if raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")

        if not self._SUBJECT_KEYWORDS_MARKER.search(prompt):
            return self._generate_once(raw_image, prompt, self.config.max_new_tokens)

        core_prompt = self._SUBJECT_KEYWORDS_MARKER.split(prompt, maxsplit=1)[0].rstrip()
        core_prompt += (
            "\n\nDo NOT output a subject_keywords line in this response. "
            "Stop immediately after the place_names line."
        )

        core_output = self._generate_once(raw_image, core_prompt, self.config.max_new_tokens)
        keyword_output = self._generate_once(
            raw_image, self._KEYWORD_ONLY_PROMPT, self.config.keywords_max_new_tokens
        )

        # keyword_output is expected to already be "subject_keywords: ..."
        # per the isolated prompt's required format. If the model ignored
        # that format entirely (no colon), wrap it defensively so the merged
        # output still has a parseable field line rather than silently
        # dropping whatever it produced.
        if ":" not in keyword_output.split("\n", 1)[0]:
            keyword_output = f"subject_keywords: {keyword_output}"

        return f"{core_output}\n{keyword_output}"

    def _generate_once(self, raw_image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        """
        Single generate() call. Shared by both the core-fields call and the
        isolated keywords call in _run_generate, so generation-parameter
        handling (sampling gate, repetition_penalty, stop_strings, the
        presence_penalty fallback) isn't duplicated between them.
        """
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
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        # Mirrors QwenLoader's gating: sampling params are meaningless (and
        # can trigger transformers warnings) when do_sample=False, so only
        # pass them when sampling is actually enabled.
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
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
        if self.config.presence_penalty is not None:
            gen_kwargs["presence_penalty"] = self.config.presence_penalty
        if self.config.stop_string:
            # HF's stop_strings needs the tokenizer passed at call time (not
            # just at model init) to match the string across token boundaries.
            gen_kwargs["stop_strings"] = [self.config.stop_string]
            gen_kwargs["tokenizer"] = self.processor.tokenizer

        with torch.inference_mode():
            try:
                generated_ids = self.model.generate(**inputs, **gen_kwargs)
            except TypeError as e:
                if "presence_penalty" in gen_kwargs and "presence_penalty" in str(e):
                    print(f"[Qwen3VLLoader] presence_penalty not accepted by "
                          f"this model's generate(): {e}. Retrying without it.")
                    gen_kwargs.pop("presence_penalty")
                    generated_ids = self.model.generate(**inputs, **gen_kwargs)
                else:
                    raise

        # Trim to only the new tokens (exclude prompt)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        result_text = output_text[0].strip() if output_text else ""
        result_text = self._strip_thinking(result_text)

        # generate() with stop_strings includes the terminator itself in the
        # output - strip it here so raw_output/reports stay clean rather than
        # every successful call ending in a dangling "<END>" line.
        if self.config.stop_string and result_text.endswith(self.config.stop_string):
            result_text = result_text[: -len(self.config.stop_string)].rstrip()

        return result_text

    def _parse_classification(self, file_path: str, raw_output: str):
        raise NotImplementedError(
            "Qwen3VLLoader is not assigned a classification role. "
            "Classification is handled by GemmaLoader per pipeline.yaml."
        )

    def _parse_extraction(self, file_path: str, category: str, raw_output: str) -> ExtractionResult:
        """
        Parses the shared key:value block format (core/extraction_parsing.py)
        that every extraction-role loader uses - this is the actual contract
        the extraction prompts produce, not a reinvented format. Using the
        shared parser (not a local copy) keeps this comparable with every
        other extraction loader in assessment reports, and means a parsing
        fix only needs to be made in one place.
        """
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
