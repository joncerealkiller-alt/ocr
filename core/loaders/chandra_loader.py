"""
Chandra loader — raw OCR stage, first half of a two-stage extraction
pipeline (2026-07-13, per Jon's direction: "even if we have to use a
non-prompt OCR engine... use a VLM or LLM to combine them after
extraction").

GENUINELY DIFFERENT interface from every other loader in this project:
Chandra is not a free-form instruction-following VLM - it's a fixed-
task OCR engine (prompt_type="ocr_layout" etc., not natural-language
prompts), accessed through its own dedicated `chandra-ocr` package
(InferenceManager / BatchInputItem / generate_hf / parse_markdown), not
a plain transformers generate() call. Same architectural category as
FlorenceLoader and OlmOcrLoader - _run_generate() ALWAYS runs the same
fixed OCR task regardless of what prompt string is passed in, since our
column-format prompts don't apply to a task-fixed model.

Built on Qwen3-VL architecture (per its own HF tags), no
trust_remote_code needed - confirmed via official examples, unlike
PaddleOCR-VL which showed concrete, current bug reports (numeric
decoding errors, version breakage) that made it a real risk rather
than a hypothetical one for this project's numeric-heavy census data.

Requires: pip install chandra-ocr[hf]

Role in the two-stage pipeline: this loader's raw markdown output
becomes STAGE 1 input - a separate structuring pass (see
core/row_extraction.py's build_structuring_prompt / run_two_stage_
extraction) then asks an already-proven instruction-following model
(Qwen3-VL-4B, Gemma) to organize Chandra's raw reading into our actual
census column schema, given BOTH the row image and Chandra's raw text -
not text-only, so stage 2 can cross-reference the source image directly
if Chandra's reading looks ambiguous, rather than being blind to it.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from core.loaders.base_loader import BaseLoader


class ChandraLoader(BaseLoader):
    def initialize_model_and_tokenizer(self) -> tuple[Any, Any, Any]:
        # Set BEFORE any chandra import - confirmed necessary (2026-07-13,
        # real test): row 2 of a live run produced 24,737 chars (vs. 128
        # for row 1), the same repetition/degeneration signature every
        # other model needed a fix for tonight. Unlike every other loader,
        # I could NOT find evidence of a fine-grained repetition_penalty/
        # no_repeat_ngram_size control at chandra's generate_hf() level -
        # the only confirmed lever is MAX_OUTPUT_TOKENS (env var or CLI
        # --max-output-tokens flag per chandra's own docs), which bounds
        # the COST of a runaway loop without preventing the underlying
        # tendency to loop at all. Set here (not left to the environment)
        # because a single row crop legitimately needs a small fraction
        # of chandra's whole-page-scale default (documented as 8192-
        # 12384 depending on version) - capping it low makes a future
        # loop cheap and fast instead of burning the full budget, same
        # mitigation already used for GLM-OCR's degeneration earlier
        # this session. chandra-ocr uses pydantic-settings, which
        # typically reads env vars at first settings-object access, not
        # per-call - set as early as possible, before any chandra import,
        # to maximize the chance this is actually respected. UNCONFIRMED
        # without a live test that this env var is read at the point I'm
        # setting it - worth verifying the next run actually stays bounded.
        import os
        os.environ.setdefault("MAX_OUTPUT_TOKENS", str(self.config.max_new_tokens))

        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "chandra-ocr's HF backend needs transformers with "
                "AutoModelForImageTextToText support. "
                "Install with: pip install chandra-ocr[hf]"
            ) from e

        import torch

        model = AutoModelForImageTextToText.from_pretrained(
            self.config.repo_id, dtype=torch.bfloat16, device_map="auto",
        ).eval()
        processor = AutoProcessor.from_pretrained(self.config.repo_id)
        # Confirmed necessary from the official example - left-padding
        # required for Chandra's batched generation path, not an
        # arbitrary choice.
        processor.tokenizer.padding_side = "left"
        model.processor = processor  # chandra's generate_hf() expects
                                      # the processor attached to the
                                      # model object directly, not
                                      # passed as a separate argument

        if hasattr(model, "hf_device_map"):
            device_counts = {}
            for layer, device in model.hf_device_map.items():
                device_counts[str(device)] = device_counts.get(str(device), 0) + 1
            print(f"[ChandraLoader] Device placement: {device_counts}")
            print(f"[ChandraLoader] torch.cuda.is_available(): {torch.cuda.is_available()}")

        self._execution_meta = {
            "device_map": "auto",
            "max_memory": None,
            "vram_headroom_gb": self.config.vram_headroom_gb,
            "cpu_offload_limit_gb": self.config.cpu_offload_limit_gb,
        }
        self._oom_recovered = False

        self.model = model
        self.tokenizer = processor.tokenizer
        self.processor = processor
        return model, self.tokenizer, processor

    def _build_prompt(self, task: str) -> str:
        # Deliberately ignores whatever this would normally return -
        # see module docstring. Chandra doesn't take free-form prompts;
        # this exists only to satisfy BaseLoader's interface.
        return "ocr_layout"

    def _run_generate(self, raw_image: Any, prompt: str) -> str:
        """
        ALWAYS runs Chandra's fixed OCR/layout task, ignoring whatever
        `prompt` string was passed in (see module docstring - same
        pattern as FlorenceLoader/OlmOcrLoader). Returns raw markdown
        text with layout information, NOT confidence-tagged structured
        fields - this is stage 1 of a two-stage pipeline, not a final
        extractor.
        """
        try:
            from chandra.model.hf import generate_hf
            from chandra.model.schema import BatchInputItem
            from chandra.output import parse_markdown
        except ImportError as e:
            raise ImportError(
                "The 'chandra-ocr' package is required for ChandraLoader. "
                "Install with: pip install chandra-ocr[hf]"
            ) from e

        if not isinstance(raw_image, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(raw_image)}")
        if raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")

        batch = [BatchInputItem(image=raw_image, prompt_type="ocr_layout")]
        result = generate_hf(batch, self.model)[0]
        markdown = parse_markdown(result.raw)
        return markdown

    def _parse_classification(self, file_path: str, raw_output: str):
        raise NotImplementedError(
            "ChandraLoader is not assigned a classification role. "
            "Classification is handled by GemmaLoader per pipeline.yaml."
        )

    def _parse_extraction(self, file_path: str, category: str, raw_output: str):
        raise NotImplementedError(
            "ChandraLoader is a raw-OCR stage (stage 1 of a two-stage "
            "pipeline), not a confidence-tagged extractor - it has no "
            "way to produce our column schema natively. This Schema: "
            "FAIL is EXPECTED, same pattern as FlorenceLoader/"
            "OlmOcrLoader. Read raw_output directly, or use "
            "run_two_stage_extraction() in core/row_extraction.py to "
            "feed this output into a structuring pass."
        )
