"""
GOT-OCR-2.0-hf loader — stage-1 extraction candidate (2026-07-14, per
Jon's voice-relayed suggestion, independently verified before building).

VERIFICATION NOTES (2026-07-14): the original suggestion claimed a
"1.3B version" - WRONG, corrected via two independent official sources
(HF transformers docs + the original paper): GOT-OCR2 is 580M
parameters total. Also confirmed a real trap to avoid: the ORIGINAL/
community repos (stepfun-ai/GOT-OCR2_0, ucaslcl/GOT-OCR2_0, and every
fork of them) explicitly require trust_remote_code=True in every single
example found - a direct conflict with this project's policy. The
NATIVE transformers integration used here instead - repo_id
stepfun-ai/GOT-OCR-2.0-hf (note the "-hf" suffix) - needs NO remote
code at all, confirmed via official transformers docs.

GENUINELY DIFFERENT ROLE from most loaders, same category as
FlorenceLoader/ChandraLoader/OlmOcrLoader: this is a fixed-task pure
OCR engine, not an instruction-following VLM. The official example
shows NO text/task prompt at all in the processor call - just images
in, text out. _run_generate() therefore IGNORES whatever prompt is
passed, matching that pattern (nothing for a caller to configure here
beyond which image to read). Possible real trade-off, unconfirmed:
official docs note "this implementation of the model will only output
plain text" - suggests the native "-hf" version may not expose the
original's fancier region/color-targeted OCR modes (ocr_box, ocr_color
etc. from the trust_remote_code version's .chat() interface). Worth
testing directly rather than assumed to be a limitation in practice.

Official calling convention (confirmed via huggingface.co/docs/
transformers/model_doc/got_ocr2), genuinely different from every other
loader's chat-template pattern - no messages/apply_chat_template at
all, images passed directly to the processor:
    processor([image], return_tensors="pt", device=device).to(device)
    model.generate(**inputs, do_sample=False, tokenizer=processor.tokenizer,
                    stop_strings="<|im_end|>", max_new_tokens=...)
    processor.batch_decode(generate_ids[:, inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)
"""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from core.loaders.base_loader import BaseLoader


class GotOcr2Loader(BaseLoader):
    def initialize_model_and_tokenizer(self) -> tuple[Any, Any, Any]:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            self.config.repo_id, use_fast=True, token=False
        )

        max_memory = self.config.build_max_memory_map()

        model_kwargs = dict(
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
            print(f"[GotOcr2Loader] Device placement: {device_counts}")
            print(f"[GotOcr2Loader] torch.cuda.is_available(): {torch.cuda.is_available()}")

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
        # Deliberately ignores whatever this would normally return - see
        # module docstring. GOT-OCR-2.0-hf's basic call takes no text/
        # task prompt at all; this exists only to satisfy BaseLoader's
        # interface.
        return ""

    def _run_generate(self, raw_image: Any, prompt: str) -> str:
        """
        ALWAYS runs plain OCR, ignoring whatever `prompt` string was
        passed in (see module docstring - same pattern as
        FlorenceLoader/ChandraLoader/OlmOcrLoader). Returns raw
        transcribed text, NOT confidence-tagged structured fields -
        this is a stage-1 candidate for the two-stage pipeline
        (core/row_extraction.py's run_two_stage_extraction), not a
        final extractor on its own.
        """
        if not isinstance(raw_image, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(raw_image)}")
        if raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")

        inputs = self.processor(
            [raw_image], return_tensors="pt"
        ).to(self.model.device)

        stop_strings = self.config.stop_string or "<|im_end|>"

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                do_sample=self.config.do_sample,
                tokenizer=self.processor.tokenizer,
                stop_strings=stop_strings,
                max_new_tokens=self.config.max_new_tokens,
            )

        decoded = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return decoded[0].strip() if decoded else ""

    def _parse_classification(self, file_path: str, raw_output: str):
        raise NotImplementedError(
            "GotOcr2Loader is not assigned a classification role. "
            "Classification is handled by GemmaLoader per pipeline.yaml."
        )

    def _parse_extraction(self, file_path: str, category: str, raw_output: str):
        raise NotImplementedError(
            "GotOcr2Loader is a raw-OCR stage (stage 1 of a two-stage "
            "pipeline), not a confidence-tagged extractor - it has no "
            "way to produce our column schema natively. This Schema: "
            "FAIL is EXPECTED, same pattern as FlorenceLoader/"
            "OlmOcrLoader/ChandraLoader. Read raw_output directly, or use "
            "run_two_stage_extraction() in core/row_extraction.py to "
            "feed this output into a structuring pass."
        )
