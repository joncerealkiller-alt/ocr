"""
deepseek-vl2-tiny loader - extraction-stage CANDIDATE, prep/sanity-test
only (2026-07-28). NOT wired into config/pipeline.yaml - registering
this loader only makes it available to the same testing tools every
other model profile already uses (model_assessment.py, ad-hoc
diagnostics scripts), it does not activate it for any real bucket.

Runs via SubprocessLoaderBase, same pattern as MoondreamLoader - see
core/loaders/_deepseek_vl2_worker.py's module docstring for the full,
confirmed compatibility story (transformers 5.12.1 doesn't recognize
this checkpoint's "deepseek_vl_v2" architecture at all; the real load
path needs DeepSeek's own un-pip-installable deepseek_vl2 package
cloned to .venv_deepseek_vl2/DeepSeek-VL2-src/, an older pinned
transformers==4.38.2, a tokenizers-ABI workaround, and an xformers
bypass for this GPU's too-new compute capability).

Setup this loader depends on (already done as of 2026-07-28, see the
worker script's docstring for exactly why each step exists):
    python -m venv --system-site-packages .venv_deepseek_vl2
    .venv_deepseek_vl2/Scripts/python.exe -m pip install "transformers==4.38.2" accelerate
    .venv_deepseek_vl2/Scripts/python.exe -m pip install "tokenizers==0.21.4"  # overrides the pin below
    # then hand-loosen .venv_deepseek_vl2/Lib/site-packages/transformers/
    # dependency_versions_table.py's "tokenizers" entry to "tokenizers>=0.14"
    # (that file's own runtime check hard-ImportErrors otherwise)
    .venv_deepseek_vl2/Scripts/python.exe -m pip install attrdict timm einops
    .venv_deepseek_vl2/Scripts/python.exe -m pip install xformers --no-deps  # stub only, see worker docstring - the real fix is the SDPA monkeypatch, not this package
    git clone --depth 1 https://github.com/deepseek-ai/DeepSeek-VL2.git .venv_deepseek_vl2/DeepSeek-VL2-src
"""

from __future__ import annotations

from pathlib import Path

from core.loaders.subprocess_loader_base import SubprocessLoaderBase
from core.schema import (
    ExtractionResult, PersonalName, PlaceName, VisibleDate, DocumentCategory,
)
from core.extraction_parsing import parse_kv_block, parse_pipe_entries, parse_keyword_list, REQUIRED_KEYS


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class DeepseekVL2Loader(SubprocessLoaderBase):
    VENV_DIR = PROJECT_ROOT / ".venv_deepseek_vl2"
    WORKER_SCRIPT = Path(__file__).resolve().parent / "_deepseek_vl2_worker.py"

    def extra_request_fields(self) -> dict:
        # Threads real generation settings through to the worker's
        # generate() call - see _deepseek_vl2_worker.py's _query(),
        # which reads these back out of the request's `extra` dict.
        return {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "repetition_penalty": self.config.repetition_penalty,
        }

    def _build_prompt(self, task: str) -> str:
        if task != "extract":
            raise NotImplementedError(
                "DeepseekVL2Loader is not assigned a classification role."
            )
        return self.config.prompt_text

    def _parse_classification(self, file_path: str, raw_output: str):
        raise NotImplementedError(
            "DeepseekVL2Loader is not assigned a classification role."
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
