"""
Moondream2 loader - extraction stage candidate.

vikhyatk/moondream2 ships as a `trust_remote_code=True` repo (auto_map
in config.json points AutoModelForCausalLM at hf_moondream.HfMoondream,
not a native transformers architecture), and its remote code only
produces correct output on transformers==4.52.4 (its own declared
version) - on this project's main transformers (5.12.1), generation
silently degenerates to garbage ("1" followed by hundreds of blank-token
lines), confirmed by direct A/B test, not a guess. Since two transformers
versions can't coexist in one Python process, this loader runs the model
in a separate venv via SubprocessLoaderBase (core/loaders/
subprocess_loader_base.py) - see that file's docstring for the shared
plumbing (venv resolution, spawn+READY handshake, JSON I/O, cleanup) and
why it now exists as its own reusable base rather than living only here:
a real inventory of other candidate models' own config.json files
confirmed this pattern needs to generalize (moondream3-preview alone
wants a DIFFERENT pinned version, 4.51.1, than this model's 4.52.4).

This class now only supplies what's actually moondream-specific: the
worker script path, the `reasoning` extra request field, and the
extraction-parsing logic (_parse_extraction/_build_prompt) shared with
every other loader's own extraction schema.

Network cutoff (LoRA "variant" download): moondream.py's query() (and
friends) thread an optional `settings={"variant": ...}` dict down to
lora.py's variant_state_dict(), which - only when variant_id is not
None - calls lora.py's cached_variant_path(), which does a bare
urllib.request.urlopen() against MOONDREAM_ENDPOINT (default
https://api.moondream.ai), sending MOONDREAM_API_KEY as a header if set.
Neither this loader nor the worker script ever passes settings/variant
(so that branch is dead code as called here) - but that's a property of
these call sites, not of the model code, so the worker script also
monkeypatches lora.py's cached_variant_path() to raise instead of opening
a socket, as a second layer against a future edit accidentally passing a
variant. On top of that, the worker's model load uses
local_files_only=True, so loading itself can't reach the Hub either.
"""

from __future__ import annotations

from pathlib import Path

from core.loaders.subprocess_loader_base import SubprocessLoaderBase
from core.schema import (
    ExtractionResult, PersonalName, PlaceName, VisibleDate, DocumentCategory,
)
from core.extraction_parsing import parse_kv_block, parse_pipe_entries, parse_keyword_list, REQUIRED_KEYS


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class MoondreamLoader(SubprocessLoaderBase):
    VENV_DIR = PROJECT_ROOT / ".venv_moondream"
    WORKER_SCRIPT = Path(__file__).resolve().parent / "_moondream_worker.py"

    def extra_request_fields(self) -> dict:
        return {"reasoning": self.config.reasoning_enabled}

    def _build_prompt(self, task: str) -> str:
        if task != "extract":
            raise NotImplementedError(
                "MoondreamLoader is not assigned a classification role."
            )
        return self.config.prompt_text

    def _parse_classification(self, file_path: str, raw_output: str):
        raise NotImplementedError(
            "MoondreamLoader is not assigned a classification role."
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
