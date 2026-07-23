"""
Moondream2 loader — extraction stage candidate.

vikhyatk/moondream2 ships as a `trust_remote_code=True` repo (auto_map
in config.json points AutoModelForCausalLM at hf_moondream.HfMoondream,
not a native transformers architecture), and its remote code only
produces correct output on transformers==4.52.4 (its own declared
version) - on this project's main transformers (5.12.1), generation
silently degenerates to garbage ("1" followed by hundreds of blank-token
lines), confirmed by direct A/B test, not a guess. Since two transformers
versions can't coexist in one Python process, this loader does NOT load
the model itself. Instead it drives a persistent subprocess
(core/loaders/_moondream_worker.py) running under .venv_moondream - a
separate venv, --system-site-packages so it shares the main env's
torch/CUDA install without a second multi-GB download, with transformers
pinned to 4.52.4 - talking to it over stdin/stdout as newline-delimited
JSON. See _moondream_worker.py's own docstring for the wire protocol and
the full compatibility story.

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

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from core.loaders.base_loader import BaseLoader
from core.schema import (
    ExtractionResult, PersonalName, PlaceName, VisibleDate, DocumentCategory,
)
from core.extraction_parsing import parse_kv_block, parse_pipe_entries, parse_keyword_list, REQUIRED_KEYS


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VENV_DIR = PROJECT_ROOT / ".venv_moondream"
WORKER_SCRIPT = Path(__file__).resolve().parent / "_moondream_worker.py"

# Windows venvs put the interpreter under Scripts/, not bin/ - this loader
# is developed/run on Windows (see repo's PowerShell-first tooling), but
# fall back to bin/python for portability if this ever runs elsewhere.
_VENV_PYTHON_CANDIDATES = [
    VENV_DIR / "Scripts" / "python.exe",
    VENV_DIR / "bin" / "python",
]


class MoondreamLoader(BaseLoader):
    def initialize_model_and_tokenizer(self) -> tuple[Any, Any, Any]:
        venv_python = next(
            (p for p in _VENV_PYTHON_CANDIDATES if p.exists()), None
        )
        if venv_python is None:
            raise RuntimeError(
                f"No moondream2 worker venv found at {VENV_DIR}. Create it "
                f"with:\n  python -m venv --system-site-packages {VENV_DIR}\n"
                f"  {VENV_DIR}/Scripts/python.exe -m pip install "
                "transformers==4.52.4\n"
                "(--system-site-packages reuses this environment's already-"
                "installed torch/CUDA rather than re-downloading it.)"
            )

        revision = self.config.extra.get("revision", "main")

        self._proc = subprocess.Popen(
            [str(venv_python), str(WORKER_SCRIPT), self.config.repo_id, str(revision)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        ready_line = self._proc.stdout.readline()
        if ready_line.strip() != "READY":
            stderr_output = self._proc.stderr.read()
            self._proc.kill()
            raise RuntimeError(
                f"moondream2 worker failed to start (repo_id="
                f"{self.config.repo_id!r}, revision={revision!r}). Expected "
                f"'READY', got {ready_line!r}. Worker stderr:\n{stderr_output}"
            )

        self._execution_meta = {
            "device_map": "moondream_worker_subprocess",
            "max_memory": None,
            "vram_headroom_gb": self.config.vram_headroom_gb,
            "cpu_offload_limit_gb": self.config.cpu_offload_limit_gb,
        }
        self._oom_recovered = False

        # No in-process model/tokenizer/processor - everything lives in the
        # subprocess. Kept as None rather than fabricating placeholders, so
        # any code that actually tries to use self.model directly (it
        # shouldn't - _run_generate is the only thing that talks to the
        # worker) fails loudly instead of silently doing nothing.
        self.model = None
        self.tokenizer = None
        self.processor = None
        return None, None, None

    def release(self) -> None:
        """
        Explicit, deterministic subprocess teardown - added 2026-07-22
        after tracing a real gap: core.row_extraction._release_model()'s
        own `del loader` only removes ITS internal local binding, not
        the CALLER's - the caller's own `loader` variable keeps this
        object alive until the caller's frame itself exits. Relying on
        __del__ alone meant the worker subprocess (with its own
        separate CUDA context, in a different venv/process entirely -
        NOT freed by the parent process's torch.cuda.empty_cache(),
        which only touches the calling process's own CUDA context)
        could keep running well past the point run_two_stage_
        extraction() intended it to be freed before loading the SECOND
        model - undermining that function's own documented "avoid two
        models resident in VRAM simultaneously" guarantee. Now called
        explicitly by _release_model() (see that function's docstring)
        via this override, rather than left to non-deterministic GC
        timing.
        """
        self._terminate_worker()

    def _terminate_worker(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.poll() is None:
            try:
                proc.stdin.close()
            except Exception:
                pass
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._proc = None

    def __del__(self):
        # Last-resort safety net ONLY - release() (called explicitly by
        # _release_model()) is the real cleanup path now. This still
        # matters for any code path that constructs a MoondreamLoader
        # without ever calling _release_model (e.g. an error before
        # extraction starts) - better an eventual GC-triggered cleanup
        # than a permanently orphaned subprocess.
        self._terminate_worker()

    def _build_prompt(self, task: str) -> str:
        if task != "extract":
            raise NotImplementedError(
                "MoondreamLoader is not assigned a classification role."
            )
        return self.config.prompt_text

    def _run_generate(self, raw_image: Any, prompt: str) -> str:
        if not isinstance(raw_image, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(raw_image)}")

        if self._proc.poll() is not None:
            stderr_output = self._proc.stderr.read()
            raise RuntimeError(
                f"moondream2 worker process died (exit code "
                f"{self._proc.returncode}). Stderr:\n{stderr_output}"
            )

        if raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")

        # IPC goes through a temp file rather than serializing pixel data
        # over the pipe - simplest, and matches how the rest of this
        # codebase already hands images between processes/tools (see
        # model_assessment.py's preprocessed_path).
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            raw_image.save(tmp_path)

            request = {
                "image_path": tmp_path,
                "question": prompt,
                "reasoning": self.config.reasoning_enabled,
            }
            self._proc.stdin.write(json.dumps(request) + "\n")
            self._proc.stdin.flush()

            response_line = self._proc.stdout.readline()
            if not response_line:
                stderr_output = self._proc.stderr.read()
                raise RuntimeError(
                    "moondream2 worker closed its stdout unexpectedly "
                    f"(exit code {self._proc.poll()}). Stderr:\n{stderr_output}"
                )
            response = json.loads(response_line)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if "error" in response:
            raise RuntimeError(f"moondream2 worker error: {response['error']}")

        return response["answer"]

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
