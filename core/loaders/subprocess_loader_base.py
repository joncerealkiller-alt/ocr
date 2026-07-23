"""
SubprocessLoaderBase - reusable plumbing for a loader whose model must
run in a SEPARATE venv (pinned to a transformers/dependency version
incompatible with this project's main environment) rather than loading
in-process like every other loader here.

Extracted 2026-07-22 from MoondreamLoader (the first real case - see
that file's own docstring for the full moondream2/transformers 5.x
incompatibility story) after confirming, via a real inventory of
several other candidate models' own config.json files, that this
pattern needs to be reused: moondream3-preview alone declares a
DIFFERENT transformers_version (4.51.1) than moondream2's pinned
4.52.4, and HunyuanOCR's own subfolders disagree with EACH OTHER
(4.57.1 / 4.49.0 / 4.57.1). This is not a one-venv-fits-all situation -
each distinct required version needs its own venv (though models that
happen to agree on a version can share one), and each new model needs
its own worker script (the actual "how do I call this specific model's
API" logic can't be generalized further than the shared wire protocol
below), but the SUBPROCESS PLUMBING ITSELF - venv resolution, spawn +
READY handshake, the request/response JSON loop, and deterministic
cleanup - is identical every time and belongs in exactly one place.

Subclasses must set two CLASS attributes:
    VENV_DIR: Path       - the venv this model's worker runs under
    WORKER_SCRIPT: Path  - the worker script to spawn

Subclasses may override:
    extra_worker_args() -> list[str]
        Extra positional CLI args appended after repo_id/revision when
        spawning the worker. Default: none.
    extra_request_fields() -> dict
        Extra fields merged into every request dict sent to the
        worker, beyond image_path/question. Default: none. (Moondream's
        subclass uses this for its `reasoning` bool.)

Subclasses still implement (same as every other BaseLoader subclass):
    _build_prompt(), _parse_extraction(), _parse_classification()

WIRE PROTOCOL (the shared contract every worker script must implement -
see _subprocess_worker_common.py for the worker-side helper that
implements the read/dispatch loop, so a new worker script only needs to
supply "how do I load this model" and "how do I call it", not hand-roll
the JSON I/O loop from scratch):
    Request:  {"image_path": str, "question": str, ...extra fields}
    Response: {"answer": str} or {"error": str}
    A single literal "READY" line printed to stdout once the model has
    finished loading, before the request loop starts - the parent
    process blocks on reading that line to know the worker is usable.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from core.loaders.base_loader import BaseLoader


class SubprocessLoaderBase(BaseLoader):
    VENV_DIR: Path  # subclasses must set
    WORKER_SCRIPT: Path  # subclasses must set

    # Windows venvs put the interpreter under Scripts/, not bin/ - this
    # project is developed/run on Windows (see repo's PowerShell-first
    # tooling), but fall back to bin/python for portability if a worker
    # venv is ever created elsewhere.
    _VENV_PYTHON_SUBPATHS = [("Scripts", "python.exe"), ("bin", "python")]

    def _venv_python_candidates(self) -> list[Path]:
        return [self.VENV_DIR.joinpath(*parts) for parts in self._VENV_PYTHON_SUBPATHS]

    def extra_worker_args(self) -> list[str]:
        """Extra positional CLI args appended after repo_id/revision.
        Override for a worker script that needs more than those two -
        default is none."""
        return []

    def extra_request_fields(self) -> dict:
        """Extra fields merged into every request dict sent to the
        worker. Override for model-specific generation params (see
        MoondreamLoader's `reasoning` bool) - default is none."""
        return {}

    def initialize_model_and_tokenizer(self) -> tuple[Any, Any, Any]:
        venv_python = next(
            (p for p in self._venv_python_candidates() if p.exists()), None
        )
        if venv_python is None:
            raise RuntimeError(
                f"No worker venv found at {self.VENV_DIR} for "
                f"{type(self).__name__}. Create it with:\n"
                f"  python -m venv --system-site-packages {self.VENV_DIR}\n"
                f"  {self.VENV_DIR}/Scripts/python.exe -m pip install "
                "<the specific dependency versions this model needs>\n"
                "(--system-site-packages reuses this environment's already-"
                "installed torch/CUDA rather than re-downloading it.)"
            )

        revision = self.config.extra.get("revision", "main")

        self._proc = subprocess.Popen(
            [str(venv_python), str(self.WORKER_SCRIPT), self.config.repo_id,
             str(revision), *self.extra_worker_args()],
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
                f"{type(self).__name__} worker failed to start (repo_id="
                f"{self.config.repo_id!r}, revision={revision!r}). Expected "
                f"'READY', got {ready_line!r}. Worker stderr:\n{stderr_output}"
            )

        self._execution_meta = {
            "device_map": f"{type(self).__name__.lower()}_worker_subprocess",
            "max_memory": None,
            "vram_headroom_gb": self.config.vram_headroom_gb,
            "cpu_offload_limit_gb": self.config.cpu_offload_limit_gb,
        }
        self._oom_recovered = False

        # No in-process model/tokenizer/processor - everything lives in
        # the subprocess. Kept as None rather than fabricating
        # placeholders, so any code that actually tries to use
        # self.model directly (it shouldn't - _run_generate is the only
        # thing that talks to the worker) fails loudly instead of
        # silently doing nothing.
        self.model = None
        self.tokenizer = None
        self.processor = None
        return None, None, None

    def _run_generate(self, raw_image: Any, prompt: str) -> str:
        if not isinstance(raw_image, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(raw_image)}")

        if self._proc.poll() is not None:
            stderr_output = self._proc.stderr.read()
            raise RuntimeError(
                f"{type(self).__name__} worker process died (exit code "
                f"{self._proc.returncode}). Stderr:\n{stderr_output}"
            )

        if raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")

        # IPC goes through a temp file rather than serializing pixel
        # data over the pipe - simplest, and matches how the rest of
        # this codebase already hands images between processes/tools
        # (see model_assessment.py's preprocessed_path). NamedTempFile's
        # handle is closed (the `with` block exits) BEFORE raw_image.
        # save(tmp_path) writes to it - required on Windows, where a
        # file opened by NamedTemporaryFile is exclusively locked until
        # its own handle closes; writing through a second, later open()
        # on the same path only works once the first handle is gone.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            raw_image.save(tmp_path)

            request = {
                "image_path": tmp_path,
                "question": prompt,
                **self.extra_request_fields(),
            }
            self._proc.stdin.write(json.dumps(request) + "\n")
            self._proc.stdin.flush()

            response_line = self._proc.stdout.readline()
            if not response_line:
                stderr_output = self._proc.stderr.read()
                raise RuntimeError(
                    f"{type(self).__name__} worker closed its stdout "
                    f"unexpectedly (exit code {self._proc.poll()}). "
                    f"Stderr:\n{stderr_output}"
                )
            response = json.loads(response_line)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if "error" in response:
            raise RuntimeError(f"{type(self).__name__} worker error: {response['error']}")

        return response["answer"]

    def release(self) -> None:
        """
        Explicit, deterministic subprocess teardown - called by
        core.row_extraction._release_model() / model_assessment.py's
        own copy (see BaseLoader.release()'s docstring for why this
        needs to be explicit rather than left to __del__'s non-
        deterministic timing: a worker subprocess holds its own
        separate CUDA context in a different venv/process entirely,
        not touched by the calling process's torch.cuda.empty_cache()).
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
        # _release_model()) is the real cleanup path. This still
        # matters for any code path that constructs a loader without
        # ever calling _release_model (e.g. an error before extraction
        # starts) - better an eventual GC-triggered cleanup than a
        # permanently orphaned subprocess.
        self._terminate_worker()
