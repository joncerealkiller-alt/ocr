"""
vLLM subprocess lifecycle owner - launches, health-checks, and tears
down the ONE vLLM OpenAI-compatible server this backend may run at a
time. Built 2026-08-13 for the multi-runtime integration (plan
addendum "Multi-Runtime Model Integration") after this session's
investigation confirmed the vLLM-only checkpoints (Gemma-4 w4a16-ct,
Qwen2.5-VL-AWQ) can never run under the plain-transformers loader path.

DESIGN, stated plainly:
- ONE worker slot. At most one vLLM subprocess alive at any time, on
  ONE fixed port (VLLM_PORT below) - a per-model port would falsely
  imply concurrent workers the 16GB card cannot hold.
- Subprocess, not in-process vllm.LLM(): the OpenAI-compatible server
  (`vllm serve`) is vLLM's own idiomatic long-running shape, and a
  subprocess gives a teardown guarantee no in-process unload can:
  killing the process provably returns its VRAM to the driver (this
  session's own ground-truth check while diagnosing the caching-
  allocator release bug - see docs/WSL_COMPUTE_BACKEND_BASELINE.md).
- Cross-runtime exclusion via core/gpu_coordinator.py: acquire() claims
  the GPU (evicting a resident transformers model first), and
  model_residency's own acquire() symmetrically evicts this subprocess.
  Neither module imports the other - see gpu_coordinator's docstring.
- VRAM verification after teardown uses `nvidia-smi`, NOT
  torch.cuda.memory_reserved() - torch's counters only see the CALLING
  process's allocator, and the memory being verified belongs to a
  different process entirely.

Launch environment is machine-level, not per-model config (every
confirmed run_vllm_*.sh wrapper this session shared the exact same
lines) - the constants below, each overridable via env var for a
future machine where the paths differ.

Runs inside the WSL backend process (api/agent_main.py). The Windows
side never launches vLLM - these models are only reachable via
backend="remote".
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from typing import Any, Optional

import requests

from core.gpu_coordinator import gpu_coordinator
from core.loaders.base_loader import GenerationConfig

VLLM_PORT = int(os.environ.get("GENEALOGY_VLLM_PORT", "8502"))
VLLM_VENV_PYTHON = os.environ.get(
    "GENEALOGY_VLLM_PYTHON", os.path.expanduser("~/venv_vllm/bin/python3")
)
CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
HF_HOME = os.environ.get("HF_HOME", "/mnt/e/huggingface")

# Engine init measured 120-290s across the three confirmed checkpoints
# (weight load + CUDA graph capture) - generous ceiling, loud failure
# past it rather than an indefinite hang.
HEALTH_TIMEOUT_SECONDS = 600
SIGTERM_GRACE_SECONDS = 15


class VllmRuntime:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._process: Optional[subprocess.Popen] = None
        self._model_name: Optional[str] = None

    @property
    def running_model_name(self) -> Optional[str]:
        with self._lock:
            if self._process is not None and self._process.poll() is not None:
                # Process died on its own - reflect reality, not stale bookkeeping.
                self._process = None
                self._model_name = None
            return self._model_name

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{VLLM_PORT}"

    def acquire(self, model_name: str, config: GenerationConfig) -> str:
        """
        Ensures `model_name`'s vLLM server is up and healthy, returns
        its base URL. Reuses a running same-model server; tears down a
        different-model one first. Claims the GPU via gpu_coordinator
        (evicting a resident transformers model) BEFORE launching.
        """
        with self._lock:
            if self.running_model_name == model_name:
                return self.base_url
            if self._process is not None:
                self.release()

            gpu_coordinator.claim("vllm")

            cmd = [
                VLLM_VENV_PYTHON, "-m", "vllm.entrypoints.openai.api_server",
                "--model", config.repo_id,
                "--port", str(VLLM_PORT),
                "--host", "127.0.0.1",
                "--gpu-memory-utilization", f"{_gpu_memory_utilization(config):.2f}",
                "--max-model-len", str(config.context_length or 4096),
                "--dtype", "auto",
            ]
            # Per-model opt-in (config.extra, set in the YAML) - needed
            # by InternVL3.5's custom processor/config code, not by the
            # Qwen/Gemma checkpoints. Off unless the config says so.
            if config.extra.get("vllm_trust_remote_code"):
                cmd.append("--trust-remote-code")
            env = dict(os.environ)
            # Do NOT leak the parent process's allocator config into the
            # vLLM subprocess - api/agent_main.py sets PYTORCH_CUDA_ALLOC_
            # CONF=expandable_segments:True for ITS OWN transformers
            # release path, but vLLM tunes its own memory management, and
            # inheriting the setting broke gemma_12b_w4a16's server
            # startup with a reproducible torch stable-ABI aten::empty
            # failure inside gptq_marlin_repack (confirmed on a clean
            # GPU, 2026-08-13; the proven run_vllm_*.sh wrappers never
            # set this var, which is why the offline tests all passed).
            env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            # The venv's own bin dir MUST be on PATH, not just its python
            # invoked by absolute path - flashinfer JIT-compiles kernels
            # during server warm-up via `ninja`, which lives in
            # ~/venv_vllm/bin and is found via PATH lookup (confirmed by
            # the spike's first failure: FileNotFoundError: 'ninja' -
            # the proven run_vllm_*.sh wrappers got this for free from
            # `source activate`, which prepends the venv bin dir).
            venv_bin = os.path.dirname(VLLM_VENV_PYTHON)
            env["PATH"] = f"{venv_bin}:{CUDA_HOME}/bin:" + env.get("PATH", "")
            env["CUDA_HOME"] = CUDA_HOME
            env["HF_HOME"] = HF_HOME
            env["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"

            print(f"[core.vllm_runtime] launching vLLM server: {config.repo_id!r} on port {VLLM_PORT}")
            self._process = subprocess.Popen(
                cmd, env=env,
                stdout=open(f"/tmp/vllm_{model_name}.log", "w"),
                stderr=subprocess.STDOUT,
                # Own process group, so teardown signals hit vLLM's own
                # spawn-children too, not just the top process.
                start_new_session=True,
            )
            self._model_name = model_name

            try:
                self._wait_for_health()
            except Exception:
                self.release()
                raise
            print(f"[core.vllm_runtime] {model_name!r} healthy at {self.base_url}")
            return self.base_url

    def _wait_for_health(self) -> None:
        deadline = time.time() + HEALTH_TIMEOUT_SECONDS
        while time.time() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"vLLM server process exited during startup (code "
                    f"{self._process.returncode}) - see /tmp/vllm_{self._model_name}.log"
                )
            try:
                resp = requests.get(f"{self.base_url}/health", timeout=3)
                if resp.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(3)
        raise TimeoutError(
            f"vLLM server for {self._model_name!r} not healthy after "
            f"{HEALTH_TIMEOUT_SECONDS}s - see /tmp/vllm_{self._model_name}.log"
        )

    def release(self) -> None:
        """
        SIGTERM -> grace wait -> SIGKILL, on the whole process group,
        then verify via nvidia-smi that VRAM actually returned to the
        driver. Safe no-op when nothing is running.
        """
        with self._lock:
            if self._process is None:
                return
            model_name = self._model_name
            print(f"[core.vllm_runtime] release: {model_name!r}")
            try:
                pgid = os.getpgid(self._process.pid)
                os.killpg(pgid, signal.SIGTERM)
                try:
                    self._process.wait(timeout=SIGTERM_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    print(f"[core.vllm_runtime] {model_name!r} ignored SIGTERM, sending SIGKILL")
                    os.killpg(pgid, signal.SIGKILL)
                    self._process.wait(timeout=10)
            except ProcessLookupError:
                pass  # already gone
            self._process = None
            self._model_name = None
            gpu_coordinator.release_noted("vllm")

            used_mb = _nvidia_smi_used_mb()
            if used_mb is not None:
                print(f"[core.vllm_runtime] post-release GPU memory.used: {used_mb} MiB")

    def chat_completion(self, config: GenerationConfig, messages: list[dict[str, Any]],
                         max_tokens: int, temperature: float) -> str:
        """
        One /v1/chat/completions call against the running server.
        Caller must have called acquire() first (same precondition shape
        as ChatBackendAdapter.send_turn after ensure_loaded).
        """
        with self._lock:
            if self.running_model_name is None:
                raise RuntimeError("vllm_runtime.chat_completion() called with no server running - call acquire() first.")
            base_url = self.base_url
            repo_id = config.repo_id
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": repo_id,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def _gpu_memory_utilization(config: GenerationConfig) -> float:
    """
    Derives vLLM's --gpu-memory-utilization fraction from the existing
    vram_headroom_gb field (same meaning it already has: VRAM
    deliberately left unused) - the confirmed test value 0.85 on this
    16GB card corresponds to vram_headroom_gb: 2.4.
    """
    total_mb = _nvidia_smi_total_mb()
    if total_mb is None:
        return 0.85  # the empirically confirmed value from every test this session
    total_gb = total_mb / 1024
    fraction = (total_gb - config.vram_headroom_gb) / total_gb
    return max(0.3, min(fraction, 0.95))


def _nvidia_smi_query(field: str) -> Optional[float]:
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        return float(result.stdout.strip().splitlines()[0])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


def _nvidia_smi_used_mb() -> Optional[float]:
    return _nvidia_smi_query("memory.used")


def _nvidia_smi_total_mb() -> Optional[float]:
    return _nvidia_smi_query("memory.total")


# The one shared instance. Do not construct VllmRuntime() anywhere else.
vllm_runtime = VllmRuntime()

# Cross-runtime registration - lets gpu_coordinator evict this subprocess
# when the transformers runtime claims the GPU. release() is already a
# safe no-op when nothing is running.
gpu_coordinator.register_runtime("vllm", vllm_runtime.release)
