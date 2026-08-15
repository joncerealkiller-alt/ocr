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
from core.resource_guard import check_resources_or_raise, log_resource_trend

# Process-name patterns confirmed live 2026-08-15 to match every real
# process this runtime spawns (both the top-level API server and its
# child engine-core worker) - used by _kill_stray_processes() below,
# NOT by the normal SIGTERM/SIGKILL teardown in release() (which tracks
# self._process directly). This is the reconciliation safety net for
# orphans that fall OUTSIDE that tracking - confirmed real: an external
# kill -9 of a test script (before it could tear down its own
# subprocess tree) left a full vLLM server+engine-core pair running
# with nothing in THIS process's bookkeeping aware of them, stranding
# ~13GB of VRAM+RAM until a manual `wsl --shutdown`.
_STRAY_PROCESS_PATTERNS = ("vllm.entrypoints.openai.api_server", "VLLM::EngineCore")

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
        # Engine-startup telemetry (2026-08-16, benchmark backend-
        # comparison work) - mirrors core/model_residency.py's
        # last_load_telemetry, same shape philosophy: captures the ONE
        # most recent acquire() that actually launched a subprocess
        # (the reuse-already-running fast path leaves this untouched,
        # same as residency's own reuse-in-place branch not re-timing a
        # load that didn't happen). None until the first real launch.
        self._last_load_telemetry: Optional[dict[str, Any]] = None

    @property
    def last_load_telemetry(self) -> Optional[dict[str, Any]]:
        return self._last_load_telemetry

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

            # Reconciliation sweep (2026-08-15 hardening pass) - kills
            # any stray vLLM process this runtime's own bookkeeping
            # doesn't know about, BEFORE trusting a clean slate to
            # launch into. See _kill_stray_processes()'s docstring.
            _kill_stray_processes()

            # OOM-hardening pre-flight check - same discipline as
            # core/model_residency.py's acquire(): only on the real-
            # launch path (the reuse-already-running fast path above
            # returns before this), raises ResourceExhaustedError which
            # api/agent_main.py turns into a clean 503.
            check_resources_or_raise(context=f"launching vLLM server for {model_name!r}")

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
            # Skips CUDA-graph capture at startup (2026-08-16, added for
            # the eager-vs-graphs VRAM/throughput experiment -
            # gemma_extract_vllm_w4a16_eager.yaml is the first config to
            # set this). Same opt-in-via-extra pattern as
            # vllm_trust_remote_code above - off unless a config asks for
            # it, so every existing vLLM config's launch command is
            # byte-for-byte unchanged. This is the ONE knob this project
            # has found so far that actually targets the CUDA-graph-
            # capture overhead identified in gemma_extract_vllm.yaml's
            # vram_headroom_gb investigation (that overhead did not
            # shrink with --gpu-memory-utilization OR with a smaller
            # quantized checkpoint - --enforce-eager skips graph capture
            # entirely rather than trying to shrink its footprint).
            if config.extra.get("vllm_enforce_eager"):
                cmd.append("--enforce-eager")
            # KV-cache quantization (2026-08-16, the fourth VRAM lever
            # tried after --gpu-memory-utilization, --enforce-eager, and
            # --max-model-len all failed to meaningfully move this
            # checkpoint's ~15.7-15.9GB floor - see gemma_extract_vllm_
            # w4a16_ctx1024.yaml's own vision_validation_status comment
            # for that history). Distinct mechanism from the other three:
            # shrinks the per-token KV-cache entry size itself (fp8 vs
            # the default fp16/bf16), rather than changing how much of
            # the card vLLM is ALLOWED to use or how large a context it
            # plans for. Same opt-in-via-extra pattern - off unless a
            # config sets vllm_kv_cache_dtype (e.g. "fp8", "fp8_e4m3",
            # "fp8_e5m2"), so every existing config is unaffected. No
            # calibration/scale file required for plain "fp8" (vLLM
            # defaults K/V scales to 1.0 without one - see vLLM's
            # quantized_kvcache docs) - accuracy impact is checkpoint/
            # architecture-dependent (sliding-window attention layers are
            # documented as more sensitive) and gets verified the same
            # way every other lever here has been: a real vision_baseline
            # run, not assumed from the docs alone.
            kv_cache_dtype = config.extra.get("vllm_kv_cache_dtype")
            if kv_cache_dtype:
                cmd.extend(["--kv-cache-dtype", str(kv_cache_dtype)])
            # Fifth VRAM lever (2026-08-16) - caps the max concurrent
            # sequence count vLLM plans for, which bounds two things its
            # own startup log (gpu_worker.py:789) shows separately:
            # "peak activation" memory (profiled for a worst-case batch
            # up to this count) and, via the CUDA-graph batch-size list
            # this project's launches already log spanning 1-512, how
            # many graph shapes get captured. On gemma_extract_vllm_
            # w4a16.yaml's own real log those two categories were 0.26
            # GiB and 0.45 GiB respectively (0.71 GiB combined ceiling,
            # already mostly probed by --enforce-eager's -216 to -470MB
            # result) - default vLLM max_num_seqs is 256; this project's
            # real workload never runs more than 1 concurrent sequence.
            # Same opt-in-via-extra pattern - off unless a config sets
            # vllm_max_num_seqs, every existing config unaffected.
            max_num_seqs = config.extra.get("vllm_max_num_seqs")
            if max_num_seqs:
                cmd.extend(["--max-num-seqs", str(max_num_seqs)])
            # Multimodal item limits (2026-08-15 audit follow-up).
            # Verified from the installed vLLM 0.27.1 source
            # (config/multimodal.py get_limit_per_prompt): an UNSPECIFIED
            # modality defaults to 999 items per prompt - so a Gemma4
            # server (T+I+V+A) is by default provisioned to accept up to
            # 999 images/videos/audios per request, and the 12B log shows
            # a real cost ("Raising max_num_batched_tokens from 2048 to
            # 2496 to accommodate 'video' input"). This project's real
            # workload is exactly ONE image per prompt, never video or
            # audio. Same opt-in-via-extra pattern - a config sets e.g.
            # vllm_limit_mm_per_prompt: '{"image": 1, "video": 0,
            # "audio": 0}' (JSON string passed through verbatim);
            # existing configs unaffected.
            limit_mm = config.extra.get("vllm_limit_mm_per_prompt")
            if limit_mm:
                cmd.extend(["--limit-mm-per-prompt", str(limit_mm)])
            # Resolution capping (2026-08-15, real gap found live): every
            # transformers-path loader already caps image resolution via
            # GenerationConfig.min_pixels/max_pixels BEFORE tokenization
            # (e.g. qwen3b.yaml: 200704-1003520) - vLLM configs never had
            # this wired through at all. Confirmed live: a full-page scan
            # (1968x4880, ~9.6MP) sent to qwen25_vl_7b_awq with no cap
            # produced 12627 image tokens, blowing a 4096 context_length
            # ("Input length exceeds model's maximum context length").
            # --mm-processor-kwargs is vLLM's equivalent knob for Qwen2/
            # 2.5-VL's dynamic-resolution processor - only passed when the
            # config actually sets these fields, so checkpoints that don't
            # need it (Gemma, MiniCPM) are unaffected.
            if config.min_pixels is not None or config.max_pixels is not None:
                import json as _json
                mm_kwargs = {}
                if config.min_pixels is not None:
                    mm_kwargs["min_pixels"] = config.min_pixels
                if config.max_pixels is not None:
                    mm_kwargs["max_pixels"] = config.max_pixels
                cmd.extend(["--mm-processor-kwargs", _json.dumps(mm_kwargs)])
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
            baseline_vram_mb = _nvidia_smi_used_mb()
            load_start = time.time()
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
            load_time_s = round(time.time() - load_start, 3)
            after_load_vram_mb = _nvidia_smi_used_mb()
            # Distinct from core/model_residency.py's last_load_telemetry
            # (task #8: subprocess startup + engine init are NOT the same
            # cost as a transformers weight load - this whole load_time_s
            # includes subprocess spawn, weight load, AND CUDA graph
            # capture, all inseparable from outside the subprocess. See
            # this dict's own "startup_includes" note - recorded
            # explicitly rather than letting a reader assume it's
            # apples-to-apples with residency's load_time_s.
            self._last_load_telemetry = {
                "model_name": model_name,
                "baseline_vram_mb": baseline_vram_mb,
                "after_load_vram_mb": after_load_vram_mb,
                "load_time_s": load_time_s,
                "startup_includes": "subprocess_spawn+weight_load+cuda_graph_capture (not separable)",
            }
            print(f"[core.vllm_runtime] {model_name!r} healthy at {self.base_url} ({load_time_s}s)")
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

        2026-08-15 hardening: ALSO sweeps for stray processes outside
        this instance's own tracking, even when self._process is
        already None - a caller hitting "release/Eject" wants genuine
        confidence the GPU is actually free, not just that THIS
        instance's last-known handle is gone (which could be stale if
        an earlier instance's process leaked past its own tracking).
        """
        with self._lock:
            if self._process is None:
                _kill_stray_processes()
                log_resource_trend("vllm_runtime")
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

            _kill_stray_processes()
            log_resource_trend("vllm_runtime")

    def chat_completion(self, config: GenerationConfig, messages: list[dict[str, Any]],
                         max_tokens: int, temperature: float) -> dict[str, Any]:
        """
        One /v1/chat/completions call against the running server.
        Caller must have called acquire() first (same precondition shape
        as ChatBackendAdapter.send_turn after ensure_loaded).

        Returns a dict {"text": str, "usage": dict|None, "vram_used_mb":
        float|None, "generation_time_s": float} rather than a bare
        string (2026-08-16, benchmark backend-comparison work) - task
        #8 wants tokens/sec and VRAM for a vLLM run exactly as it's
        already available for a transformers run, and vLLM's own
        OpenAI-compatible response already carries a `usage` block
        (prompt_tokens/completion_tokens) this was previously
        discarding.

        Sampling-parameter equivalence (task #6, recorded here rather
        than assumed): the standard OpenAI chat-completions schema only
        has temperature/top_p/max_tokens/stop natively. top_k and
        repetition_penalty are vLLM SERVER EXTENSIONS to that schema
        (not part of the OpenAI spec) - vLLM accepts them as additional
        top-level JSON fields and honors them; a generic OpenAI client
        would silently drop them. no_repeat_ngram_size has NO vLLM
        equivalent at all (vLLM's sampler has no n-gram-repeat-block
        parameter) - it is NOT sent, and is NOT applied, full stop. Any
        config that relies on no_repeat_ngram_size for repetition
        control will behave differently under vLLM than under
        transformers for that reason alone - see benchmark/
        console_runner.py's settings_translation_notes for where this
        gets recorded per-run rather than silently assumed away.
        """
        with self._lock:
            if self.running_model_name is None:
                raise RuntimeError("vllm_runtime.chat_completion() called with no server running - call acquire() first.")
            base_url = self.base_url
            repo_id = config.repo_id

        payload: dict[str, Any] = {
            "model": repo_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if config.top_p is not None:
            payload["top_p"] = config.top_p
        # vLLM extensions (see docstring) - only sent when the config
        # actually sets a non-default value, so a config that never
        # touched these fields produces a request byte-identical to
        # before this change.
        if config.top_k and config.top_k > 0:
            payload["top_k"] = config.top_k
        if config.repetition_penalty and config.repetition_penalty != 1.0:
            payload["repetition_penalty"] = config.repetition_penalty
        if config.stop_string:
            payload["stop"] = [config.stop_string]

        start = time.time()
        resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=300)
        resp.raise_for_status()
        generation_time_s = time.time() - start
        data = resp.json()
        return {
            "text": data["choices"][0]["message"]["content"],
            "usage": data.get("usage"),
            "vram_used_mb": _nvidia_smi_used_mb(),
            "generation_time_s": generation_time_s,
        }


def _kill_stray_processes() -> list[int]:
    """
    Finds and force-kills any process matching _STRAY_PROCESS_PATTERNS
    regardless of whether this runtime instance's own bookkeeping
    (self._process) knows about it - see that constant's docstring for
    the real incident this closes. SIGKILL directly (not the graceful
    SIGTERM-then-wait release() uses for a KNOWN process) - a stray is
    by definition already outside normal lifecycle management, so
    there's nothing to gracefully hand off to.
    """
    killed: list[int] = []
    for pattern in _STRAY_PROCESS_PATTERNS:
        try:
            result = subprocess.run(
                ["pgrep", "-f", pattern], capture_output=True, text=True, timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        for pid_str in result.stdout.split():
            try:
                pid = int(pid_str)
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
            except (ValueError, ProcessLookupError, PermissionError):
                pass
    if killed:
        print(f"[core.vllm_runtime] reaped {len(killed)} stray process(es) not in this "
              f"runtime's own tracking: {killed}")
    return killed


def _gpu_memory_utilization(config: GenerationConfig) -> float:
    """
    Derives vLLM's --gpu-memory-utilization fraction from the existing
    vram_headroom_gb field (same meaning it already has: VRAM
    deliberately left unused) - the confirmed test value 0.85 on this
    16GB card corresponds to vram_headroom_gb: 2.4.

    IMPORTANT, corrected 2026-08-16 after a real live sweep (see
    config/models/gemma_extract_vllm*.yaml's vision_validation_status
    comments for the full data): this fraction is NOT a ceiling on the
    vLLM process's total observed VRAM. It is only the input to vLLM's
    own KV-cache-block-count calculation (available_kv_cache = total *
    utilization - weights - profiled_activation_memory) - it does not
    bound the separate EngineCore worker process's CUDA context, CUDA-
    graph capture buffers, or PyTorch allocator fragmentation, none of
    which shrink proportionally when this value is lowered. Four
    independent levers (this fraction, --enforce-eager, --max-model-len,
    --kv-cache-dtype fp8) were each tested in isolation against the same
    checkpoint on this project's actual hardware and NONE meaningfully
    reduced total observed VRAM below a ~15.7-16.0GB floor - do not
    assume raising vram_headroom_gb further will free real memory for
    other processes without re-verifying live; it may only shrink the
    KV-cache budget (irrelevant for a workload that never approached it
    anyway) while leaving total usage roughly unchanged.
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
