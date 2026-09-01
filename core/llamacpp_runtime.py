"""
llama.cpp (llama-server) subprocess lifecycle owner - the Windows-side
sibling of core/vllm_runtime.py, built 2026-09-01 for the two-system
architecture benchmark (see the GO WITH CONDITIONS reconnaissance
report): serve GGUF deployment artifacts through llama-server and
benchmark them against the vLLM-served checkpoints of the same weights
ancestry.

DESIGN - deliberately a mirror of VllmRuntime, same four-method
interface (acquire / release / chat_completion / running_model_name +
last_load_telemetry), so everything above the runtime seam (adapter,
benchmark console) treats the two engines symmetrically:

- ONE worker slot, ONE fixed port (LLAMACPP_PORT below) - same 16GB-card
  reasoning as vllm_runtime's single-slot design.
- Subprocess (llama-server.exe), not in-process bindings: same teardown
  guarantee - killing the process provably returns its VRAM.
- Cross-runtime exclusion via core/gpu_coordinator.py, registered as
  "llamacpp". NOTE the coordinator is per-process: this runtime runs in
  the WINDOWS process (model_console / benchmark), where it arbitrates
  against the in-process transformers residency. The vLLM runtime lives
  in the WSL backend process - cross-PROCESS arbitration does not exist
  for it either; the benchmark flow releases models between legs, and
  the standing one-model-at-a-time discipline applies as ever.
- Unlike vLLM there is no --gpu-memory-utilization reservation: llama.cpp
  allocates weights + a KV cache sized by context x parallel slots and
  nothing more, which is precisely the workstation-coexistence property
  the two-system architecture is evaluating.

Binary: defaults to the Unsloth Studio prebuilt (Windows CUDA build -
see UNSLOTH_PREBUILT_INFO.json next to it for the exact tag/commit;
b10687 at time of writing). Override via GENEALOGY_LLAMACPP_SERVER for
a differently-pinned build - the build identity is a provenance
dimension (recon report section 7), so record which one a run used.

Config surface (config/models/*.yaml, runtime: llamacpp), all via
`extra` same as the vllm_* keys:
  llamacpp_model_path   - REQUIRED, absolute path to the .gguf weights
  llamacpp_mmproj_path  - vision projector .gguf (required for VLMs)
  llamacpp_ngl          - GPU layers (default 999 = fully offloaded)
  llamacpp_parallel     - server slots (default 1; this workload is
                          sequential, and per-slot KV is a static split)
  llamacpp_cache_type_k / llamacpp_cache_type_v - optional KV quant
plus the shared fields context_length (-> -c) and the standard sampling
fields translated in chat_completion().
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import Any, Optional

import requests

from core.gpu_coordinator import gpu_coordinator
from core.loaders.base_loader import GenerationConfig
from core.resource_guard import check_resources_or_raise, log_resource_trend

LLAMACPP_PORT = int(os.environ.get("GENEALOGY_LLAMACPP_PORT", "8503"))
LLAMACPP_SERVER_EXE = os.environ.get(
    "GENEALOGY_LLAMACPP_SERVER",
    os.path.expanduser("~/.unsloth/llama.cpp/build/bin/Release/llama-server.exe"),
)
# CUDA 13 runtime DLLs (cudart64_13/cublas64_13/cublasLt64_13) - the
# Unsloth prebuilt's ggml-cuda.dll needs these loadable via PATH, and
# they are NOT next to the exe; Unsloth Studio's launcher supplies them
# from its own torch install. Confirmed live 2026-09-01: without this
# on PATH the server SILENTLY falls back to CPU-only (no error, no CUDA
# line in the log, ~5.8 tok/s and zero VRAM movement on a 12B).
LLAMACPP_DLL_DIR = os.environ.get(
    "GENEALOGY_LLAMACPP_DLL_DIR",
    os.path.expanduser("~/.unsloth/studio/unsloth_studio/Lib/site-packages/torch/lib"),
)

# llama-server loads a mmap'd GGUF much faster than a vLLM engine init,
# but a 12B + mmproj cold load on this machine is unmeasured - generous
# ceiling, loud failure past it, same philosophy as vllm_runtime.
HEALTH_TIMEOUT_SECONDS = 300
TERMINATE_GRACE_SECONDS = 10


class LlamaCppRuntime:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._process: Optional[subprocess.Popen] = None
        self._model_name: Optional[str] = None
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
        return f"http://127.0.0.1:{LLAMACPP_PORT}"

    def acquire(self, model_name: str, config: GenerationConfig) -> str:
        """
        Ensures `model_name`'s llama-server is up and healthy, returns
        its base URL. Reuses a running same-model server; tears down a
        different-model one first. Claims the GPU via gpu_coordinator
        (evicting a resident transformers model in THIS process) before
        launching.
        """
        with self._lock:
            if self.running_model_name == model_name:
                return self.base_url

            model_path = config.extra.get("llamacpp_model_path")
            if not model_path:
                raise ValueError(
                    f"{model_name!r} has runtime: llamacpp but no "
                    "extra.llamacpp_model_path in its YAML - the GGUF artifact "
                    "path is required (see core/llamacpp_runtime.py's docstring)."
                )
            if not os.path.isfile(model_path):
                raise FileNotFoundError(
                    f"llamacpp_model_path for {model_name!r} does not exist: {model_path}"
                )
            if not os.path.isfile(LLAMACPP_SERVER_EXE):
                raise FileNotFoundError(
                    f"llama-server binary not found at {LLAMACPP_SERVER_EXE} - "
                    "set GENEALOGY_LLAMACPP_SERVER to a llama-server(.exe) build."
                )

            if self._process is not None:
                self.release()
            # Reconciliation sweep - anything else already bound to our
            # port is a stray from a previous crashed session. Port-scoped
            # on purpose: killing by image name (llama-server.exe) would
            # also take down Unsloth Studio's own server.
            _kill_port_owners()

            check_resources_or_raise(context=f"launching llama-server for {model_name!r}")
            gpu_coordinator.claim("llamacpp")

            cmd = [
                LLAMACPP_SERVER_EXE,
                "-m", str(model_path),
                "--port", str(LLAMACPP_PORT),
                "--host", "127.0.0.1",
                "-c", str(config.context_length or 4096),
                "-ngl", str(config.extra.get("llamacpp_ngl", 999)),
                "--parallel", str(config.extra.get("llamacpp_parallel", 1)),
                # GGUF-embedded jinja chat template - same "the server owns
                # the template" split the vLLM path already established.
                "--jinja",
                "--no-webui",
            ]
            # Thought-tag handling (default: the server's own "auto"
            # parser, which strips thought channels out of content -
            # correct once thinking is suppressed below). Opt-in override
            # via extra for a model whose template mis-parses: "none"
            # leaves tags unparsed in content verbatim. Confirmed live
            # 2026-09-01 on the Gemma-4 QAT GGUF: format none + thinking
            # on floods content with the raw thought channel; format auto
            # + thinking on buries the whole answer in reasoning_content
            # (chat_completion()'s fallback covers that); the production
            # combination is reasoning off + default auto format.
            reasoning_format = config.extra.get("llamacpp_reasoning_format")
            if reasoning_format:
                cmd.extend(["--reasoning-format", str(reasoning_format)])
            # Template-level thinking toggle (per-model opt-in via extra,
            # same pattern as the vllm_* keys). gemma_12b_qat_gguf sets
            # "off": the QAT GGUF's template defaults Gemma-4 thinking ON,
            # which buries the answer in a thought channel and burns the
            # token budget - the vLLM leg of the A/B answers directly, so
            # parity requires suppressing it here (confirmed live
            # 2026-09-01: with thinking on, a 400-token budget ran out
            # mid-thought and produced no answer at all).
            reasoning = config.extra.get("llamacpp_reasoning")
            if reasoning:
                cmd.extend(["--reasoning", str(reasoning)])
            mmproj = config.extra.get("llamacpp_mmproj_path")
            if mmproj:
                if not os.path.isfile(mmproj):
                    raise FileNotFoundError(
                        f"llamacpp_mmproj_path for {model_name!r} does not exist: {mmproj}"
                    )
                cmd.extend(["--mmproj", str(mmproj)])
            for flag, key in (("--cache-type-k", "llamacpp_cache_type_k"),
                              ("--cache-type-v", "llamacpp_cache_type_v")):
                value = config.extra.get(key)
                if value:
                    cmd.extend([flag, str(value)])

            log_path = os.path.join(
                os.environ.get("TEMP", os.path.expanduser("~")),
                f"llamacpp_{model_name}.log",
            )
            print(f"[core.llamacpp_runtime] launching llama-server: {model_path!r} "
                  f"on port {LLAMACPP_PORT} (log: {log_path})")
            env = dict(os.environ)
            if os.path.isdir(LLAMACPP_DLL_DIR):
                env["PATH"] = LLAMACPP_DLL_DIR + os.pathsep + env.get("PATH", "")
            baseline_vram_mb = _nvidia_smi_used_mb()
            load_start = time.time()
            self._process = subprocess.Popen(
                cmd, env=env,
                stdout=open(log_path, "w"),
                stderr=subprocess.STDOUT,
            )
            self._model_name = model_name

            try:
                self._wait_for_health(log_path)
            except Exception:
                self.release()
                raise
            load_time_s = round(time.time() - load_start, 3)
            after_load_vram_mb = _nvidia_smi_used_mb()
            self._last_load_telemetry = {
                "model_name": model_name,
                "baseline_vram_mb": baseline_vram_mb,
                "after_load_vram_mb": after_load_vram_mb,
                "load_time_s": load_time_s,
                "startup_includes": "subprocess_spawn+gguf_mmap_load (not separable)",
            }
            print(f"[core.llamacpp_runtime] {model_name!r} healthy at {self.base_url} ({load_time_s}s)")
            return self.base_url

    def _wait_for_health(self, log_path: str) -> None:
        deadline = time.time() + HEALTH_TIMEOUT_SECONDS
        while time.time() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"llama-server process exited during startup (code "
                    f"{self._process.returncode}) - see {log_path}"
                )
            try:
                resp = requests.get(f"{self.base_url}/health", timeout=3)
                if resp.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
        raise TimeoutError(
            f"llama-server for {self._model_name!r} not healthy after "
            f"{HEALTH_TIMEOUT_SECONDS}s - see {log_path}"
        )

    def release(self) -> None:
        """
        terminate -> grace wait -> kill, then a port-scoped stray sweep
        and an nvidia-smi report. Safe no-op when nothing is running.
        Windows has no process groups in the POSIX sense - llama-server
        spawns no child workers (unlike vLLM's EngineCore), so
        terminating the one tracked process is the whole teardown.
        """
        with self._lock:
            if self._process is None:
                _kill_port_owners()
                log_resource_trend("llamacpp_runtime")
                return
            model_name = self._model_name
            print(f"[core.llamacpp_runtime] release: {model_name!r}")
            try:
                self._process.terminate()
                try:
                    self._process.wait(timeout=TERMINATE_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    print(f"[core.llamacpp_runtime] {model_name!r} ignored terminate, killing")
                    self._process.kill()
                    self._process.wait(timeout=10)
            except OSError:
                pass  # already gone
            self._process = None
            self._model_name = None
            gpu_coordinator.release_noted("llamacpp")

            used_mb = _nvidia_smi_used_mb()
            if used_mb is not None:
                print(f"[core.llamacpp_runtime] post-release GPU memory.used: {used_mb} MiB")

            _kill_port_owners()
            log_resource_trend("llamacpp_runtime")

    def chat_completion(self, config: GenerationConfig, messages: list[dict[str, Any]],
                        max_tokens: int, temperature: float) -> dict[str, Any]:
        """
        One /v1/chat/completions call against the running server - same
        return shape as VllmRuntime.chat_completion(): {"text", "usage",
        "vram_used_mb", "generation_time_s"}.

        Sampling-parameter translation (recorded here, mirrored per-run
        in the adapter's settings_translation_notes): llama-server
        accepts top_k natively; its repetition-penalty field is named
        `repeat_penalty` (llama.cpp's own name), NOT vLLM's
        `repetition_penalty` - translated below. no_repeat_ngram_size
        has no llama.cpp equivalent either (same gap as vLLM) - NOT
        sent, NOT applied.
        """
        with self._lock:
            if self.running_model_name is None:
                raise RuntimeError(
                    "llamacpp_runtime.chat_completion() called with no server "
                    "running - call acquire() first."
                )
            base_url = self.base_url

        payload: dict[str, Any] = {
            "model": config.repo_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if config.top_p is not None:
            payload["top_p"] = config.top_p
        if config.top_k and config.top_k > 0:
            payload["top_k"] = config.top_k
        if config.repetition_penalty and config.repetition_penalty != 1.0:
            payload["repeat_penalty"] = config.repetition_penalty
        if config.stop_string:
            payload["stop"] = [config.stop_string]

        start = time.time()
        resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=300)
        resp.raise_for_status()
        generation_time_s = time.time() - start
        data = resp.json()
        message = data["choices"][0]["message"]
        # --reasoning-format none (in acquire()'s argv) should make
        # content carry everything; the reasoning_content fallback is a
        # defensive belt for a template/server combination that splits
        # anyway - better the raw text than a silently empty answer.
        text = message.get("content") or message.get("reasoning_content") or ""
        return {
            "text": text,
            "usage": data.get("usage"),
            "vram_used_mb": _nvidia_smi_used_mb(),
            "generation_time_s": generation_time_s,
        }


def _port_owner_pids() -> list[int]:
    """PIDs of processes LISTENING on LLAMACPP_PORT, via `netstat -ano`
    (present on every Windows install - no psutil dependency)."""
    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    pids = set()
    needle = f":{LLAMACPP_PORT}"
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] == "TCP" and parts[1].endswith(needle) \
                and parts[3] == "LISTENING":
            try:
                pids.add(int(parts[4]))
            except ValueError:
                pass
    return sorted(pids)


def _kill_port_owners() -> list[int]:
    """Force-kills any process bound to our port that this runtime's own
    bookkeeping doesn't know about - port-scoped, NOT image-name-scoped
    (see acquire()'s comment: Unsloth Studio runs its own llama-server)."""
    killed = []
    for pid in _port_owner_pids():
        try:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                           capture_output=True, timeout=10)
            killed.append(pid)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    if killed:
        print(f"[core.llamacpp_runtime] reaped {len(killed)} stray process(es) "
              f"on port {LLAMACPP_PORT}: {killed}")
    return killed


def _nvidia_smi_used_mb() -> Optional[float]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        return float(result.stdout.strip().splitlines()[0])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


# The one shared instance. Do not construct LlamaCppRuntime() anywhere else.
llamacpp_runtime = LlamaCppRuntime()

# Cross-runtime registration - lets gpu_coordinator evict this subprocess
# when the (same-process) transformers residency claims the GPU.
gpu_coordinator.register_runtime("llamacpp", llamacpp_runtime.release)
