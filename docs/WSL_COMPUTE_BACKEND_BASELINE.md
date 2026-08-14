# WSL Compute Backend — Known-Good Baseline (2026-08-13)

**Status: FROZEN.** This document records the exact, verified-working WSL2
compute environment as of 2026-08-13, confirmed end-to-end with a real
quantized VLM generating correctly on this machine's RTX 5060 Ti. Treat
this as the reference state - do not modify the WSL environment casually;
if something needs to change, update this doc in the same change.

## Why this exists

The Windows-native stack hit a real, structural wall trying to run
quantized VLMs (Gemma-4 E4B QAT-mobile, compressed-tensors w4a16,
AWQ/GPTQ via `gptqmodel`) - see `project_hybrid_cpu_planner_architecture`
and the E4B/vLLM audit memory notes for the full investigation. Every
Windows-native "free swap" attempt converged on the same class of gap:
`triton` has no official Windows wheel, `gptqmodel` needs a C build
toolchain (`pypcre` → CMake/Autoconf) that isn't present, vLLM has no
official Windows support at all. Rather than keep chasing Windows-specific
workarounds, this session moved the compute backend to WSL2 Ubuntu, kept
Windows as the client/UI, and confirmed the whole chain works.

## Environment

| Component | Value |
|---|---|
| WSL version | 2.7.11.0 |
| WSL kernel | 6.18.33.2-2 |
| Distro | Ubuntu 24.04.4 LTS (Noble Numbat), installed via `wsl --install -d Ubuntu-24.04` |
| Default WSL user | `genealogy` (member of `sudo` group; set as default via `/etc/wsl.conf`'s `[user] default=genealogy`) |
| Python (WSL, system) | 3.12.3 |
| GPU passthrough | Via the Windows NVIDIA driver directly (`/usr/lib/wsl/lib/`) - **no Linux NVIDIA driver installed inside WSL**, that would break passthrough |
| `nvidia-smi` inside WSL | Confirmed working: RTX 5060 Ti, 16311MiB, CUDA UMD 13.3 |
| CUDA Toolkit (WSL) | 13.0.3, installed via NVIDIA's official WSL-Ubuntu apt repo (`cuda-keyring_1.1-1_all.deb` → `apt-get install cuda-toolkit-13-0`) - needed for `nvcc`/JIT-compiling kernel extensions (e.g. GPTQModel's Marlin kernel), NOT for basic PyTorch CUDA ops (those work via the passthrough driver alone) |
| `CUDA_HOME` | `/usr/local/cuda` - must be set explicitly (and `PATH` must include `/usr/local/cuda/bin`) before running anything that JIT-compiles a CUDA extension; not set by default even after installing the toolkit |
| Build toolchain | `build-essential` (gcc/g++/make), `cmake`, `autoconf` - all via one-line `apt-get install`, no friction (this is the exact toolchain Windows lacked for `gptqmodel`'s `pypcre` dependency) |

## Python environment

Venv at `~/venv_test` (on the `genealogy` user, inside WSL - **not** the
Windows filesystem). Created via `python3 -m venv ~/venv_test`. Full
`pip freeze` as of this baseline:

```
accelerate==1.14.0
aiohappyeyeballs==2.7.1
aiohttp==3.14.3
aiosignal==1.4.0
annotated-doc==0.0.5
anyio==4.14.2
attrs==26.1.0
certifi==2026.7.22
charset-normalizer==3.5.0
click==8.4.2
cuda-bindings==13.0.3
cuda-pathfinder==1.2.2
cuda-toolkit==13.0.3.0
datasets==5.0.1
Defuser==0.0.25
Device-SMI==0.5.6
dill==0.4.1
filelock==3.29.0
frozenlist==1.8.0
fsspec==2026.4.0
GPTQModel==7.3.2
h11==0.16.0
hf-xet==1.6.0
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.27.0
idna==3.18
Jinja2==3.1.6
LogBar==0.4.12
markdown-it-py==4.2.0
MarkupSafe==3.0.3
maturin==1.14.1
mdurl==0.1.2
mpmath==1.3.0
multidict==6.7.1
multiprocess==0.70.19
networkx==3.6.1
ninja==1.13.0
numpy==2.2.6
nvidia-cublas==13.1.1.3
nvidia-cuda-cupti==13.0.85
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-runtime==13.0.96
nvidia-cudnn-cu13==9.20.0.48
nvidia-cufft==12.0.0.61
nvidia-cufile==1.15.1.6
nvidia-curand==10.4.0.35
nvidia-cusolver==12.0.4.66
nvidia-cusparse==12.6.3.3
nvidia-cusparselt-cu13==0.8.1
nvidia-nccl-cu13==2.29.7
nvidia-nvjitlink==13.2.78
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.0.85
optimum==2.3.0
packaging==26.3
pandas==3.0.5
pillow==12.3.0
propcache==0.5.2
protobuf==7.35.1
psutil==7.2.2
pyarrow==25.0.1
Pygments==2.20.0
PyPcre==0.6.0
python-dateutil==2.9.0.post0
PyYAML==6.0.3
regex==2026.7.19
requests==2.34.2
rich==15.0.0
safetensors==0.8.0
setuptools==78.1.0
shellingham==1.5.4
six==1.17.0
sympy==1.14.0
threadpoolctl==3.6.0
TokeNicer==0.0.14
tokenizers==0.22.2
torch==2.13.0+cu130
torchao==0.18.0
torchvision==0.28.0+cu130
tqdm==4.70.0
transformers==5.15.0
triton==3.7.1
typer==0.27.1
typing_extensions==4.15.0
urllib3==2.7.0
xxhash==4.0.0
yarl==1.24.5
```

**Note the transformers version**: 5.15.0 here (pulled in fresh by
`gptqmodel`/`optimum` as dependencies), vs 5.12.1 on the Windows-native
side. These are two independent environments by design - the Windows
side stays on whatever this project's loaders are tested against; this
WSL side tracks whatever the Linux quant stack actually needs. Do not
assume version parity between the two.

Install commands used, in order (all via `apt-get`/`pip`, no source
workarounds needed beyond one config fix - see below):
```bash
# System packages (as root)
apt-get update
apt-get install -y python3-venv python3-pip git curl build-essential cmake autoconf
wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update
apt-get install -y cuda-toolkit-13-0

# Python env (as genealogy user)
python3 -m venv ~/venv_test
source ~/venv_test/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install gptqmodel optimum
```

## Shared filesystem with Windows

**Updated 2026-08-13 (later same day)**: the HF cache moved from
`C:\Users\jonny\.cache\huggingface` to `E:\huggingface` (222GB, disk-
space pressure on C:) - see `feedback_robocopy_symlink_dereference.md`
memory note for a real gotcha hit during the move. Windows-side
`HF_HOME` is now a persistent user env var pointing at `E:\huggingface`.

`HF_HOME` for WSL-side Python processes must point at
**`/mnt/e/huggingface`** (not `/mnt/c/...` - that path is gone). Reuses
every checkpoint already downloaded on the Windows side (confirmed: no
re-download needed for a checkpoint already cached). This is the same
principle the portability audit identified for `workspace_root` - point
both environments at the same physical drive and artifact bytes never
need to cross a real transfer boundary, only a path reference does.

**If you moved the cache again after this doc was written, update this
section AND every WSL test script that hardcodes the old path.**

## Confirmed working end-to-end (2026-08-13)

Real bf16 matmul on GPU inside WSL:
```
torch: 2.13.0+cu130 | cuda build: 13.0 | compute capability: (12, 0)
arch list includes sm_120
matmul OK, VRAM allocated: 58.7 MB
```

Real quantized VLM, `openbmb/MiniCPM-V-4_5-GPTQ` (genuine GPTQ,
`checkpoint_format: "gptq"`, not compressed-tensors), full load + real
generation:
```
Kernel: selected -> MarlinLinear
LOADED OK in 33.6s
VRAM after load: 8342.9 MB
GENERATION RESULT: 'PONG'  (correct, for prompt "Reply with exactly one word: PONG")
generation time (Marlin kernel already JIT-cached): 6.51s
peak VRAM during generation: 8394.5 MB
```

First run only: Marlin's bf16 kernel needs a one-time JIT compile
(~63s), cached afterward at
`~/.cache/gptqmodel/torch_extensions/marlin_fp16/<hash>/`. Every
subsequent load reuses the cached `.so`, no recompile.

## Required fix, not a workaround: MiniCPM-V's block-pattern override

`openbmb/MiniCPM-V-4_5-GPTQ` failed to load with GPTQModel's default
auto-detection: `"Block pattern could not be match. Pass
block_name_to_quantize argument"`. Root cause: MiniCPM-V's custom
`trust_remote_code` wrapper nests the real transformer backbone at
`self.llm = Qwen3ForCausalLM(config)` - a non-standard location
GPTQModel's naming heuristic doesn't recognize.

**Passing `quantization_config=GPTQConfig(..., block_name_to_quantize=...)`
to `from_pretrained()` does NOT work** - transformers' own quantizer
silently ignores everything except `backend`/`max_input_length` from a
passed-in config when the checkpoint already embeds one (confirmed via
the actual warning: *"loading attributes (e.g. ['backend',
'max_input_length']) will be overwritten... The rest will be ignored"*).

**The fix that worked**: edit the checkpoint's own cached `config.json`
directly, adding `"block_name_to_quantize": "llm.model.layers"` inside
its `quantization_config` block. This is a real HuggingFace cache blob
file, not a symlink target you can edit directly - resolve the real path
first (`os.path.realpath()` on the snapshot's `config.json` symlink,
which points into `.../blobs/<hash>`) and edit the blob.

**This means: if this checkpoint's cache is ever cleared or
re-downloaded, this fix must be reapplied** - it lives in the HF cache,
not in this repo. If this checkpoint becomes a real production
candidate, consider baking the fix into a proper loader-level override
instead of relying on a hand-edited cache file (e.g. a small pre-load
hook that patches the loaded config's `quantization_config` object
in-memory before `from_pretrained` resolves the block pattern).

## vLLM venv (2026-08-13, later same day) — the actual intended runtime for w4a16-ct

Separate venv `~/venv_vllm` (kept isolated from `~/venv_test` deliberately -
vLLM pins its own dependency versions, don't risk the frozen GPTQModel
baseline). Install: `pip install vllm` (pulled vLLM 0.27.1, its own
pinned `torch==2.13.0+cu130`, `transformers==5.15.0`).

**`google/gemma-4-12B-it-qat-w4a16-ct` under plain transformers is a
confirmed dead end (see above) - under its ACTUALLY intended runtime,
vLLM, it works correctly and efficiently.** Three real, distinct issues
had to be resolved in sequence to get there - all genuine bugs/gaps,
not workarounds papering over a fundamentally broken path:

1. **`AmbiguousGlobalPerLayerAttributeError`** on `head_dim` - vLLM's
   Gemma4Unified code reads `config.head_dim` globally, but Gemma-4's
   architecture genuinely has heterogeneous per-layer head_dim (256 for
   sliding-attention layers, 512 for the periodic full-attention
   layers) and transformers 5.15.0 added a guard against silently
   reading the wrong one. **First attempt** (editing the cached
   `config.json` to add `"allow_global_per_layer_attribute_access":
   true`) was the WRONG fix - it silenced the guard but then vLLM
   built every layer with head_dim=256 and crashed loading a
   512-sized weight into a 256-sized slot
   (`AssertionError: Attempted to load weight (torch.Size([512])) into
   parameter (torch.Size([256]))`). **Correct fix, found via the actual
   upstream bug report** (vllm-project/vllm#51744): pin
   `transformers==5.14.1` in the `venv_vllm` venv specifically (5.14.1
   predates the stricter guard; vLLM-side fixes for 5.15.0 exist as
   PRs #49797/#49959 but hadn't reached the released package as of
   vLLM 0.27.1). **The `config.json` edit was reverted** - it's not
   needed and was actively wrong.
2. **Multiprocessing bootstrap error**: WSL requires vLLM's `spawn`
   start method (not `fork` - "NVML is not compatible with fork" under
   WSL, vLLM detects this automatically), which requires the calling
   script's own code to be wrapped in `if __name__ == "__main__":` -
   a standard Python multiprocessing requirement, not vLLM/WSL-specific.
3. **`RuntimeError: UVA is not available`** - a confirmed, documented
   WSL2-specific vLLM issue (CUDA Unified Virtual Addressing not
   detected correctly through the WSL2 GPU passthrough layer for
   vLLM's newest V1 engine memory buffers). Fix: environment variable
   `VLLM_WSL2_ENABLE_PIN_MEMORY=1`.

**Confirmed working, full numbers**:
```
Checkpoint size: 9.56 GiB (on disk)
Model loading: 8.28 GiB resident, 54.4s
Available KV cache: 3.77 GiB (11,717 tokens)
CUDAGraph capture: 0.57 GiB, 20s
Total engine init (profile + KV cache + warmup): 173.5s one-time cost
Generation: 'PONG' (correct, for "Reply with exactly one word: PONG")
Generation time (after warmup): 1.34s
```

Real, compact 8.28GB weight footprint - genuine 4-bit compression
realized, a completely different result from the ~25GB decompress-on-
first-use behavior confirmed under plain transformers. This is the
clean confirmation the whole E4B/w4a16 investigation was aimed at:
the checkpoint was never a dead end, it just needed its actual
intended runtime.

**E4B confirmed too, same session, no new fixes needed**:
`google/gemma-4-E4B-it-qat-w4a16-ct` (11.55GB download, the standard
compressed-tensors format - not the mobile-transformers checkpoint the
original investigation started with) loaded and generated correctly
under the exact same `venv_vllm` setup (transformers 5.14.1 pin, `/mnt/e`
HF_HOME, `VLLM_WSL2_ENABLE_PIN_MEMORY=1`, `__main__` guard) - all three
fixes from the 12B test carried over cleanly, nothing E4B-specific
needed:
```
Checkpoint size: 10.72 GiB (on disk)
Model loading: 9.93 GiB resident, 68.1s
Available KV cache: 2.29 GiB (42,692 tokens)
CUDAGraph capture: 0.49 GiB, 13s
Total engine init: 118.75s (compilation: 60.9s) - faster than 12B's
  173.5s, as expected for the smaller model
Generation: 'PONG' (correct), 1.18s after warmup
```
This is the actual resolution to the ORIGINAL question this whole
investigation started from: the E4B checkpoint is not, and was never,
architecturally deficient relative to E2B - the "4.8x slower, more
VRAM" Benchmark 2.3 numbers were entirely a runtime-format artifact
(mobile-QAT checkpoint under plain transformers), not a real E2B vs E4B
capability comparison. Under its correct format + runtime, E4B loads
in a compact ~10GB and generates in ~1.2s, in the same neighborhood as
E2B's own numbers, not 4.8x worse.

**Third checkpoint confirmed, same session, still no new fixes needed**:
`Qwen/Qwen2.5-VL-7B-Instruct-AWQ` - the exact checkpoint that hit the
`out_features=3420` Marlin tiling error under plain
transformers/GPTQModel earlier in this investigation. Under vLLM's own
kernel dispatch, that shape issue doesn't occur at all - loaded and
generated correctly, same `venv_vllm` setup, zero new fixes:
```
Checkpoint size: 6.45 GiB (on disk)
Model loading: 6.67 GiB resident, 59.9s
Available KV cache: 4.24 GiB
CUDAGraph capture: 0.40 GiB, 95s (longer than the two Gemma models -
  likely more capture configurations for the multimodal/vision path)
Total engine init: 181.1s (compilation: 19.0s)
Generation: 'PONG' (correct), 0.26s after warmup - fastest of the three
  checkpoints tested, consistent with being the smallest (7B)
```
Confirms the pattern holds generally, not just for the two Gemma
checkpoints: a genuine AWQ/GPTQ/compressed-tensors checkpoint that
fails or performs badly under plain transformers on this machine is
very likely to just work under vLLM once the three WSL2/transformers-
version fixes above are in place - the earlier shape-specific kernel
failures were symptoms of the wrong runtime, not the checkpoint itself.

**venv_vllm's exact versions** (differs from `venv_test` deliberately -
do not try to unify them): `vllm==0.27.1`, `transformers==5.14.1`
(downgraded from the 5.15.0 vLLM installed by default),
`torch==2.13.0+cu130` (vLLM's own pin, matches venv_test's).

**Script requirement**: any script calling `vllm.LLM(...)` under WSL
must guard its own top-level code with `if __name__ == "__main__":` -
vLLM's `spawn` multiprocessing start method re-imports the main module
in the child process.

## Known non-blocking finding, logged for later

`Qwen/Qwen2.5-VL-7B-Instruct-AWQ` (genuine AWQ, `modules_to_not_convert:
["visual"]`, native `Qwen2_5_VLForConditionalGeneration` class) got
through the same install/kernel-selection path (GPTQModel correctly
selected `AwqMarlinLinear`) but failed on a specific layer:
`out_features=3420` isn't divisible by 64, a Marlin kernel tiling
requirement. Not attempted further this session - a real fix would need
either a different kernel candidate for that layer or accepting a
non-Marlin fallback for it specifically. Not a platform issue, same
category as the MiniCPM-V fix above, just not yet solved.

## `~/venv_backend` - the production agent/chat backend venv

Built 2026-08-13, a THIRD isolated WSL venv alongside `venv_test`
(GPTQModel scratch/benchmark environment) and `venv_vllm` (vLLM
scratch/benchmark environment) - this one is not a benchmark
environment, it's the actual runtime for `api/agent_main.py`, the
FastAPI service the Windows-side `model_console` client is meant to
call over HTTP per the portability plan's "minimum viable migration"
(wrap `run_agent_chat_turn()`/`ChatBackendAdapter` as one HTTP-
reachable service, Windows stays client/UI-only).

Deliberately pinned to match the WINDOWS-side `pip freeze` exactly
(not `venv_test`'s or `venv_vllm`'s own pins) - this venv runs the
project's real `core/`/`model_console/` code unmodified against the
same dependency versions that code is already proven correct under on
Windows, so behavior stays consistent across the client/backend split
rather than introducing a second set of version-drift unknowns:

```
torch==2.13.0+cu130 / torchvision==0.28.0+cu130   (--index-url https://download.pytorch.org/whl/cu130)
transformers==5.12.1
accelerate==1.14.0
pydantic==2.13.4
PyYAML==6.0.3
pillow==12.3.0
huggingface_hub==1.20.1
requests==2.34.2
beautifulsoup4==4.15.0
fastapi==0.138.0
uvicorn==0.49.0
psutil==7.2.2
qwen-vl-utils==0.0.14
bitsandbytes==0.49.2
numpy==2.4.6
opencv-python-headless==5.0.0.93
scipy==1.18.0
timm==1.0.28
```

`HF_HOME` is NOT set separately for this venv - it inherits whatever
the calling shell/service sets (should point at `/mnt/e/huggingface`,
same shared cache as `venv_test`/`venv_vllm`, so no duplicate
downloads - see the HF cache section above).

**Confirmed live (2026-08-13):**
1. `core/schema.py`, `core/document_templates.py`, `core/loader_registry.py`,
   `core/model_residency.py`, `core/loaders/gemma_loader.py`,
   `core/loaders/base_loader.py` all import cleanly against the real
   repo at `/mnt/j/Genealogy/genealogy_pipeline` (no reimplemented
   scratchpad scripts - the actual project code, running from WSL).
2. All of `core/agent_tools/` (registry, dispatcher, tools, planner,
   research_llm, web_research_agent) and the non-GUI half of
   `model_console/` (session, session_log, adapter, agent_bridge,
   conversation_manager, image_prep) import cleanly too.
   `tkinter` is correctly ABSENT from this venv/WSL - confirms none of
   the backend-logic modules accidentally depend on the GUI layer
   (`chat_tab.py`/`app.py` were never imported, by design - see
   `api/agent_main.py`'s own docstring on the import boundary).
3. `api/agent_main.py` (new FastAPI app, separate from `api/main.py`'s
   Phase 1 read-only pipeline-monitoring API - a different concern)
   starts under `uvicorn api.agent_main:app` and serves real requests:
   `/model/status`, `/network/status`, `/network/toggle` all round-
   tripped correctly over `curl`, and a REAL end-to-end chat turn
   (`POST /chat/turn`, `model_name="gemma_extract"`, `use_agent=false`,
   "Reply with exactly one word: PONG") loaded Gemma-4-E2B (10.2GB
   VRAM, 97s cold load) and generated "Pong" correctly in 0.16s
   generation time after load - the full real pipeline (FastAPI ->
   `ChatBackendAdapter` -> `core.model_residency` -> `GemmaLoader`)
   works end-to-end from a single HTTP call, in this venv, against the
   real repo.

**UPDATE (2026-08-13, later same day): ChatBackendAdapter is now a
switchable HTTP client.** Per Jon's explicit direction ("don't redesign
chat_tab.py around HTTP - make the adapter absorb the transport
change... keep the existing in-process implementation as a selectable/
fallback backend until the HTTP route has passed both plain-chat and
use_agent=true regression tests"):

- `model_console/adapter.py`'s `ChatBackendAdapter` now takes a
  `backend: "local" | "remote"` constructor arg (or `GENEALOGY_CHAT_
  BACKEND` env var), defaulting to `"local"` - the pre-existing,
  unchanged, still-frozen Windows-native GPU path. `backend="remote"`
  makes every method (`ensure_loaded`, `send_turn`, `release`) an HTTP
  call to this venv_backend service instead
  (`GENEALOGY_CHAT_BACKEND_URL`, default `http://localhost:8001` -
  WSL2 auto-forwards this to Windows localhost, no static WSL VM IP or
  `.wslconfig` mirrored-networking needed, confirmed by direct test).
- `chat_tab.py` needed **zero changes** - same public method names,
  signatures, and return shapes on both backends.
- Agent mode is the one place that couldn't stay a thin per-call proxy:
  `core.agent_tools.planner.run_agent_turn()`'s loop (cpu_planner ->
  dispatcher -> tools, all in-process) can't run half on Windows and
  half on WSL. `model_console/agent_bridge.py::run_agent_chat_turn()` -
  the one file allowed to know about both `adapter.backend` and
  `core.agent_tools.planner`'s dataclasses - branches: local backend
  keeps running the loop in-process exactly as before; remote backend
  sends the WHOLE turn to WSL in one `POST /chat/turn` (`use_agent=
  true`) via a new `adapter.send_agent_turn()`, then reconstructs real
  `AgentTurnResult`/`AgentStep`/`PlanStep`/`ToolResultBlock` dataclass
  instances from the JSON (`dataclasses.asdict()` on the server side
  means field names match exactly) so `chat_tab.py`'s attribute-access
  rendering code (`step.plan.action`, `step.tool_result.success`) never
  knows the difference.
- `api/agent_main.py`'s `HistoryTurn` now carries `turn_id` (not just
  role/text) so remote-backend `history_turn_ids`/`dropped_turn_ids`
  context provenance refers to the SAME ids the calling `ChatSession`
  already tracks, not server-regenerated ones.

**Both acceptance tests Jon asked for passed live** (Windows client,
`C:\Python314\python.exe`, against the running `venv_backend` server):
1. Plain chat (`send_turn`, `use_agent=false`): `model_name=
   "gemma_extract"`, prompt "Reply with exactly one word: PONG" ->
   `"Pong"`, 3.6s.
2. Full agent turn (`run_agent_chat_turn`, `use_agent=true`): "What is
   2 + 2?" -> planner chose `answer_directly` (no tool needed) ->
   `final_answer="4"`, reconstructed into a real `AgentTurnResult`
   with one correctly-typed `AgentStep`/`PlanStep` - confirming
   planner -> dispatcher -> response survives the HTTP boundary intact,
   not just the model-call plumbing.

**Still not done / deliberately deferred**: the old Windows GPU path is
NOT retired - `backend` still defaults to `"local"`, per Jon's explicit
instruction to retire it deliberately later, not as a side effect of
this migration. A live agent turn that actually DISPATCHES a GPU tool
(`extract_fields`/`classify_document` - the "tools/model borrowing"
half of Jon's acceptance criteria) hasn't been exercised yet - today's
test used a no-tool-needed question, so the tool-dispatch code path
inside `run_agent_turn` ran (planner correctly chose not to call one)
but no actual tool call crossed the boundary. The cross-boundary
`pipeline_db`/`genealogy_memory` DB access question (direct shared-file
access vs. API-mediated) flagged in the portability audit is also still
open.

## Real bug found + fixed: VRAM not released after `/model/release`

**Found 2026-08-13** when Jon checked `nvidia-smi` after the acceptance
tests above and VRAM was still at 11.5GB despite `/model/release`
having returned 200 OK and `core.model_residency`'s own log line
confirming `"release: 'gemma_extract'"` ran. Confirmed real (not a
stale reading) by killing the WSL uvicorn process outright: VRAM
dropped to the 1.6GB idle baseline immediately, proving the memory was
genuinely held by that process, not some other Windows GPU consumer.

**Root cause, confirmed via a `GET /debug/gpu` diagnostic endpoint**
(added to `api/agent_main.py`, reads `torch.cuda.memory_allocated()` /
`memory_reserved()` directly): after release,
`memory_allocated()` correctly dropped to near-zero (33.6MB - see
below) but `memory_reserved()` stayed pinned at ~10.2GB, matching
nvidia-smi exactly. This means the real tensors WERE freed
(`core/model_residency.py`'s `_release_current()` - `loader.model =
None` + `gc.collect()` + `torch.cuda.empty_cache()` - is correct and
unchanged) but PyTorch's default (non-expandable) CUDA caching
allocator can only return an entire memory SEGMENT to the driver once
every byte in it is free. A small (~33.6MB) allocation that persists
for the process's whole lifetime by design - almost certainly a
cuBLAS/cuDNN workspace buffer, not anything `core/model_residency.py`
owns or could clear - happened to share the same large ~10GB segment
as the model's weights, pinning the WHOLE segment even after every
real model tensor was gone. Confirmed this isn't a timing/ordering
issue: a second, more forceful clear (`gc.collect()` +
`torch.cuda.synchronize()` + `torch.cuda.empty_cache()` +
`torch.cuda.ipc_collect()`, exposed as `POST /debug/empty_cache`) made
zero difference - the segment truly cannot be partially returned under
the default allocator.

**Fix**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, PyTorch's
own documented allocator mode for exactly this fragmentation pattern -
segments become elastically resizable instead of all-or-nothing. Set
in `api/agent_main.py` itself (`os.environ.setdefault(...)`, at the
very top of the file, BEFORE `torch` is imported anywhere in the
process via `model_console.adapter`) rather than relying on a launch-
command flag someone has to remember - CUDA context/allocator behavior
is only fixed at first actual GPU allocation, not at `import torch`
time, so setting it this early is sufficient and safe.

**Confirmed fixed, live, with the in-code fix alone (no external env
var)**: same load -> `send_turn` -> `run_agent_chat_turn` -> `release`
cycle as the acceptance tests - before the fix, `reserved_mb` stayed at
10206.8MB / nvidia-smi at 11520MiB after release; after, `reserved_mb`
drops to 41.9MB / nvidia-smi to 1821MiB (idle baseline), reproduced
twice.

**This is a general PyTorch/CUDA-allocator behavior, not specific to
this HTTP backend** - `core/model_residency.py`'s release logic itself
needed no changes and is not implicated. It was only newly VISIBLE
here because the acceptance tests exercised a load-generate-release
cycle in a monitored window; a typical Windows `chat_tab.py` session
holds one model resident for a whole GUI session and rarely watches
VRAM immediately after an Eject-button release, so this same
fragmentation may well affect the Windows-native path too and has
simply gone unnoticed - worth setting the same env var Windows-side if
this is ever seen there (not yet done, not yet confirmed necessary
there).

`GET /debug/gpu` and `POST /debug/empty_cache` are kept as permanent
diagnostic endpoints on `api/agent_main.py`, not stripped after this
investigation - genuinely useful for any future "is VRAM actually
free" question on this service.

**UPDATE (2026-08-13, same day): applied to the Windows side too, but
with a DIFFERENT value - `expandable_segments:True` does not work on
Windows.** Jon asked for the same fix on `model_console/app.py` (the
Windows-native entry point, the only other client of `core.model_
residency.residency`). Testing there surfaced a real platform
difference, not caught by just copying the WSL setting: PyTorch prints
`"expandable_segments not supported on this platform"` on Windows and
silently no-ops - confirmed live, `reserved_mb` stayed pinned at
~10.2GB after release exactly as before the "fix," because it never
took effect. The value that DOES work on Windows, confirmed live the
same way (`gemma_extract` load -> `send_turn` -> `release`, `torch.
cuda.memory_reserved()` checked directly): `PYTORCH_CUDA_ALLOC_CONF=
backend:cudaMallocAsync` - a different PyTorch allocator mode
(delegates to the CUDA driver's own stream-ordered allocator instead
of PyTorch's segment-based caching allocator) that IS supported on
Windows. Reserved memory went from 10234.1MB (after load) to 67.1MB
(after release) with this setting, vs. staying pinned near 10.2GB
either with no fix or with the (silently ineffective) `expandable_
segments` setting on this platform. Set in `model_console/app.py` via
`os.environ.setdefault(...)`, same placement discipline as the WSL
fix (before any import in the process pulls in `torch`). Generation
itself was unaffected (`"Pong"` came back correctly) - this is purely
an allocator-mode change, not a functional one.

**Takeaway for any future cross-platform env-var fix in this
project**: verify platform-specific allocator/runtime settings on
BOTH platforms independently rather than assuming a Linux-confirmed
fix carries over - the failure mode here (silent no-op with a printed
warning easy to miss in log noise) would have looked identical to
"fixed" if the reserved-memory check hadn't been done explicitly on
Windows too.

## What NOT to do

- Do not install a Linux NVIDIA driver inside WSL - GPU passthrough
  uses the Windows host driver directly; installing a Linux driver
  breaks this.
- Do not assume the Windows-side `transformers==5.12.1` and `venv_test`'s
  `transformers==5.15.0` / `venv_vllm`'s `transformers==5.14.1` need to
  match each other - `venv_test`/`venv_vllm` are independent scratch/
  benchmark environments serving different purposes. `venv_backend` is
  the one deliberately pinned to match Windows, because it runs the
  real production code, not a benchmark.
- Do not re-attempt any of the Windows-native workarounds this baseline
  was built to replace (no CMake-toolchain installs on the Windows
  side, no `triton-windows`/`autoawq` chase, no vLLM-on-Windows attempts)
  - see the E4B/vLLM investigation memory notes for why those were
  abandoned in favor of this WSL path.

## Multi-runtime model integration (2026-08-13, same session, Phases 1-6)

The approved plan addendum ("Multi-Runtime Model Integration" in the
architecture document) is now IMPLEMENTED and live-verified. Summary of
what exists:

**New modules:**
- `core/gpu_coordinator.py` - tiny cross-runtime GPU ownership arbiter
  (owner tag + per-runtime release callbacks, `claim(runtime_name)`).
  Deliberately a third module so `core/model_residency.py` and
  `core/vllm_runtime.py` never import each other (a mutual-release
  design would be a real Python import cycle). Both runtimes register
  at import time; claiming evicts the other side's holdings first.
- `core/vllm_runtime.py` - vLLM subprocess lifecycle owner. ONE worker
  slot on ONE fixed port (8502, `GENEALOGY_VLLM_PORT` to override), at
  most one subprocess alive; launches `vllm.entrypoints.openai.api_server`
  in `~/venv_vllm` (venv bin dir MUST be on the subprocess PATH -
  flashinfer JIT-compiles its sampling kernel during server warm-up via
  `ninja`, found via PATH; missing it was the spike's first failure),
  waits on `/health` (up to 600s), tears down via SIGTERM -> SIGKILL on
  the whole process group, verifies via `nvidia-smi` (NOT torch's own
  counters - they can't see a subprocess). Per-model
  `vllm_trust_remote_code: true` in config extra (InternVL3.5, MiniCPM).

**Config schema:** `GenerationConfig.runtime` ("transformers" default |
"vllm") is the ONLY new field. vLLM knobs reuse existing fields:
`context_length` -> `--max-model-len`, `vram_headroom_gb` ->
`--gpu-memory-utilization` = (total - headroom)/total.

**Dispatch:** `api/agent_main.py::/chat/turn` branches on
`config.runtime`. The vLLM branch builds a send_turn_fn-compatible
callable proxying `/v1/chat/completions` (image as base64 data-URI
content block), so BOTH plain chat and the full agent loop
(`core.agent_tools.planner.run_agent_turn`) run unchanged against a
vLLM model - verified live: plain "Pong", and an agent turn (CPU
planner -> answer_directly -> "7") through the production API.
`/model/load`/`/model/release`/`/model/status` are runtime-aware
(status reads `residency.resident_model_name`, not the adapter's
possibly-stale bookkeeping, OR the vLLM running model).

**Client (Windows):** `chat_tab.py` gained a Backend local/remote
selector (swaps the adapter, releases the outgoing backend's model,
keeps session state - transport is not a conversation boundary);
`model_console/adapter.py::model_requires_remote_backend()` gates
remote-only models with a pre-send warning on Backend=local.

**Per-checkpoint confirmed status (all under vLLM 0.27.1, ~/venv_vllm):**

| model_name | checkpoint | result |
|---|---|---|
| `qwen25_vl_7b_awq` | Qwen/Qwen2.5-VL-7B-Instruct-AWQ | FULL spike + live API plain-chat AND agent turns; 6.67GiB / 0.26s (earlier offline test) |
| `gemma_12b_w4a16` | google/gemma-4-12B-it-qat-w4a16-ct | PONG confirmed earlier (8.28GiB / 1.34s); not yet re-run through the new dispatch |
| `gemma_e4b_w4a16` | google/gemma-4-E4B-it-qat-w4a16-ct | PONG confirmed earlier (9.93GiB / 1.18s); not yet re-run through the new dispatch |
| `internvl3_5_8b_awq` | cyankiwi/InternVL3_5-8B-AWQ-4bit | FIRST-EVER confirmation under any runtime: 6.96GiB weights, ~145s init, PONG in 0.1s (needs trust_remote_code) |
| `minicpm_v_gptq` | openbmb/MiniCPM-V-4_5-GPTQ | PONG in 0.2s, 189s init - TIGHT FIT: 7.3GiB weights + 6.7GiB peak activation leaves ~0.66GiB KV (~4,800 tokens) at 0.92 util / 2048 ctx; 0.85/4096 fails with "No available memory for cache blocks" |

**Vision spot-check (2026-08-13, same day): PASSED for
`qwen25_vl_7b_awq`** - a real 1906 prairie census crop
(`genealogy_workspace/research/calibration/column_calibration_workspace/
blob_crop_test/e001211812_L_blob_crop_line.png`, 300x400, printed
header partially cut off at the image edge) sent through the FULL
production path (Windows remote ChatBackendAdapter -> POST /chat/turn
-> runtime dispatch -> vLLM data-URI image content block). Ground truth
verified by human-readable inspection first: "Census of Ma[cut] /
Saskatchewan / Alberta, 1906". Model returned exactly
`'Census of Man\nSaskatchewan\nAlberta, 1906'` in 0.54s - and
critically transcribed only the VISIBLE part of the cut-off word
("Man"), not a hallucinated completion to "Manitoba", following the
"transcribe the visible part" instruction - the
abstention-over-fabrication behavior this project treats as the real
quality bar. **UPDATE (2026-08-13, same day): ALL FIVE vLLM models vision-spot-
checked on the same crop, same prompt, same production path.** Ground
truth (human-verified before any model ran): "Census of Ma[cut at
edge] / Saskatchewa[n, partially cut] / Alberta, 1906" plus a faint
"Page" at bottom. The instruction explicitly said to transcribe only
the visible part of edge-cut words - honest cut-off handling is the
pass bar, per this project's abstention-over-fabrication stance.

| model | result | output | time |
|---|---|---|---|
| `qwen25_vl_7b_awq` | PASS | "Census of Man / Saskatchewan / Alberta, 1906" | 0.54s |
| `gemma_e4b_w4a16` | PASS | exact + caught "Page" | 1.71s |
| `gemma_12b_w4a16` | PASS (most conservative) | "Ma" / "Saskatchewa" as literally visible + "Page" | 1.85s |
| `minicpm_v_gptq` | PASS (fastest) | "Ma" + "Page", no KV trouble despite the tight cache | 0.48s |
| `internvl3_5_8b_awq` | **PARTIAL FAIL - fabrication** | "Census of **Nova**" - invented completion of the cut-off word, not visible in the image and factually wrong (prairie census, not Nova Scotia); Saskatchewan/Alberta correct; missed "Page" | 2.18s |

The InternVL3.5 result is the actionable one: on the FIRST real vision
test it hallucinated a cut-off word's completion - the exact failure
mode `feedback_abstention_is_a_feature` names as this project's real
quality axis. One sample, not a verdict - but it starts the
candidate-ranking evidence with a strike, while all four others
(including both Gemma w4a16 checkpoints and MiniCPM) handled the same
ambiguity honestly.

**Two real infrastructure bugs found + fixed during these checks:**
1. `gemma_12b_w4a16`'s server reproducibly crashed at startup
   (`gptq_marlin_repack` -> torch stable-ABI `aten::empty` failure)
   because the vLLM subprocess INHERITED the backend process's
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (set by
   api/agent_main.py for its own transformers release path). Fixed in
   `core/vllm_runtime.py`: the subprocess env now strips that var -
   vLLM tunes its own allocator; the proven run_vllm_*.sh wrappers
   never set it, which is why every offline test passed. Confirmed
   fixed: 12B loads and generates correctly through the server now.
   (Curiously E4B/Qwen tolerated the inherited setting - only 12B's
   larger repack allocation tripped it.)
2. Teardown straggler: one model-switch left a vLLM child process
   alive after `release()`'s killpg (caused the first 12B attempt's
   misleading OOM-shaped failure). Not yet root-caused - the
   killpg/SIGKILL path usually works; if a switch fails with an
   allocation-ish error, check `pgrep -f vllm.entrypoints` for
   stragglers first. A future hardening: verify-and-re-kill loop in
   release() before returning.

**MiniCPM venv conflict, documented for the record:** the original
Phase 4 plan (transformers path via new `core/loaders/minicpm_v_loader.py`
+ gptqmodel in venv_backend) was ABANDONED after install: gptqmodel
7.3.2 requires transformers>=5.15 (venv_backend is pinned 5.12.1 to
match Windows) and downgraded numpy 2.4.6 -> 2.2.6. Fully rolled back -
venv_backend verified back at numpy 2.4.6 / transformers 5.12.1, app
imports clean. The loader file stays as a documented, unused fallback;
the checkpoint runs under vLLM instead. Do not reinstall gptqmodel into
venv_backend without deciding the transformers-version question
deliberately.
