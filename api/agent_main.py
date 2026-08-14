"""
Agent/chat compute backend - the WSL-side production API that a
Windows-side model_console client talks to over HTTP (see the
"minimum viable migration" portability plan: wrap run_agent_chat_turn()
as one endpoint, small control endpoints for model selection and the
internet-access toggle, ChatBackendAdapter becomes an HTTP client on
the Windows side instead of an in-process caller).

Separate FastAPI app from api/main.py on purpose - that one is the
Phase 1 read-only pipeline-monitoring API (wraps core/pipeline_db.py,
a completely different concern: batch pipeline status, not chat/agent
turns). Follows the same "thin API layer, real logic lives in core/ /
model_console/" discipline as api/main.py: this module does HTTP
plumbing and request/response shaping only - every real decision
(which model loads, how a turn runs, how history gets trimmed) already
lives in model_console/adapter.py, model_console/agent_bridge.py, and
core/agent_tools/planner.py, unchanged.

One process = one GPU owner (core/model_residency.py's existing
single-authority discipline extends unchanged to this process being
the sole thing touching the GPU on the WSL side) - this module holds
exactly one module-level ChatBackendAdapter instance, matching
"one instance per open chat window/session" loosely relaxed to "one
instance per backend process," since Windows-side model_console is now
the thing that owns per-window chat state (ChatSession/ChatTurn
history), not this process.

History is passed as plain {role, text, turn_id} triples over the wire
(not real ChatTurn/ChatImage objects - those are Windows-side session
state) and rebuilt into ChatTurn(image=None) here, matching adapter.py's
own text-only-history contract (images are never represented in
history, only the CURRENT turn's image is used - see adapter.py's
_IMAGE_OMITTED_NOTE docstring; the client-side adapter already inserts
that same marker into a turn's text before it ever reaches this
endpoint, see model_console/adapter.py's _serialize_history()).
turn_id is preserved (not regenerated) so this response's
history_turn_ids/dropped_turn_ids meta refers to the SAME ids the
calling ChatSession already knows. The current turn's image, if any,
arrives as a base64-encoded string and is decoded to a PIL.Image here.

Run via: uvicorn api.agent_main:app --host 0.0.0.0 --port 8001
(bound to 0.0.0.0, not 127.0.0.1 - this process runs inside WSL and
must be reachable from the Windows host across the WSL virtual network
boundary; see docs/WSL_COMPUTE_BACKEND_BASELINE.md for how Windows
reaches WSL's network namespace).

2026-08-13: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is set
below, BEFORE torch is imported anywhere in this process (transitively,
via model_console.adapter) - a real VRAM-not-released bug was found and
confirmed here: after a load -> generate -> release cycle,
torch.cuda.memory_allocated() correctly drops to near-zero but
memory_reserved() (and therefore nvidia-smi) stayed pinned at the
model's full size (~10.2GB for gemma_extract). Root cause: PyTorch's
default (non-expandable) caching allocator can only return an entire
memory SEGMENT to the driver once every byte in it is free: one small
~33.6MB persistent allocation (a cuBLAS/cuDNN workspace, kept alive for
the process's lifetime by design) happened to share the same large
segment as the model weights, pinning the whole ~10GB segment
indefinitely even after every real tensor was freed. Confirmed live:
without this setting, reserved_mb/nvidia-smi stayed at ~10.2GB/11.5GB
after release; with it, they drop to ~42MB/1.8MB (idle baseline) - see
GET /debug/gpu and POST /debug/empty_cache below, added while
diagnosing this. This is a general PyTorch/CUDA-allocator behavior, not
specific to this HTTP backend - core/model_residency.py's release path
itself is unchanged and correct; this only needed to be set once,
early, in the process's environment.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import base64
import io
from dataclasses import asdict
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from core.network_gate import network_gate
from core.vllm_runtime import vllm_runtime
from model_console.adapter import (
    ChatBackendAdapter, build_config, model_supports_image_input, model_supports_text_only,
)
from model_console.agent_bridge import run_agent_chat_turn
from model_console.session import ChatTurn

app = FastAPI(
    title="Genealogy Agent/Chat Backend (WSL compute backend)",
    description=(
        "Wraps ChatBackendAdapter + run_agent_chat_turn over HTTP for a "
        "Windows-side client. Single-GPU-owner: one adapter instance for "
        "this whole process, mirroring core/model_residency.py's process-"
        "wide residency authority."
    ),
    version="0.1.0",
)

_adapter = ChatBackendAdapter()


class HistoryTurn(BaseModel):
    role: str
    text: str
    turn_id: Optional[str] = None


class ChatTurnRequest(BaseModel):
    model_name: str
    system_prompt: str = ""
    user_text: str
    image_base64: Optional[str] = None
    image_path: Optional[str] = None
    history: Optional[list[HistoryTurn]] = None
    use_agent: bool = True
    config_overrides: Optional[dict[str, Any]] = None


class ChatTurnResponse(BaseModel):
    raw_text: str
    meta: dict[str, Any]
    agent_result: Optional[dict[str, Any]] = None


class ModelLoadRequest(BaseModel):
    model_name: str
    config_overrides: Optional[dict[str, Any]] = None


class ModelStatusResponse(BaseModel):
    resident_model_name: Optional[str]


class NetworkToggleRequest(BaseModel):
    enabled: bool


class NetworkStatusResponse(BaseModel):
    enabled: bool


def _decode_image(image_base64: Optional[str]) -> Optional[Image.Image]:
    if image_base64 is None:
        return None
    try:
        raw = base64.b64decode(image_base64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image_base64: {e}") from e


def _localize_client_path(path: Optional[str]) -> Optional[str]:
    """
    Converts a Windows drive path (J:\\..., E:/...) from the client into
    this host's local form when running on POSIX/WSL (J:\\x -> /mnt/j/x).
    THE one normalization point for client-supplied filesystem paths -
    applied at the request boundary so everything downstream (the
    image-open fallback, run_agent_chat_turn, the planner putting the
    path into tool args like extract_fields' file_path) sees a locally
    valid path. Found live 2026-08-14: an agent turn's extract_fields
    failed not_found because the planner received the Windows-form
    image_path verbatim inside WSL. No-op on Windows, for non-drive
    paths, and when the /mnt translation doesn't actually exist (an
    unmounted drive letter shouldn't silently produce a different
    wrong path - the original at least names what the client meant).
    """
    if not path or os.name != "posix":
        return path
    if len(path) >= 3 and path[1] == ":" and path[2] in ("\\", "/"):
        candidate = f"/mnt/{path[0].lower()}/" + path[3:].replace("\\", "/")
        if Path(candidate).exists():
            return candidate
    return path


def _encode_image_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _make_vllm_send_turn_fn(model_name: str, config):
    """
    Builds a send_turn_fn matching ChatBackendAdapter.send_turn()'s
    (prompt_text, image, system_prompt, history) -> (raw_text, meta)
    contract, backed by the vLLM server subprocess instead of an
    in-process loader - the runtime-dispatch seam (2026-08-13, plan
    addendum "Multi-Runtime Model Integration"). This is what lets both
    the plain-chat path AND core.agent_tools.planner.run_agent_turn()
    (which only needs this callable, never a loader) run unchanged
    against a vLLM model: nothing downstream knows which runtime
    served the turn.

    History is text-only role/content pairs (same policy as the
    transformers path - adapter.py's _serialize_history() already
    replaced any history image with the omitted-note marker client-
    side). The CURRENT turn's image goes in as an OpenAI image_url
    content block with a base64 data URI - the vLLM server does its own
    multimodal preprocessing; no client-side token budgeting applies.
    """
    import time as _time

    def send_turn_fn(prompt_text: str, image=None, system_prompt: str = "",
                      history=None) -> tuple[str, dict[str, Any]]:
        # Re-acquire per call, not just once per turn (2026-08-14, found
        # live the moment the path-localization fix let extract_fields
        # actually run): a GPU tool dispatched mid-agent-turn loads a
        # transformers model via residency.borrow(), whose
        # gpu_coordinator.claim("transformers") correctly EVICTS this
        # turn's own vLLM subprocess - so the final-answer step must
        # re-establish it. acquire() is a cheap no-op when the server is
        # still running; after an eviction it pays a full engine re-init
        # (~3min) - a known cost of mixing a vLLM chat model with
        # transformers GPU tools in one turn on a single 16GB card, not
        # a bug. The alternative (skipping the answer) is worse.
        vllm_runtime.acquire(model_name, config)
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for turn in (history or []):
            text = getattr(turn, "text", None) or ""
            if text:
                messages.append({"role": turn.role, "content": text})
        if image is not None:
            user_content: Any = [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": _encode_image_data_uri(image)}},
            ]
        else:
            user_content = prompt_text
        messages.append({"role": "user", "content": user_content})

        temperature = config.temperature if config.do_sample else 0.0
        start = _time.time()
        raw_text = vllm_runtime.chat_completion(
            config, messages, max_tokens=config.max_new_tokens, temperature=temperature,
        )
        meta = {
            "model_name": model_name,
            "runtime": "vllm",
            "generation_config_snapshot": asdict(config),
            "runtime_seconds": _time.time() - start,
            "context_enabled": history is not None,
        }
        return raw_text, meta

    return send_turn_fn


@app.post("/chat/turn", response_model=ChatTurnResponse)
def chat_turn(req: ChatTurnRequest) -> dict:
    """
    Runs one turn against whatever model this request asks for -
    ensure_loaded() is called here (not left to the client), so a
    single HTTP call is enough to both select and use a model, mirroring
    what chat_tab.py's picker + send button do together today.

    use_agent=True (default) runs the full plan -> tool(s) -> answer
    loop via run_agent_chat_turn(); use_agent=False calls
    adapter.send_turn() directly for a plain one-shot/history chat turn,
    matching model_console's existing "agent mode" vs. plain chat
    distinction (see agent_bridge.py's module docstring - not yet wired
    into chat_tab.py's UI, but already a real, separate code path).
    """
    image = _decode_image(req.image_base64)
    # Localize the client's path ONCE at the boundary - image_path from a
    # Windows client is Windows-form; everything below (image-open
    # fallback, planner tool args) needs this host's form.
    image_path = _localize_client_path(req.image_path)
    if image is None and image_path:
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not open image_path: {e}") from e

    if image is None and req.history is None and not model_supports_text_only(req.model_name):
        raise HTTPException(
            status_code=422,
            detail=f"{req.model_name!r} does not support text-only input - attach an image.",
        )
    if image is not None and not model_supports_image_input(req.model_name):
        raise HTTPException(
            status_code=422,
            detail=f"{req.model_name!r} has no vision tower - it cannot accept an image "
                    "(config/models/<name>.yaml's image_input_supported is False).",
        )

    history: Optional[list[ChatTurn]] = None
    if req.history is not None:
        history = []
        for h in req.history:
            kwargs: dict[str, Any] = {"role": h.role, "text": h.text}
            if h.turn_id:
                kwargs["turn_id"] = h.turn_id
            history.append(ChatTurn(**kwargs))

    # Runtime dispatch (2026-08-13, plan addendum): branch on the
    # config's runtime field BEFORE any loader/adapter involvement -
    # "vllm" models are served by core/vllm_runtime.py's subprocess and
    # never instantiate a loader class (their loader_class is a
    # sentinel). Everything downstream of send_turn_fn (planner loop,
    # response shaping) is identical between the two branches.
    config = build_config(req.model_name, overrides=req.config_overrides)

    if req.use_agent:
        # Agent turns ALWAYS run on the text-only research LLM
        # (transformers), regardless of which model the request named -
        # 2026-08-14, Jon's policy: "default qwen 7b as the transformers
        # model, if it needs VLM it can call an agent." Vision happens
        # inside tools (extract_fields' borrowed extraction model, the
        # planner's borrowed category-classify VLM - see
        # core/agent_tools/planner.py); the chat/planning/synthesis
        # brain never receives pixels. This also eliminates the
        # confirmed runtime-thrash failure (2026-08-14: an agent turn
        # on a vLLM chat model alternated GPU owners 3-4x, ~15+ min,
        # past the client's HTTP timeout) - agent turns now stay
        # entirely inside the transformers residency system, where
        # tool-model borrows are cheap swap-and-restore. The requested
        # model_name still applies to PLAIN chat turns below.
        from core.agent_tools.research_llm import MODEL_NAME as research_model_name
        research_config = build_config(research_model_name)
        _adapter.ensure_loaded(research_model_name, research_config)
        result = run_agent_chat_turn(
            _adapter, req.user_text, image, req.system_prompt,
            history=history, image_path=image_path,
        )
        return {
            "raw_text": result.final_answer,
            "meta": result.answer_meta or {},
            "agent_result": asdict(result),
        }

    raw_text, meta = _adapter.send_turn(req.user_text, image, req.system_prompt, history=history)
    return {"raw_text": raw_text, "meta": meta, "agent_result": None}


@app.post("/model/load", response_model=ModelStatusResponse)
def load_model(req: ModelLoadRequest) -> dict:
    config = build_config(req.model_name, overrides=req.config_overrides)
    if config.runtime == "vllm":
        vllm_runtime.acquire(req.model_name, config)
        return {"resident_model_name": vllm_runtime.running_model_name}
    _adapter.ensure_loaded(req.model_name, config)
    return {"resident_model_name": _adapter.resident_model_name}


@app.post("/model/release", response_model=ModelStatusResponse)
def release_model() -> dict:
    # Frees the GPU regardless of which runtime currently owns it - both
    # release paths are safe no-ops when that runtime holds nothing.
    _adapter.release()
    vllm_runtime.release()
    return {"resident_model_name": None}


@app.get("/model/status", response_model=ModelStatusResponse)
def model_status() -> dict:
    # residency (not _adapter's own bookkeeping) is the truth for the
    # transformers side: a vLLM acquire evicts the resident model via
    # gpu_coordinator WITHOUT the adapter knowing, so _adapter._model_name
    # can be stale after a runtime switch. At most one of these is
    # non-None (gpu_coordinator's exclusion).
    from core.model_residency import residency
    return {"resident_model_name": residency.resident_model_name or vllm_runtime.running_model_name}


@app.get("/model/{model_name}/text_only_supported")
def text_only_supported(model_name: str) -> dict:
    try:
        return {"text_only_supported": model_supports_text_only(model_name)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.post("/network/toggle", response_model=NetworkStatusResponse)
def toggle_network(req: NetworkToggleRequest) -> dict:
    network_gate.set_enabled(req.enabled)
    return {"enabled": network_gate.enabled}


@app.get("/network/status", response_model=NetworkStatusResponse)
def network_status() -> dict:
    return {"enabled": network_gate.enabled}


@app.post("/debug/empty_cache")
def debug_empty_cache() -> dict:
    """
    Diagnostic-only: a more forceful cache-clear than core.model_
    residency._release_current()'s own gc.collect()+empty_cache() -
    added while investigating a real case (2026-08-13) where reserved_mb
    stayed pinned near the just-released model's full size even though
    allocated_mb had already dropped near zero (tensors genuinely freed,
    caching allocator just not returning the blocks). synchronize()
    before empty_cache() rules out a pending-async-op explanation;
    ipc_collect() rules out cross-process CUDA IPC handles holding a
    block open.
    """
    import gc
    try:
        import torch
    except ImportError:
        return {"cuda_available": False}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return debug_gpu()


@app.get("/debug/gpu")
def debug_gpu() -> dict:
    """
    Raw torch.cuda memory counters for THIS process, read directly
    (not via nvidia-smi) - lets a caller tell apart "tensors genuinely
    still referenced" (memory_allocated) from "PyTorch's caching
    allocator holding freed blocks it hasn't returned to the driver
    yet" (memory_reserved) after a /model/release call, which
    nvidia-smi's single number can't distinguish.
    """
    try:
        import torch
    except ImportError:
        return {"cuda_available": False}
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    return {
        "cuda_available": True,
        "allocated_mb": round(torch.cuda.memory_allocated() / 1e6, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1e6, 1),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
    }
