"""
The ONLY module in model_console allowed to import from core.loaders/
core.loader_registry. Everything above this file (chat_tab.py, app.py)
must go through ChatBackendAdapter's plain-Python interface instead of
touching BaseLoader/LOADER_REGISTRY/GenerationConfig directly - see
docs/CODE_MAP.md's "config owns behavior, loaders own mechanics" split,
and the plan's "Component breakdown" section for why this boundary is
enforced by import discipline, not just convention.

Deliberately does NOT reuse core/extractor.py's get_or_build_loader() /
_loader_cache directly - that cache is scoped to extractor.py's own
standalone-process batch pipeline (see model_console/app.py's docstring
on why model_console is always a separate process from it) and must
stay untouched by this module.

2026-08-13 UPDATE: ChatBackendAdapter no longer owns an independent
loader cache at all - it delegates to core.model_residency.residency,
the single process-wide authority over which GPU model is physically
resident. This closes a real gap found while planning agent-mode
hardware behavior: core/agent_tools/tools.py's extract_fields_tool
(a GPU tool usable from THIS process, unlike extractor.py's own
run()) could previously load its own model via extractor's cache while
ChatBackendAdapter's old self._loader independently held a DIFFERENT
model resident at the same time - two independent owners, one GPU,
real OOM risk on a 16GB card. See core/model_residency.py's module
docstring for the full root-cause writeup.

This also resolves the "Known boundary" this docstring used to
document: multiple ChatTab windows now correctly SHARE one resident
loader through the same shared manager, rather than each independently
loading its own copy of a possibly-large model - exactly the direction
this docstring previously flagged as the eventual fix, not a new
tradeoff introduced here. ensure_loaded()/send_turn()'s public
signatures are UNCHANGED - this is a purely internal migration, every
external call site (chat_tab.py) needed zero changes.

2026-08-13 UPDATE #2: ChatBackendAdapter now has TWO backends -
"local" (everything above, unchanged) and "remote" (this adapter
becomes an HTTP client of api/agent_main.py, the FastAPI service
running inside the WSL compute backend - see
docs/WSL_COMPUTE_BACKEND_BASELINE.md's venv_backend section). Selected
via the `backend` constructor arg or the GENEALOGY_CHAT_BACKEND env
var, defaulting to "local" - the remote path is opt-in until it's
passed both the plain-chat and use_agent=True regression tests Jon
asked for; the local Windows-native GPU path is NOT being retired by
this change, only made switchable. chat_tab.py needed ZERO changes for
this - same discipline as the residency migration above: the adapter
absorbs the transport change, every public method keeps its existing
signature and return shape. The one exception is agent mode: since
core.agent_tools.planner.run_agent_turn() (cpu_planner + dispatcher +
tools, all in-process) can't sensibly run partly on Windows and partly
on WSL, a remote-backed adapter runs the ENTIRE agent turn server-side
via one POST to /chat/turn (use_agent=True) - see
model_console/agent_bridge.py's run_agent_chat_turn(), which is the
one place that branches on adapter.backend, not chat_tab.py.

Calls loader._run_generate(raw_image, prompt) directly - NOT
loader.classify()/extract(), which hard-code task="classify"/"extract"
prompt-building and pipeline-schema output parsing that free-form chat
text isn't. This is an established pattern already used the same way
by scripts/model_assessment.py, not a new violation of loader
internals. System prompt flows through config.extra["system_prompt"],
the same channel every loader's own _run_generate already reads to
build its model-correct chat-template message list (see
core/loaders/gemma_loader.py) - so no chat-template code lives here,
and modality/message ordering can never be gotten wrong by this file.
"""

from __future__ import annotations

import base64
import io
import os
import time
from dataclasses import asdict
from typing import Any, Optional

from PIL import Image

from core.agent_status import status_hub
from core.loaders.base_loader import BaseLoader, GenerationConfig, load_model_config
from core.model_residency import residency
from model_console.session import ChatTurn

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard runtime dependency
    torch = None  # noqa: N816

import requests

# Fallback context budget (tokens) used only when neither
# GenerationConfig.context_length nor the loader's own tokenizer.
# model_max_length gives a usable number - see _resolve_context_length().
# Conservative on purpose: better to trim history a bit too aggressively
# than to silently overflow a model's real window.
DEFAULT_CONTEXT_LENGTH_FALLBACK = 8192

# Remote-backend defaults (see this module's 2026-08-13 UPDATE #2
# docstring). localhost works because WSL2 auto-forwards ports bound
# inside WSL to Windows localhost - no static WSL VM IP needed, and no
# .wslconfig mirrored-networking config was required to confirm this
# (see docs/WSL_COMPUTE_BACKEND_BASELINE.md). The timeout is generous
# on purpose: a cold model load alone has been measured up to ~290s,
# and an agent turn can run several model calls (plan, tool, answer)
# back to back.
DEFAULT_REMOTE_BASE_URL = "http://localhost:8001"
DEFAULT_REMOTE_TIMEOUT_SECONDS = 600

# Marker inserted into conversation history for a turn that had an image
# attached - NOT a stand-in for the image's content (see base_loader.py's
# _run_generate_with_history docstring: history is text-only by design).
# This is honest disclosure that something was omitted, not fake visual
# memory - the model is told an image existed and isn't in scope for this
# call, never given invented content pretending to describe it.
_IMAGE_OMITTED_NOTE = "[an image was attached to this message; not included in this context]"


def _encode_image_b64(image: Optional[Image.Image]) -> Optional[str]:
    if image is None:
        return None
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _serialize_history(history: Optional[list[ChatTurn]]) -> Optional[list[dict[str, Any]]]:
    """
    Text-only wire format for the remote backend - image content is
    never sent, matching the local backend's own build_history_for_
    context() policy (see _IMAGE_OMITTED_NOTE above; this applies the
    exact same marker so remote/local produce identical prompts). The
    FULL history is sent, not a client-side pre-trim - trimming to the
    resident model's actual context budget needs that model's tokenizer,
    which only exists server-side, so build_history_for_context() runs
    there (inside the server's own ChatBackendAdapter.send_turn(),
    local-backend code path, unchanged). turn_id is carried across so
    the server's history_turn_ids/dropped_turn_ids provenance in the
    response meta refers to the SAME ids the caller's ChatSession
    already knows, not newly-generated ones.
    """
    if history is None:
        return None
    serialized = []
    for turn in history:
        text = turn.text or ""
        if turn.image is not None:
            text = f"{text}\n{_IMAGE_OMITTED_NOTE}" if text else _IMAGE_OMITTED_NOTE
        serialized.append({"role": turn.role, "text": text, "turn_id": turn.turn_id})
    return serialized


class ChatBackendAdapter:
    """
    One instance per open chat window/session, but NOT an independent
    owner of GPU residency anymore (see this module's docstring,
    2026-08-13 update) - self._model_name/self._config are just this
    adapter's bookkeeping of what IT last asked to be resident; the
    actual loader always comes from core.model_residency.residency,
    fetched fresh via _get_loader() rather than cached locally, since
    another client (an agent tool) may have swapped residency out and
    back in between calls - see _get_loader()'s docstring.
    """

    def __init__(self, backend: Optional[str] = None, base_url: Optional[str] = None) -> None:
        self._backend = (backend or os.environ.get("GENEALOGY_CHAT_BACKEND", "local")).strip().lower()
        if self._backend not in ("local", "remote"):
            raise ValueError(f"Unknown ChatBackendAdapter backend {self._backend!r} - expected 'local' or 'remote'.")
        self._base_url = (base_url or os.environ.get("GENEALOGY_CHAT_BACKEND_URL", DEFAULT_REMOTE_BASE_URL)).rstrip("/")
        self._timeout_seconds = DEFAULT_REMOTE_TIMEOUT_SECONDS
        self._model_name: Optional[str] = None
        self._config: Optional[GenerationConfig] = None

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def resident_model_name(self) -> Optional[str]:
        return self._model_name

    def server_resident_model_name(self) -> Optional[str]:
        """
        The REAL current residency truth, independent of this adapter
        instance's own bookkeeping (resident_model_name above only
        reflects what THIS instance last asked to load, which is wrong
        for "was something already resident before I called
        ensure_loaded" - a fresh adapter's self._model_name always
        starts None even if another client left a model resident).
        backend="local" reads core.model_residency.residency directly
        (the same global singleton _get_loader() already uses).
        backend="remote" asks the server's real state via GET
        /model/status, which itself already reports EITHER a
        transformers-resident model OR a vllm-runtime-resident one (see
        api/agent_main.py's status endpoint: `residency.resident_model_
        name or vllm_runtime.running_model_name`) - so this one call
        covers both engines uniformly. Added for benchmark/
        console_runner.py's was_resident_before_run (2026-08-16,
        backend-comparison work) - no other caller needed this before.
        """
        if self._backend == "remote":
            try:
                resp = requests.get(f"{self._base_url}/model/status", timeout=self._timeout_seconds)
                resp.raise_for_status()
                return resp.json().get("resident_model_name")
            except requests.RequestException:
                return None
        return residency.resident_model_name

    def _get_loader(self) -> Optional[BaseLoader]:
        """
        Always re-acquires through the shared residency manager rather
        than trusting a cached loader reference - cheap (a few field
        comparisons) when this adapter's model is already the resident
        one, but CORRECT even if some other client (e.g. extract_fields_
        tool) borrowed the GPU for a different model and restored this
        one via a fresh reload in between. A stale cached reference
        would otherwise point at a released, dead loader object
        (model=None) after such a swap - see core/model_residency.py's
        module docstring on why nothing here may assume a loader
        reference stays valid across calls.

        Returns None unconditionally for the remote backend - there is
        no local loader to acquire; local GPU residency must never be
        touched by a remote-backed adapter (that would silently load a
        SECOND copy of the model on the Windows GPU alongside whatever
        the WSL backend is doing). Every caller of this method
        (estimate_token_count, _resolve_context_length,
        build_history_for_context) already treats None as "fall back to
        a heuristic," which is exactly correct here too.
        """
        if self._backend == "remote":
            return None
        if self._model_name is None or self._config is None:
            return None
        return residency.acquire(self._model_name, self._config)

    def ensure_loaded(self, model_name: str, config: GenerationConfig) -> None:
        """
        Ensures `model_name` is the resident model - via the shared
        local residency manager for backend="local" (releasing whatever
        else was resident first if a genuinely different model is
        requested, or swapping config in place with no reload if
        `model_name` is already resident - see core/model_residency.py's
        acquire()), or via a POST /model/load to the remote backend for
        backend="remote". `config` is serialized whole
        (dataclasses.asdict) as the remote request's config_overrides -
        applied on top of a fresh load_model_config(model_name) server-
        side (see model_console.adapter.build_config()), which
        reproduces this exact config since both sides start from the
        same YAML; this avoids needing a second "just the overrides"
        representation on top of the GenerationConfig chat_tab.py
        already built.
        """
        if self._backend == "remote":
            self._remote_post("/model/load", {"model_name": model_name, "config_overrides": asdict(config)})
            self._model_name = model_name
            self._config = config
            return
        residency.acquire(model_name, config)
        self._model_name = model_name
        self._config = config

    def send_turn(self, prompt_text: str, image: Optional[Image.Image],
                   system_prompt: str, history: Optional[list[ChatTurn]] = None
                   ) -> tuple[str, dict[str, Any]]:
        """
        Runs one free-form chat turn against the currently resident
        loader. Returns (raw_text, meta) where meta carries the fields
        ChatTurn wants recorded for later review: model_name,
        generation_config_snapshot, runtime_seconds, plus (2026-08-11)
        context provenance: context_enabled, history_turn_ids,
        history_turn_count, approx_input_tokens, dropped_turn_ids.

        history=None (the DEFAULT) is the one-shot path and is
        EXACTLY the code that ran before conversation context existed -
        loader._run_generate(image, prompt_text), nothing else touched.
        This is deliberate: one-shot must remain byte-for-byte identical
        to pre-context-mode behavior, not just "equivalent," since it's
        the scientific control for semantic/taxonomy experiments (see
        model_console/chat_tab.py's "Conversation context" checkbox -
        OFF by default, passes history=None here).

        history=[...] (may be an empty list, e.g. the first turn of a
        fresh context-mode session) takes the history-aware path:
        trims `history` to fit the model's context budget (see
        build_history_for_context()), then calls
        loader._run_generate_with_history(kept_messages, image, prompt_text).
        Raises NotImplementedError if the resident loader has no
        override for that method (see base_loader.py's default) -
        deliberately NOT silently falling back to one-shot, per this
        project's "loud error over silent wrong behavior" discipline;
        model_console's MVP_ALLOWED_MODELS are all confirmed to
        implement it (gemma/gemma_extract/qwen/qwen3vl/internvl/
        lfm2vl/smolvlm2 loader classes), so this should only fire for a
        model added to the picker without that check being done first.

        Raises RuntimeError if called before ensure_loaded(), or
        ValueError if image is None and the resident loader's config
        does NOT declare text_only_supported=True (see base_loader.py's
        GenerationConfig.text_only_supported docstring) - this check
        applies in both modes, unchanged.

        backend="remote": delegates entirely to _send_turn_remote() -
        one POST /chat/turn (use_agent=False) to the WSL backend, same
        (raw_text, meta) return shape, same RuntimeError/ValueError
        contract. Everything below this docstring is the backend="local"
        implementation, byte-for-byte unchanged from before the HTTP-
        client migration.
        """
        if self._model_name is None or self._config is None:
            raise RuntimeError(
                "ChatBackendAdapter.send_turn() called before ensure_loaded() - "
                "no model is resident."
            )
        if self._backend == "remote":
            return self._send_turn_remote(prompt_text, image, system_prompt, history)

        loader = self._get_loader()
        if loader is None:
            raise RuntimeError(
                "ChatBackendAdapter.send_turn() called before ensure_loaded() - "
                "no model is resident."
            )
        if image is None and not loader.config.text_only_supported:
            raise ValueError(
                f"No image attached, and {self._model_name!r}'s config "
                "does not declare text_only_supported=True - either attach an "
                "image, or confirm/add text-only support for this loader "
                "(see base_loader.py's GenerationConfig.text_only_supported)."
            )
        if image is not None and not loader.config.image_input_supported:
            raise ValueError(
                f"An image is attached, but {self._model_name!r}'s config "
                "declares image_input_supported=False - this loader has no "
                "vision path at all (see base_loader.py's GenerationConfig."
                "image_input_supported). Pick a different model for image "
                "input, or remove the attached image."
            )

        loader.config.extra["system_prompt"] = system_prompt

        context_meta: dict[str, Any] = {"context_enabled": history is not None}

        start = time.time()
        if history is None:
            # One-shot - identical to every call site before context mode
            # existed. Do not add anything to this branch.
            try:
                raw_output = loader._run_generate(image, prompt_text)
            except Exception as e:
                if torch is not None and torch.cuda.is_available() and "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    raw_output = loader._run_generate(image, prompt_text)
                else:
                    raise
        else:
            kept_messages, kept_turn_ids, dropped_turn_ids, approx_tokens = \
                self.build_history_for_context(history, system_prompt, prompt_text,
                                                 loader.config.max_new_tokens)
            context_meta.update({
                "history_turn_ids": kept_turn_ids,
                "history_turn_count": len(kept_turn_ids),
                "approx_input_tokens": approx_tokens,
                "dropped_turn_ids": dropped_turn_ids,
            })
            try:
                raw_output = loader._run_generate_with_history(kept_messages, image, prompt_text)
            except NotImplementedError:
                raise
            except Exception as e:
                if torch is not None and torch.cuda.is_available() and "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    raw_output = loader._run_generate_with_history(kept_messages, image, prompt_text)
                else:
                    raise
        runtime_seconds = time.time() - start

        # Phase 3 inference telemetry (2026-08-12) - folds whatever the
        # loader captured for THIS call (getattr default None - only
        # GemmaLoader sets this today, see base_loader.py's
        # last_inference_telemetry docstring) together with the load-side
        # telemetry from the most recent actual model load (None if
        # nothing has loaded yet this process, or reflects a PRIOR turn's
        # load if this model has been resident and reused since - load
        # only happens once, so that's correct, not stale). Both are
        # already separated at the source (residency.py owns load-side,
        # the loader owns generate-side) rather than merged into one
        # ambiguous blob.
        telemetry: dict[str, Any] = {}
        load_telemetry = residency.last_load_telemetry
        if load_telemetry is not None and load_telemetry.get("model_name") == self._model_name:
            telemetry["load"] = load_telemetry
        generate_telemetry = getattr(loader, "last_inference_telemetry", None)
        if generate_telemetry is not None:
            telemetry["generate"] = generate_telemetry

        meta = {
            "model_name": self._model_name,
            "generation_config_snapshot": asdict(loader.config),
            "runtime_seconds": runtime_seconds,
            "telemetry": telemetry or None,
            **context_meta,
        }

        # Status hub update (2026-08-13) - reuses the SAME telemetry
        # dict just built above rather than recomputing anything; this
        # is the one place that reliably knows "which model is THIS
        # chat session's own", independent of whatever a GPU tool may
        # have temporarily borrowed via core.model_residency.residency.
        gen_telemetry = telemetry.get("generate") or {}
        status_hub.update(
            originating_model_name=self._model_name,
            prompt_tokens=gen_telemetry.get("prompt_tokens"),
            generated_tokens=gen_telemetry.get("generated_tokens"),
            tokens_per_sec=gen_telemetry.get("tokens_per_sec"),
            vram_current_mb=gen_telemetry.get("final_vram_mb"),
            vram_peak_mb=gen_telemetry.get("peak_vram_mb"),
            context_enabled=context_meta.get("context_enabled"),
            context_tokens=context_meta.get("approx_input_tokens"),
        )
        return raw_output, meta

    def estimate_token_count(self, text: str) -> int:
        """
        Uses the resident loader's own tokenizer when available (every
        loader used by model_console exposes one - see
        core/loaders/base_loader.py's per-loader self.tokenizer =
        processor.tokenizer convention) - falls back to a crude ~4-chars-
        per-token heuristic only if the tokenizer call itself fails.
        The fallback is a rough English/French Latin-script estimate
        (this project's real content, per this session's own census
        testing), not a general-purpose one - fine as a safety-net
        approximation for trimming, not meant to be exact.
        """
        loader = self._get_loader()
        if loader is not None and loader.tokenizer is not None:
            try:
                return len(loader.tokenizer(text)["input_ids"])
            except Exception:
                pass
        return max(1, len(text) // 4)

    def _resolve_context_length(self) -> int:
        """
        Resolution order: GenerationConfig.context_length (explicit,
        per-model, set from the real model card when known) -> the
        loader's own tokenizer.model_max_length IF it looks like a real
        number (some HF tokenizers report a nonsense sentinel like
        1000000000000000019884624838656 when a model card never set a
        real one - anything absurdly large is treated as "not set,"
        not trusted) -> DEFAULT_CONTEXT_LENGTH_FALLBACK.
        """
        loader = self._get_loader()
        if loader is not None:
            configured = getattr(loader.config, "context_length", None)
            if configured:
                return configured
            tokenizer = getattr(loader, "tokenizer", None)
            model_max = getattr(tokenizer, "model_max_length", None) if tokenizer is not None else None
            if model_max and 0 < model_max < 10_000_000:
                return model_max
        return DEFAULT_CONTEXT_LENGTH_FALLBACK

    def build_history_for_context(
        self, turns: list[ChatTurn], system_prompt: str, current_prompt: str,
        max_new_tokens: int,
    ) -> tuple[list[dict[str, Any]], list[str], list[str], int]:
        """
        Converts real conversational ChatTurns into TEXT-ONLY role-tagged
        message dicts (oldest first) and trims OLDEST turns first until
        (system + kept history + current turn + max_new_tokens reserved)
        fits within the resolved context budget. Returns (kept_messages,
        kept_turn_ids, dropped_turn_ids, approx_input_tokens).

        `turns` should already be ChatSession.conversational_turns() -
        this method does not itself filter out error turns; that
        filtering is session-level, not adapter-level (session.py owns
        "what counts as a real conversational turn," adapter.py owns
        "how much of it fits in the budget").

        Image content is NEVER represented in history turns - see
        _IMAGE_OMITTED_NOTE and base_loader.py's _run_generate_with_history
        docstring for why (Gemma's singular images= kwarg, unconfirmed
        multi-image behavior elsewhere - MVP scope is text-only history
        with the CURRENT turn's image still fully supported).
        """
        context_length = self._resolve_context_length()
        fixed_cost = (
            self.estimate_token_count(system_prompt)
            + self.estimate_token_count(current_prompt)
            + max_new_tokens
        )
        budget = max(0, context_length - fixed_cost)

        candidates: list[tuple[str, dict[str, Any], int]] = []
        for turn in turns:
            text = turn.text or ""
            if turn.image is not None:
                text = f"{text}\n{_IMAGE_OMITTED_NOTE}" if text else _IMAGE_OMITTED_NOTE
            if not text:
                continue
            message = {"role": turn.role, "content": [{"type": "text", "text": text}]}
            candidates.append((turn.turn_id, message, self.estimate_token_count(text)))

        # Keep newest turns first (walk backward from most recent), so
        # the turns dropped when the budget runs out are the OLDEST ones,
        # per the required policy.
        kept: list[tuple[str, dict[str, Any]]] = []
        dropped_turn_ids: list[str] = []
        running_total = 0
        for turn_id, message, cost in reversed(candidates):
            if running_total + cost <= budget:
                kept.append((turn_id, message))
                running_total += cost
            else:
                dropped_turn_ids.append(turn_id)
        kept.reverse()
        dropped_turn_ids.reverse()

        kept_messages = [m for _, m in kept]
        kept_turn_ids = [tid for tid, _ in kept]
        approx_input_tokens = fixed_cost + running_total
        return kept_messages, kept_turn_ids, dropped_turn_ids, approx_input_tokens

    def release(self) -> None:
        """
        Releases the GLOBAL resident model - via the shared local
        residency manager (core.model_residency.residency.release_all())
        for backend="local" (process-wide, so this affects whatever's
        ACTUALLY resident right now, not necessarily still this specific
        adapter's last-requested model if something else has since
        swapped it - the Eject button's intent is "free the GPU," and
        there is only ever at most one model resident to free), or via
        POST /model/release to the remote backend for backend="remote".
        Safe to call when nothing is resident (no-op either way - the
        remote request errors are swallowed for the same reason: freeing
        an already-free GPU should never surface as a UI error). Always
        clears this adapter's own bookkeeping regardless.
        """
        if self._backend == "remote":
            try:
                self._remote_post("/model/release")
            except requests.RequestException:
                pass
            self._model_name = None
            self._config = None
            return
        residency.release_all()
        self._model_name = None
        self._config = None

    def _remote_post(self, path: str, body: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        resp = requests.post(f"{self._base_url}{path}", json=body or {}, timeout=self._timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    def _send_turn_remote(self, prompt_text: str, image: Optional[Image.Image],
                           system_prompt: str, history: Optional[list[ChatTurn]]
                           ) -> tuple[str, dict[str, Any]]:
        """
        The backend="remote" implementation of send_turn() - split out
        so send_turn()'s docstring/signature stay the single public
        contract callers rely on, matching this module's "adapter
        absorbs the transport change" design. model_supports_text_only()
        replaces the local path's `loader.config.text_only_supported`
        check (no local loader object exists here to read it from - it's
        a config.json-only check, defined lower in this file).
        """
        if image is None and not model_supports_text_only(self._model_name):
            raise ValueError(
                f"No image attached, and {self._model_name!r}'s config "
                "does not declare text_only_supported=True - either attach an "
                "image, or confirm/add text-only support for this loader "
                "(see base_loader.py's GenerationConfig.text_only_supported)."
            )
        if image is not None and not model_supports_image_input(self._model_name):
            raise ValueError(
                f"An image is attached, but {self._model_name!r}'s config "
                "declares image_input_supported=False - this loader has no "
                "vision path at all (see base_loader.py's GenerationConfig."
                "image_input_supported). Pick a different model for image "
                "input, or remove the attached image."
            )
        data = self._remote_chat_turn(prompt_text, image, system_prompt, history,
                                       image_path=None, use_agent=False)
        raw_text = data["raw_text"]
        meta = data["meta"]
        self._update_status_hub_from_meta(meta)
        return raw_text, meta

    def send_agent_turn(self, prompt_text: str, image: Optional[Image.Image],
                         system_prompt: str, history: Optional[list[ChatTurn]] = None,
                         image_path: Optional[str] = None) -> dict[str, Any]:
        """
        Remote-backend-only: runs a FULL agent turn (plan -> tool(s) ->
        answer, the same core.agent_tools.planner.run_agent_turn() loop)
        entirely server-side via one POST /chat/turn (use_agent=True),
        and returns the raw JSON response - model_console.agent_bridge.
        run_agent_chat_turn() is the only caller, and it's responsible
        for reconstructing the real AgentTurnResult dataclass chat_tab.py
        expects (see agent_bridge.py's _agent_turn_result_from_dict()).

        Not available for backend="local": the local path already runs
        the same loop in-process via agent_bridge calling this adapter's
        own send_turn() as the model-call primitive - there's no HTTP
        boundary to cross, so this method would be redundant there.
        """
        if self._backend != "remote":
            raise RuntimeError(
                "send_agent_turn() is only valid for backend='remote' - "
                "the local backend runs the agent loop in-process via "
                "agent_bridge.run_agent_chat_turn(), which calls this "
                "adapter's own send_turn() instead."
            )
        if self._model_name is None or self._config is None:
            raise RuntimeError(
                "ChatBackendAdapter.send_agent_turn() called before ensure_loaded() - "
                "no model is resident."
            )
        return self._remote_chat_turn(prompt_text, image, system_prompt, history,
                                       image_path, use_agent=True)

    def _remote_chat_turn(self, prompt_text: str, image: Optional[Image.Image],
                           system_prompt: str, history: Optional[list[ChatTurn]],
                           image_path: Optional[str], use_agent: bool) -> dict[str, Any]:
        body = {
            "model_name": self._model_name,
            "system_prompt": system_prompt,
            "user_text": prompt_text,
            "image_base64": _encode_image_b64(image),
            "image_path": image_path,
            "history": _serialize_history(history),
            "use_agent": use_agent,
            "config_overrides": asdict(self._config) if self._config is not None else None,
        }
        resp = requests.post(f"{self._base_url}/chat/turn", json=body, timeout=self._timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    def _update_status_hub_from_meta(self, meta: dict[str, Any]) -> None:
        """
        Mirrors the local send_turn()'s own status_hub.update() call
        (below, unchanged) using the SAME meta shape the server returns
        - keeps the Windows-side agent status dashboard populated for
        remote-backed turns too, instead of silently going dark just
        because the model call happened on WSL instead of in-process.
        """
        gen_telemetry = (meta.get("telemetry") or {}).get("generate") or {}
        status_hub.update(
            originating_model_name=self._model_name,
            prompt_tokens=gen_telemetry.get("prompt_tokens"),
            generated_tokens=gen_telemetry.get("generated_tokens"),
            tokens_per_sec=gen_telemetry.get("tokens_per_sec"),
            vram_current_mb=gen_telemetry.get("final_vram_mb"),
            vram_peak_mb=gen_telemetry.get("peak_vram_mb"),
            context_enabled=meta.get("context_enabled"),
            context_tokens=meta.get("approx_input_tokens"),
        )


def build_config(model_name: str, system_prompt: str = "",
                  overrides: Optional[dict[str, Any]] = None) -> GenerationConfig:
    """
    Loads config/models/<model_name>.yaml via the existing
    load_model_config() and applies chat-session overrides on top, on a
    fresh in-memory copy - the YAML file on disk is never touched, same
    discipline as model_assessment.py's own override handling. Does NOT
    set config.extra["system_prompt"] here - send_turn() sets it per
    call, since a chat session's system prompt can be edited between
    turns without reloading the model.
    """
    config = load_model_config(model_name)
    for key, value in (overrides or {}).items():
        setattr(config, key, value)
    return config


def list_console_models() -> list[str]:
    """
    Config-driven Model Console model list (2026-08-14) - reads
    config/pipeline.yaml's console.allowed_models instead of the old
    hardcoded MVP_ALLOWED_MODELS Python list in chat_tab.py, so Model
    Console consumes the same registry the extraction pipeline does
    rather than maintaining its own separate, drifting curation. Filters
    to models that both have a config file on disk AND are enabled -
    same tolerance the old list_model_profiles() had for a listed name
    with no matching YAML, extended to also honor the registry's
    enabled: false switch (see GenerationConfig.enabled).
    """
    import yaml
    from core.classifier import PROJECT_ROOT
    from core.loaders.base_loader import CONFIG_DIR

    with open(PROJECT_ROOT / "config" / "pipeline.yaml", "r", encoding="utf-8") as f:
        pipeline_cfg = yaml.safe_load(f)
    allowed = (pipeline_cfg.get("console") or {}).get("allowed_models") or []

    available = {p.stem for p in CONFIG_DIR.glob("*.yaml")}
    result = []
    for name in allowed:
        if name not in available:
            continue
        try:
            if not load_model_config(name).enabled:
                continue
        except Exception:
            continue
        result.append(name)
    return result


def default_console_model() -> str:
    """First entry of list_console_models() - the Model Console dropdown
    default. Raises if the registry produced no usable models at all,
    rather than letting the UI silently start with nothing selected."""
    models = list_console_models()
    if not models:
        raise ValueError(
            "config/pipeline.yaml console.allowed_models produced no "
            "usable models (none exist/are enabled in config/models/)."
        )
    return models[0]


def model_supports_text_only(model_name: str) -> bool:
    """
    Cheap capability check - reads config/models/<model_name>.yaml (no
    model load, no GPU touch) and returns its text_only_supported flag.
    Exists so chat_tab.py can decide whether to require an attached
    image WITHOUT importing core.loaders itself (see adapter.py's own
    module docstring on why that import boundary is enforced).
    """
    return load_model_config(model_name).text_only_supported


def model_supports_image_input(model_name: str) -> bool:
    """
    Mirror-image of model_supports_text_only() above (see base_loader.
    py's GenerationConfig.image_input_supported) - cheap capability
    check, no model load, no GPU touch. Exists so chat_tab.py can warn
    or block BEFORE dispatching a send (attaching an image for a
    no-vision-tower model like qwen_research_text would otherwise only
    fail after a full, possibly multi-minute, cold model load).
    """
    return load_model_config(model_name).image_input_supported


def model_requires_remote_backend(model_name: str) -> bool:
    """
    True when this model can only be served by the WSL backend, never
    by a backend="local" (Windows in-process) adapter - same cheap
    config-only check pattern as the two helpers above. Two ways a
    config ends up remote-only (2026-08-13, multi-runtime integration):
    runtime != "transformers" (vLLM subprocess models - vLLM has no
    Windows support at all), or an explicit extra key
    `requires_backend: remote` (transformers-path models whose
    dependency stack only exists in WSL, e.g. minicpm_v_gptq needing
    gptqmodel, which has no Windows build).
    """
    config = load_model_config(model_name)
    return config.runtime != "transformers" or config.extra.get("requires_backend") == "remote"
