"""
The chat widget itself. Deliberately imports ONLY from model_console.*
(adapter.py, session.py, session_log.py, image_prep.py) plus stdlib/
tkinter/PIL - never from core.loaders/core.loader_registry directly.
That import boundary is what makes it structurally impossible to write
a model-specific branch (e.g. "if model_name == 'gemma': ...") into
this file - it would have to go in adapter.py, the one place allowed
to know loader internals exist. See the plan's "Component breakdown"
section.

Threading model: model_assessment.py's OVERRIDABLE_FIELDS/override
pattern is reused, but the actual model call runs on a background
thread (adapter calls are seconds-long and must not freeze the UI),
with root.after() polling a queue.Queue for the result - the same
polling idiom debug_tools/workflow_gui.py's CommandTab uses for a
subprocess, applied here to an in-process Python call instead.
"""

from __future__ import annotations

import queue
import threading
import traceback
from pathlib import Path
from typing import Optional

import yaml
from tkinter import (
    Frame, Label, Button, Checkbutton, BooleanVar, StringVar, Entry, OptionMenu, Text, Scrollbar,
    filedialog, messagebox, END, DISABLED, NORMAL, WORD, BOTH, X, Y, LEFT, RIGHT, TOP, BOTTOM,
)
from tkinter import ttk
from PIL import Image, ImageTk

from core.loaders.base_loader import CONFIG_DIR as MODELS_DIR
from core.network_gate import network_gate
from model_console.adapter import (
    ChatBackendAdapter, build_config, model_requires_remote_backend,
    model_supports_image_input, model_supports_text_only,
    list_console_models,
)
from model_console.agent_bridge import run_agent_chat_turn
from model_console.conversation_manager import get_effective_context, maybe_summarize_session
from model_console.image_prep import prep_for_chat, max_dimension_for_model
from model_console.session import ChatSession, ChatTurn
from model_console import session_log

# Model list + default (2026-08-14, config-driven registry pass) now
# live in config/pipeline.yaml's console.allowed_models, read via
# model_console.adapter.list_console_models() - NOT a hardcoded Python
# list here anymore. See that config section's own comment block for
# the full history/reasoning behind which models are listed, in what
# order, and why gemma_extract (not gemma) is the default. This file
# still enforces the import boundary from its own module docstring
# above: it reads the list through adapter.py, never touches
# core.loaders/core.loader_registry directly.

OVERRIDABLE_FIELDS = [
    "temperature", "top_p", "top_k", "max_new_tokens",
    "do_sample", "repetition_penalty", "no_repeat_ngram_size",
]

THUMBNAIL_SIZE = (160, 160)

# Named, reusable system-prompt snippets (2026-08-11) - a lighter-weight
# precursor to the plan's future "agent profiles" (Phase 2: system_prompt
# + generation_overrides + capabilities + output_mode), scoped down to
# just the one thing asked for right now: quick-select prompt text so it
# doesn't have to be retyped/re-pasted after every app restart (the
# system prompt box itself is NOT persisted across restarts - this
# doesn't change that, it just makes reloading a saved one fast). One
# file per prompt, same directory-of-YAMLs convention as
# config/models/*.yaml and config/bucket_profiles/*.yaml.
CONSOLE_PROMPTS_DIR = MODELS_DIR.parent / "console_prompts"


def list_model_profiles() -> list[str]:
    return list_console_models()


def load_console_prompts() -> dict[str, dict]:
    """
    Scans config/console_prompts/*.yaml. Returns {stem: {"display_name":
    str, "system_prompt": str}}. A malformed individual file is skipped
    with a console warning rather than aborting the whole load - one bad
    file shouldn't block every other saved prompt from showing up.
    """
    prompts: dict[str, dict] = {}
    if not CONSOLE_PROMPTS_DIR.is_dir():
        return prompts
    for path in sorted(CONSOLE_PROMPTS_DIR.glob("*.yaml")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            prompts[path.stem] = {
                "display_name": data.get("display_name", path.stem),
                "system_prompt": data.get("system_prompt", ""),
            }
        except Exception as e:
            print(f"[model_console.chat_tab] WARNING: skipping malformed "
                  f"console prompt {path.name!r}: {type(e).__name__}: {e}")
    return prompts


class ChatTab(Frame):
    """One chat session per instance. Owns its own ChatBackendAdapter
    and ChatSession - opening a second ChatTab gets its own model
    instance entirely (see adapter.py's "known boundary" note about
    this not scaling past a couple of concurrent windows)."""

    def __init__(self, master):
        super().__init__(master)
        self.adapter = ChatBackendAdapter()
        self.session: Optional[ChatSession] = None
        self._result_queue: "queue.Queue" = queue.Queue()
        self._pending_image_path: Optional[Path] = None
        self._pending_image_thumb = None  # keep a reference, Tk drops GC'd images
        self._sending = False

        self._build_widgets()
        self._on_model_change()

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _build_widgets(self) -> None:
        top = Frame(self)
        top.pack(side=TOP, fill=X, padx=6, pady=4)

        Label(top, text="Model:").pack(side=LEFT)
        profiles = list_model_profiles()
        self.model_var = StringVar(value=profiles[0] if profiles else "(none found)")
        OptionMenu(top, self.model_var, *(profiles or ["(none found)"]),
                   command=lambda _v: self._on_model_change()).pack(side=LEFT, padx=(2, 12))

        # Backend selector (2026-08-13, multi-runtime integration) -
        # "local" = the unchanged Windows-native in-process path,
        # "remote" = HTTP client of api/agent_main.py running in WSL
        # (see adapter.py's 2026-08-13 UPDATE #2 docstring). Default
        # stays "local" per the migration decision: the local path is
        # NOT retired, remote is opt-in. Switching mid-session releases
        # the CURRENT backend's model first (each backend's release is
        # a safe no-op when nothing is resident), then swaps in a fresh
        # adapter - session/transcript state is untouched, only the
        # execution transport changes.
        Label(top, text="Backend:").pack(side=LEFT)
        self.backend_var = StringVar(value=self.adapter.backend)
        OptionMenu(top, self.backend_var, "local", "remote",
                   command=lambda _v: self._on_backend_change()).pack(side=LEFT, padx=(2, 12))

        self.eject_button = Button(top, text="Eject", command=self._on_eject)
        self.eject_button.pack(side=LEFT, padx=(0, 12))

        # Conversation context checkbox (2026-08-11). DEFAULT OFF - one-shot
        # stays the default so it remains a clean scientific control for
        # semantic/taxonomy experiments (per explicit direction). Purely
        # functional, no styling beyond the default Checkbutton look -
        # this whole UI is a temporary placeholder pending a later polish
        # pass, not something worth investing design time in now.
        self.context_var = BooleanVar(value=False)
        Checkbutton(top, text="Conversation context", variable=self.context_var).pack(
            side=LEFT, padx=(0, 12))

        # Agent mode (2026-08-13, Phase 2 of the agent architecture plan).
        # DEFAULT OFF for the same reason context defaults off: plain chat
        # must stay a clean, unmediated one-shot/context call to the model
        # (core.agent_tools.planner adds a planning turn + optional tool
        # call + a second answer turn in front of what the user sees, so
        # it is never the default). When on, _worker() routes through
        # model_console.agent_bridge.run_agent_chat_turn() instead of
        # calling adapter.send_turn() directly - see that module and
        # core/agent_tools/planner.py for what actually happens.
        self.agent_mode_var = BooleanVar(value=False)
        Checkbutton(top, text="Agent mode (tools)", variable=self.agent_mode_var).pack(
            side=LEFT, padx=(0, 12))

        # Internet access (2026-08-13, Phase 7) - explicit, visible,
        # OFF-by-default gate for core/agent_tools/tools.py's web_search
        # tool, the FIRST tool that can leave the local machine. Setting
        # this immediately flips core.network_gate.network_gate, read
        # LIVE by web_search at call time (not snapshotted per-turn) -
        # so toggling mid-conversation takes effect on the very next
        # tool call. See core/network_gate.py's own module docstring.
        self.internet_access_var = BooleanVar(value=False)
        Checkbutton(
            top, text="Internet access", variable=self.internet_access_var,
            command=self._on_internet_access_toggle,
        ).pack(side=LEFT, padx=(0, 12))

        self.status_label = Label(top, text="Not loaded", anchor="w")
        self.status_label.pack(side=LEFT, fill=X, expand=True)

        Label(self, text="System prompt (editable, per-session):", anchor="w").pack(
            side=TOP, fill=X, padx=6)
        self.system_prompt_box = Text(self, height=3, wrap=WORD)
        self.system_prompt_box.pack(side=TOP, fill=X, padx=6, pady=(0, 4))

        prompt_picker_row = Frame(self)
        prompt_picker_row.pack(side=TOP, fill=X, padx=6, pady=(0, 4))
        Label(prompt_picker_row, text="Quick prompt:").pack(side=LEFT)
        self._console_prompts = load_console_prompts()
        self.prompt_picker_var = StringVar()
        display_names = [p["display_name"] for p in self._console_prompts.values()]
        self.prompt_picker = ttk.Combobox(
            prompt_picker_row, textvariable=self.prompt_picker_var,
            values=display_names, state="readonly", width=30,
        )
        self.prompt_picker.pack(side=LEFT, padx=(4, 4))
        Button(prompt_picker_row, text="Load",
               command=self._load_selected_console_prompt).pack(side=LEFT)
        Button(prompt_picker_row, text="Reload list",
               command=self._reload_console_prompts).pack(side=LEFT, padx=(4, 0))

        settings_frame = Frame(self)
        settings_frame.pack(side=TOP, fill=X, padx=6, pady=(0, 4))
        Label(settings_frame, text="Generation settings (pre-filled from the selected "
                                    "model's config - edit to override):").pack(
            side=TOP, anchor="w")
        self._override_entries: dict[str, StringVar] = {}
        fields_row = Frame(settings_frame)
        fields_row.pack(side=TOP, fill=X)
        for field_name in OVERRIDABLE_FIELDS:
            cell = Frame(fields_row)
            cell.pack(side=LEFT, padx=(0, 8))
            Label(cell, text=field_name).pack(side=TOP)
            var = StringVar()
            Entry(cell, textvariable=var, width=8).pack(side=TOP)
            self._override_entries[field_name] = var
        Button(settings_frame, text="Reset to model default",
               command=self._reset_overrides).pack(side=LEFT, padx=(8, 0))

        # Transcript - read-only, appended to programmatically only.
        transcript_frame = Frame(self)
        transcript_frame.pack(side=TOP, fill=BOTH, expand=True, padx=6, pady=(0, 4))
        scrollbar = Scrollbar(transcript_frame)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.transcript = Text(transcript_frame, wrap=WORD, state=DISABLED,
                                yscrollcommand=scrollbar.set)
        self.transcript.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.config(command=self.transcript.yview)
        self.transcript.tag_configure("role_user", foreground="#1a4d8f")
        self.transcript.tag_configure("role_assistant", foreground="#1a7a3d")
        self.transcript.tag_configure("role_error", foreground="#a01414")
        self.transcript.tag_configure("role_agent", foreground="#8a6d00")
        self.transcript.tag_configure("role_tool", foreground="#0a7a5c")
        self.transcript.tag_configure("telemetry", foreground="#888888", font=("TkDefaultFont", 8))

        # Image attach row.
        image_row = Frame(self)
        image_row.pack(side=TOP, fill=X, padx=6, pady=(0, 4))
        Button(image_row, text="Attach image...", command=self._pick_image).pack(side=LEFT)
        self.image_thumb_label = Label(image_row)
        self.image_thumb_label.pack(side=LEFT, padx=8)
        self.image_name_label = Label(image_row, text="(no image attached)", anchor="w")
        self.image_name_label.pack(side=LEFT, fill=X, expand=True)
        Button(image_row, text="Clear image", command=self._clear_image).pack(side=RIGHT)

        # Input + send.
        input_row = Frame(self)
        input_row.pack(side=BOTTOM, fill=X, padx=6, pady=(0, 6))
        self.input_box = Text(input_row, height=3, wrap=WORD)
        self.input_box.pack(side=LEFT, fill=BOTH, expand=True)
        self.send_button = Button(input_row, text="Send", command=self._on_send)
        self.send_button.pack(side=RIGHT, fill="y", padx=(6, 0))

        self._append_transcript(
            "system", "Each message is answered independently - the model does "
            "not see earlier turns yet (no multi-turn context in this build). "
            "Text-only messages are supported for models confirmed to allow "
            "it (see the model picker) - others still require an attached "
            "image per message."
        )

    def _reload_console_prompts(self) -> None:
        """
        Re-scans config/console_prompts/*.yaml and refreshes the
        dropdown's values in place - added 2026-08-13 because this is
        the ONE thing in the app genuinely cached at startup
        (load_console_prompts() is only ever called once, in
        _build_widgets()). Every other prompt/config path this session
        touched (agent planning/final-answer prompts, extraction bucket
        prompts, pipeline.yaml routing) is already read fresh from disk
        on every call - no caching, no restart needed for those. This
        button exists ONLY to avoid a full app restart (losing the
        resident model + VRAM reload cost) for the narrow case of
        editing/adding a console_prompts YAML file mid-session.

        Preserves the current selection by display_name if it still
        exists after reload; clears the selection if that prompt was
        renamed/removed, rather than silently pointing at a stale index.
        """
        previous_selection = self.prompt_picker_var.get()
        self._console_prompts = load_console_prompts()
        display_names = [p["display_name"] for p in self._console_prompts.values()]
        self.prompt_picker["values"] = display_names
        if previous_selection in display_names:
            self.prompt_picker_var.set(previous_selection)
        else:
            self.prompt_picker_var.set("")
        self.status_label.config(text=f"Reloaded {len(display_names)} console prompt(s).")

    def _load_selected_console_prompt(self) -> None:
        selected_display_name = self.prompt_picker_var.get()
        if not selected_display_name:
            return
        match = next(
            (p for p in self._console_prompts.values()
             if p["display_name"] == selected_display_name),
            None,
        )
        if match is None:
            return
        self.system_prompt_box.delete("1.0", END)
        self.system_prompt_box.insert("1.0", match["system_prompt"])

    # ------------------------------------------------------------------
    # Model / session lifecycle
    # ------------------------------------------------------------------

    def _on_model_change(self) -> None:
        # Explicit design choice (2026-08-11, conversation context): a
        # fresh ChatSession here means switching models ALWAYS starts a
        # new conversational context boundary - prior turns stay in their
        # own session's log, but are never fed as history to a different
        # model. This is deliberate, not an oversight: history is
        # reconstructed from logical role/text/image turns each call
        # (never a persisted model-specific KV-cache - see base_loader.py's
        # _run_generate_with_history docstring), so there's no technical
        # reason history COULDN'T cross a model switch, but doing so would
        # let one model's chat-template quirks/tokenization silently leak
        # into another's context - safer and more predictable to draw the
        # line at "one model = one conversation" for MVP.
        model_name = self.model_var.get()
        self.session = ChatSession(model_name=model_name)
        self._populate_overrides_from_model(model_name)
        self.status_label.config(text=f"Selected: {model_name} (not loaded until first Send)")

    def _on_internet_access_toggle(self) -> None:
        network_gate.set_enabled(self.internet_access_var.get())

    def _on_backend_change(self) -> None:
        """
        Swaps this tab's adapter between the local (Windows in-process)
        and remote (WSL HTTP) backends - 2026-08-13, multi-runtime
        integration. Releases the OUTGOING backend's model first so the
        GPU/service isn't left holding something the UI no longer
        tracks, then constructs a fresh ChatBackendAdapter for the new
        backend. Session/transcript state is deliberately untouched -
        the backend is a transport choice, not a conversation boundary
        (unlike _on_model_change(), which IS one - see its docstring).
        Blocked mid-send for the same reason _on_eject() is.
        """
        selected = self.backend_var.get()
        if selected == self.adapter.backend:
            return
        if self._sending:
            messagebox.showinfo("Busy", "Can't switch backend while a response is being generated.")
            self.backend_var.set(self.adapter.backend)
            return
        try:
            self.adapter.release()
        except Exception as e:
            print(f"[model_console.chat_tab] WARNING: release on backend switch failed: {e}")
        self.adapter = ChatBackendAdapter(backend=selected)
        self.status_label.config(text=f"Backend: {selected} (no model loaded)")

    def _on_eject(self) -> None:
        """
        Releases the resident model (adapter.release() - same teardown
        sequence as model_assessment.py's _release_model()) without
        switching models or closing the app, so VRAM can be freed
        between test runs on demand. Blocked while a send is in flight
        (self._sending) - releasing loader.model/tokenizer/processor
        out from under a background-thread generate() call still using
        them would crash that thread, not just no-op safely.
        """
        if self._sending:
            messagebox.showinfo("Busy", "Can't eject while a response is being generated.")
            return
        if self.adapter.resident_model_name is None:
            self.status_label.config(text="Nothing loaded to eject.")
            return
        ejected_name = self.adapter.resident_model_name
        self.adapter.release()
        self.status_label.config(text=f"Ejected {ejected_name!r} - VRAM released. "
                                       f"Not loaded until next Send.")

    def _populate_overrides_from_model(self, model_name: str) -> None:
        """
        Fills every override field with THIS model's real config/models/
        <name>.yaml value (2026-08-10 fix - blank fields meant "use model
        default" correctly, but that default was invisible in the UI, so
        a genuinely neutral default like gemma.yaml's repetition_penalty:
        1.0 looked indistinguishable from "not set yet" and only surfaced
        as a real repetition-loop failure during actual use). Fields now
        always show the effective value, editable from there - WYSIWYG
        instead of an invisible fallback.
        """
        if model_name not in list_model_profiles():
            return
        config = build_config(model_name)
        for field_name, var in self._override_entries.items():
            value = getattr(config, field_name)
            # no_repeat_ngram_size (and other Optional[int] fields) default
            # to null in several yaml configs (gemma, internvl3_8b, qwen2b/
            # qwen3b) - showing that as the literal string "None" would
            # both look wrong and fail to parse as a number if left
            # untouched. Blank means the same "don't override, use
            # whatever's loaded" thing None does, so use blank instead.
            if value is None:
                var.set("")
            elif isinstance(value, bool):
                var.set(str(value).lower())
            else:
                var.set(str(value))

    def _reset_overrides(self) -> None:
        self._populate_overrides_from_model(self.model_var.get())

    def _collect_overrides(self) -> dict:
        overrides = {}
        for field_name, var in self._override_entries.items():
            raw = var.get().strip()
            if raw == "":
                continue
            if field_name == "do_sample":
                overrides[field_name] = raw.lower() in ("1", "true", "yes")
                continue
            try:
                overrides[field_name] = int(raw) if "." not in raw else float(raw)
            except ValueError:
                raise ValueError(f"'{field_name}' must be a number, got {raw!r}")
        return overrides

    # ------------------------------------------------------------------
    # Image attach
    # ------------------------------------------------------------------

    def _pick_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Attach image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All files", "*.*")],
        )
        if not path:
            return
        self._pending_image_path = Path(path)
        try:
            with Image.open(path) as preview_src:
                preview = preview_src.copy()
            preview.thumbnail(THUMBNAIL_SIZE)
            self._pending_image_thumb = ImageTk.PhotoImage(preview)
            self.image_thumb_label.config(image=self._pending_image_thumb)
            self.image_name_label.config(text=Path(path).name)
        except Exception as e:
            messagebox.showerror("Image error", f"Could not open image:\n{e}")
            self._pending_image_path = None

    def _clear_image(self) -> None:
        self._pending_image_path = None
        self._pending_image_thumb = None
        self.image_thumb_label.config(image="")
        self.image_name_label.config(text="(no image attached)")

    # ------------------------------------------------------------------
    # Transcript
    # ------------------------------------------------------------------

    @staticmethod
    def _format_tool_result(tool_result) -> str:
        """One field per line, not raw JSON - readable in a transcript
        widget the same way the plan/rationale lines are. tool_result is
        a core.agent_tools.schema.ToolResultBlock; may be None if
        dispatch itself never ran (shouldn't happen for action=call_tool,
        but handled defensively rather than assumed)."""
        if tool_result is None:
            return "(no tool result)"
        if not tool_result.success:
            err = tool_result.error or {}
            return f"FAILED ({err.get('error_class', 'unknown')}): {err.get('message', '')}"
        lines = [f"{key}: {value}" for key, value in (tool_result.result or {}).items()]
        return "\n".join(lines) if lines else "(empty result)"

    def _append_transcript(self, role: str, text: str, model_name: Optional[str] = None,
                            has_image: bool = False, telemetry_summary: Optional[str] = None) -> None:
        """
        model_name (2026-08-12 QoL addition) - shown on assistant/error
        turns so it's visible which model actually produced a given
        reply without cross-referencing the session log, especially
        useful once a session has switched models or mixed one-shot/
        context-mode turns. has_image marks a user turn that had an
        image attached, since "(image only)" reuse of a persisted
        system prompt (see earlier session discussion) makes the
        transcript's own text alone ambiguous about whether an image
        was really sent that turn. telemetry_summary (Phase 3, 2026-08-12)
        - a one-line VRAM/timing summary shown in a dim tag under an
        assistant reply when the resident loader captured it (only
        GemmaLoader does today - see base_loader.py's
        last_inference_telemetry); None (most loaders, or before any
        real call happens) adds nothing.
        """
        self.transcript.config(state=NORMAL)
        tag = {"user": "role_user", "assistant": "role_assistant", "agent": "role_agent",
               "tool": "role_tool"}.get(role, "role_error")
        label = f"[{role}"
        if model_name:
            label += f" - {model_name}"
        label += "]"
        if has_image:
            label += " [image attached]"
        self.transcript.insert(END, f"{label} ", (tag,))
        self.transcript.insert(END, f"{text}\n")
        if telemetry_summary:
            self.transcript.insert(END, f"    {telemetry_summary}\n", ("telemetry",))
        self.transcript.insert(END, "\n")
        self.transcript.config(state=DISABLED)
        self.transcript.see(END)

    @staticmethod
    def _format_telemetry_summary(telemetry: Optional[dict]) -> Optional[str]:
        """
        Compact one-line rendering of the Phase 3 telemetry dict (see
        model_console/adapter.py's send_turn()) for the transcript -
        "for directly comparing models" per the original ask, so this
        favors a dense side-by-side-readable line over a nicer multi-
        line layout. Full detail always still lands in the session log
        JSON (session_log.py) regardless of this formatting.
        """
        if not telemetry:
            return None
        parts = []
        load = telemetry.get("load")
        if load:
            parts.append(
                f"load: {load.get('after_load_vram_mb')}MB "
                f"(+{load.get('load_time_s')}s)"
            )
        gen = telemetry.get("generate")
        if gen:
            parts.append(
                f"prefill: {gen.get('vram_after_prefill_mb')}MB/{gen.get('prefill_time_s')}s | "
                f"peak: {gen.get('peak_vram_mb')}MB | "
                f"gen: {gen.get('generated_tokens')}tok/{gen.get('generation_time_s')}s "
                f"({gen.get('tokens_per_sec')}tok/s) | stop: {gen.get('stop_reason')}"
                + (f" | kv: {gen.get('cache_implementation')}" if gen.get("cache_implementation") else "")
            )
        return "  ".join(parts) if parts else None

    # ------------------------------------------------------------------
    # Send
    # ------------------------------------------------------------------

    def _on_send(self) -> None:
        if self._sending:
            return
        model_name = self.model_var.get()
        if model_name not in list_model_profiles():
            messagebox.showwarning("No model", "Pick a model profile first.")
            return

        prompt_text = self.input_box.get("1.0", "end-1c").strip()
        if not prompt_text and self._pending_image_path is None:
            return
        if self.adapter.backend == "local" and model_requires_remote_backend(model_name):
            # Multi-runtime pre-check (2026-08-13): vLLM-runtime models
            # and WSL-only-dependency models (see adapter.py's
            # model_requires_remote_backend()) cannot run in-process on
            # Windows at all - stop here with a clear pointer to the
            # Backend selector instead of letting the send fail after a
            # wasted attempt (same before-the-cold-load principle as the
            # modality pre-checks below).
            messagebox.showwarning(
                "Remote backend required",
                f"{model_name!r} can only run on the WSL compute backend "
                "(vLLM runtime or WSL-only dependencies) - switch the "
                "Backend selector to 'remote' first, with the WSL agent "
                "backend server running.",
            )
            return
        if self._pending_image_path is None and not model_supports_text_only(model_name):
            # Per-model check, not a blanket rule (2026-08-10 finding -
            # see adapter.py's model_supports_text_only()): some models
            # DO support this, confirmed by real generation calls; this
            # one's config just doesn't declare it. Stop here with a
            # clear message instead of loading the model and hitting a
            # per-loader crash/rejection.
            messagebox.showwarning(
                "Image required",
                f"{model_name!r}'s config does not declare text-only support "
                "(config/models/<name>.yaml's text_only_supported is False or "
                "unconfirmed). Attach an image, or pick a model that supports "
                "text-only chat.",
            )
            return
        if (self._pending_image_path is not None and not model_supports_image_input(model_name)
                and not self.agent_mode_var.get()):
            # Mirror-image of the check above (2026-08-13 finding - a
            # real live error: qwen_research_text has no vision tower at
            # all, so attaching an image to it isn't just unnecessary,
            # it's impossible - see base_loader.py's GenerationConfig.
            # image_input_supported). Stop here, before a multi-minute
            # cold model load, instead of surfacing the loader's own
            # RuntimeError only after that load completes.
            #
            # Agent mode is EXEMPT from this block (2026-08-14
            # text-only-brain policy): in agent mode the dropdown model
            # isn't what looks at the image anyway - the turn's chat/
            # planning model is always the research LLM, and vision
            # happens inside tools (extract_fields' borrowed extraction
            # model, the planner's borrowed category-classify VLM - see
            # core/agent_tools/planner.py). Plain chat mode has no such
            # indirection (the picker is a literal "run exactly this
            # model" choice there), so it still blocks.
            messagebox.showwarning(
                "No vision support",
                f"{model_name!r} has no vision tower (config/models/<name>.yaml's "
                "image_input_supported is False) - it cannot accept an attached "
                "image. Remove the image, pick a model with vision support, or "
                "turn on agent mode (where vision is handled by tools).",
            )
            return

        try:
            overrides = self._collect_overrides()
        except ValueError as e:
            messagebox.showerror("Invalid override", str(e))
            return

        system_prompt = self.system_prompt_box.get("1.0", "end-1c")

        chat_image = None
        if self._pending_image_path is not None:
            try:
                chat_image = prep_for_chat(self._pending_image_path)
            except Exception as e:
                messagebox.showerror("Image error", f"Could not prepare image:\n{e}")
                return

        # Snapshot history BEFORE adding this turn's own user message to
        # the session - conversational_turns() must not include the
        # message currently being sent (that's passed separately as
        # prompt_text). context_enabled=False -> history stays None,
        # taking adapter.send_turn()'s untouched one-shot path exactly as
        # before conversation context existed.
        context_enabled = self.context_var.get()
        history_snapshot = self.session.conversational_turns() if context_enabled else None

        user_turn = ChatTurn(role="user", text=prompt_text, image=chat_image)
        self.session.add_turn(user_turn)
        session_log.write_turn(self.session, user_turn)
        self._append_transcript("user", prompt_text or "(image only)",
                                 has_image=chat_image is not None)

        self.input_box.delete("1.0", END)
        self._clear_image()

        self._sending = True
        self.send_button.config(state=DISABLED, text="Sending...")
        self.status_label.config(text=f"Loading/running {model_name}...")

        pil_image = chat_image.pil_image if chat_image else None
        image_path = str(chat_image.source_path) if chat_image and chat_image.source_path else None
        if chat_image is not None:
            # session_log.write_turn() above already persisted this image to
            # disk (<session_dir>/turns/<basename>_image.png), and nothing
            # downstream ever reads a past turn's pil_image back out of
            # session.turns - build_history_for_context()/_serialize_history()
            # only check `turn.image is not None` for the text marker, never
            # touch .pil_image (see adapter.py's _IMAGE_OMITTED_NOTE). Without
            # this release, every image ever attached in a session stays
            # fully decompressed in memory for the session's whole lifetime -
            # confirmed as the real cause of a ~9GB idle Windows RSS reading
            # with no model loaded (2026-08-14). `pil_image` above already
            # holds the reference this turn's send actually needs, so this
            # is safe to clear immediately.
            chat_image.pil_image = None
        agent_mode = self.agent_mode_var.get()
        thread = threading.Thread(
            target=self._worker,
            args=(model_name, overrides, system_prompt, prompt_text, pil_image,
                  history_snapshot, agent_mode, image_path),
            daemon=True,
        )
        thread.start()
        self.after(100, self._poll_result_queue)

    def _worker(self, model_name: str, overrides: dict, system_prompt: str,
                prompt_text: str, pil_image, history_snapshot, agent_mode: bool,
                image_path: Optional[str]) -> None:
        try:
            if agent_mode:
                # 2026-08-14 (Jon's text-only-brain policy: "default
                # qwen 7b as the transformers model, if it needs VLM it
                # can call an agent"): agent mode ALWAYS uses the
                # dedicated research LLM as the turn's chat/planning/
                # synthesis model, image or no image - vision work
                # happens inside tools (extract_fields borrows its
                # extraction model; the planner's category classify
                # borrows a VLM - see core/agent_tools/planner.py),
                # never by making the chat model itself a VLM. This
                # supersedes both earlier special cases below (text-only
                # -> research LLM swap; image + no-vision-model ->
                # DEFAULT_VISION_MODEL_NAME swap) with one uniform rule,
                # and eliminates the confirmed runtime-thrash failure
                # (2026-08-14: a vLLM chat model + transformers tools
                # alternated GPU owners 3-4x in one turn, ~15+ min).
                from core.agent_tools.research_llm import MODEL_NAME as research_model_name
                from core.loaders.base_loader import load_model_config as load_research_config
                config = load_research_config(research_model_name)
                self.adapter.ensure_loaded(research_model_name, config)
            else:
                config = build_config(model_name, overrides=overrides)
                self.adapter.ensure_loaded(model_name, config)

            # Phase 3: rolling summarization. Only relevant when context
            # mode is on (history_snapshot is not None) - one-shot stays
            # untouched. maybe_summarize_session() mutates self.session
            # in place (summary_text/summarized_up_to_turn_id) and costs
            # one extra model call, but only once the unsummarized
            # backlog actually crosses the token trigger - most turns
            # this is a cheap no-op check, not a call.
            if history_snapshot is not None:
                maybe_summarize_session(self.adapter, self.session, history_snapshot)
                system_prompt, history_snapshot = get_effective_context(
                    self.session, history_snapshot, system_prompt)

            if agent_mode:
                # run_agent_chat_turn runs 2-3 model calls (plan, optional
                # tool, answer) against the same resident adapter/loader -
                # see core/agent_tools/planner.py. Any of those calls can
                # raise (e.g. CUDA OOM on the planning turn itself); the
                # except below catches all of it the same as the non-agent
                # path. image_path is the attached image's REAL on-disk
                # path (chat_image.source_path) - the planner never trusts
                # a model-guessed path for tool args, see planner.py's
                # run_agent_turn docstring.
                result = run_agent_chat_turn(
                    self.adapter, prompt_text, pil_image, system_prompt,
                    history=history_snapshot, image_path=image_path)
                self._result_queue.put(("ok_agent", result, None))
            else:
                raw_output, meta = self.adapter.send_turn(
                    prompt_text, pil_image, system_prompt, history=history_snapshot)
                self._result_queue.put(("ok", raw_output, meta))
        except Exception as e:
            self._result_queue.put(("error", str(e), traceback.format_exc()))

    def _poll_result_queue(self) -> None:
        try:
            kind, payload, extra = self._result_queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_result_queue)
            return

        self._sending = False
        self.send_button.config(state=NORMAL, text="Send")

        if kind == "ok_agent":
            result = payload  # AgentTurnResult
            # Phase 4 (2026-08-13): result.steps is the FULL chain this
            # turn took, not just one plan+tool - render every step in
            # order so a multi-tool chain (classify_document ->
            # extract_fields -> ...) is visible, same reasoning as the
            # single-tool case: showing only the LAST step would hide
            # whether an earlier step in the chain went wrong even
            # though the turn as a whole still produced an answer.
            for i, step in enumerate(result.steps, 1):
                plan = step.plan
                prefix = f"[step {i}/{len(result.steps)}] " if len(result.steps) > 1 else ""
                if plan.action == "call_tool":
                    tool_status = "ok" if (step.tool_result and step.tool_result.success) else "FAILED"
                    self._append_transcript(
                        "agent",
                        f"{prefix}[plan: call_tool={plan.tool_name} ({tool_status})] {plan.rationale}",
                    )
                    # Show what the tool actually returned - not just
                    # whether it succeeded - so it's possible to eyeball
                    # whether Gemma's final answer (below) actually
                    # tracks the tool's own numbers rather than
                    # re-deriving/re-guessing them from the image on its
                    # own. Per Jon's direction (2026-08-13): the whole
                    # point of wiring a tool result into the final-answer
                    # prompt is defeated if there's no way to see whether
                    # the model actually used it.
                    self._append_transcript("tool", self._format_tool_result(step.tool_result))
                else:
                    self._append_transcript("agent", f"{prefix}[plan: {plan.action}] {plan.rationale}")

            meta = result.answer_meta or result.planner_meta_last or {}
            assistant_turn = ChatTurn(
                role="assistant",
                text=result.final_answer,
                raw_output=result.final_answer,
                generation_config_snapshot=meta.get("generation_config_snapshot"),
                model_name=meta.get("model_name"),
                profile_name=self.session.profile_name,
                runtime_seconds=meta.get("runtime_seconds"),
                context_enabled=meta.get("context_enabled"),
                history_turn_ids=meta.get("history_turn_ids"),
                history_turn_count=meta.get("history_turn_count"),
                approx_input_tokens=meta.get("approx_input_tokens"),
                dropped_turn_ids=meta.get("dropped_turn_ids"),
                telemetry=meta.get("telemetry"),
            )
            self.session.add_turn(assistant_turn)
            session_log.write_turn(self.session, assistant_turn)
            self._append_transcript("assistant", result.final_answer, model_name=meta.get("model_name"),
                                     telemetry_summary=self._format_telemetry_summary(meta.get("telemetry")))
            self.status_label.config(
                text=f"Loaded: {self.adapter.resident_model_name} (agent turn done)")
        elif kind == "ok":
            raw_output, meta = payload, extra
            assistant_turn = ChatTurn(
                role="assistant",
                text=raw_output,
                raw_output=raw_output,
                generation_config_snapshot=meta.get("generation_config_snapshot"),
                model_name=meta.get("model_name"),
                profile_name=self.session.profile_name,
                runtime_seconds=meta.get("runtime_seconds"),
                context_enabled=meta.get("context_enabled"),
                history_turn_ids=meta.get("history_turn_ids"),
                history_turn_count=meta.get("history_turn_count"),
                approx_input_tokens=meta.get("approx_input_tokens"),
                dropped_turn_ids=meta.get("dropped_turn_ids"),
                telemetry=meta.get("telemetry"),
            )
            self.session.add_turn(assistant_turn)
            session_log.write_turn(self.session, assistant_turn)
            self._append_transcript("assistant", raw_output, model_name=meta.get("model_name"),
                                     telemetry_summary=self._format_telemetry_summary(meta.get("telemetry")))
            self.status_label.config(
                text=f"Loaded: {self.adapter.resident_model_name} "
                     f"({meta.get('runtime_seconds', 0):.1f}s last turn)")
        else:
            error_message, tb = payload, extra
            error_turn = ChatTurn(role="assistant", text="", error=error_message,
                                   model_name=self.model_var.get())
            self.session.add_turn(error_turn)
            session_log.write_turn(self.session, error_turn)
            self._append_transcript("error", error_message, model_name=error_turn.model_name)
            self.status_label.config(text="ERROR - see transcript")
            print(f"[model_console.chat_tab] ERROR: {tb}")

    def shutdown(self) -> None:
        """Called on window close - releases the resident model so
        VRAM isn't held by a closed window's adapter."""
        self.adapter.release()
