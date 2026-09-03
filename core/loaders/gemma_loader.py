"""
Gemma loader — currently used only for the classification stage.

NOTE: model_id/exact HF repo, processor class, and reasoning-toggle
mechanism below are placeholders pending confirmation of the exact
Gemma vision-capable checkpoint you're using (the LM Studio testing
referenced "google/gemma-4-e2b" — confirm the equivalent HF repo id
and whether it ships as AutoModelForImageTextToText or a
model-specific class before running this for real).

Parsing strategy: key:value lines rather than JSON. Smaller models
were observed throughout testing to wrap JSON in markdown fences,
add trailing commentary after the closing brace, or emit malformed
JSON under any output-length pressure. A line-based key:value parser
degrades more predictably and makes partial-failure detection
(missing required key) simpler than a JSON parse try/except.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from transformers.generation.streamers import BaseStreamer

from core.loaders.base_loader import BaseLoader, GenerationConfig
from core.schema import ClassificationResult, ExtractionResult, DocumentCategory


REQUIRED_CLASSIFY_KEYS = {"category", "confidence", "reason"}

# --debug-only vision-capture persistence (2026-08-05). Dev/debug tool
# only, per direct instruction - never wired into any production default,
# never written to unless --debug was passed to build_classifier_loader().
# docs/GEMMA_INSTRUMENTATION_AND_SENSOR_SURVEY.md §1.10's abstraction
# sketch is now partially applied here (persistence added; the sketch's
# own standalone GemmaVisionInstrumentation class was NOT extracted -
# kept as GemmaLoader methods instead, since this loader is the only
# caller and a separate class would be an abstraction with one user).
DEFAULT_VISION_DEBUG_CAPTURE_DIR = Path("data/outputs/gemma_vision_debug_captures")

# Verified hook locations - docs/GEMMA_INSTRUMENTATION_AND_SENSOR_SURVEY.md
# Part 1 (§1.2/§1.3), confirmed against the installed transformers'
# modeling_gemma4.py directly, not assumed from Gemma 3's shape. The
# ".model." indirection is required - Gemma4ForConditionalGeneration has
# no top-level vision_tower/embed_vision convenience properties, only
# self.model (a Gemma4Model) does.
_VISION_HOOK_TARGETS = {
    "encoder": "model.model.vision_tower.encoder",     # Candidate 1 - pre-pooling patch tokens
    "pooled": "model.model.vision_tower",               # Candidate 2 - post-pool, post-standardize
    "projected": "model.model.embed_vision",             # Candidate 3 - projected into LM embedding space
}


class _FirstGeneratedTokenStreamer(BaseStreamer):
    """
    Records wall-clock time + CUDA memory_allocated() the moment the
    FIRST genuinely generated token exists - used by
    _generate_from_messages() (Phase 3 inference telemetry, 2026-08-12)
    as a "prefill just finished" marker, without altering generation
    behavior (a streamer's put()/end() return values are never consumed
    by generate()).

    Verified directly against the installed transformers==5.12.1
    generate() source (transformers/generation/utils.py) before writing
    this: `streamer.put(...)` is called exactly twice before any real
    per-step token: once with the full prompt `input_ids` BEFORE the
    decode loop starts (`streamer.put(input_ids.cpu())`), and then once
    per decode step with only the newly generated token(s)
    (`streamer.put(next_tokens.cpu())` / `valid_tokens.cpu()`). The
    first call is therefore an echo of the prompt, not a signal that any
    forward pass has run - the SECOND call is the first one that fires
    only after the model's first forward pass (prefill + first decode
    step) has actually completed. This class ignores call #1 and
    captures on call #2, rather than naively trusting the first put()
    call the way a generic example might.
    """

    def __init__(self) -> None:
        self._call_count = 0
        self.first_token_time: Optional[float] = None
        self.vram_after_prefill_mb: Optional[float] = None

    def put(self, value) -> None:
        self._call_count += 1
        if self._call_count == 2 and self.first_token_time is None:
            self.first_token_time = time.perf_counter()
            if torch.cuda.is_available():
                self.vram_after_prefill_mb = round(torch.cuda.memory_allocated() / 1e6, 1)

    def end(self) -> None:
        pass


class GemmaLoader(BaseLoader):
    def initialize_model_and_tokenizer(self) -> tuple[Any, Any, Any]:
        # Correction from earlier placeholder: Gemma-4 uses
        # AutoModelForCausalLM per the actual model card, not
        # AutoModelForImageTextToText or AutoModelForMultimodalLM as
        # previously guessed. Also note the -it (instruction-tuned)
        # suffix on the repo id - the base (non-it) checkpoint is not
        # what you want for prompted extraction/classification tasks.
        from transformers import AutoProcessor, AutoModelForCausalLM

        processor = AutoProcessor.from_pretrained(
            self.config.repo_id,
            token=False,
        )

        # Config-driven, not hardcoded to any one model: extra.load_in_8bit
        # (default off) lets a specific config opt into bitsandbytes 8-bit
        # loading - e.g. config/models/gemma_e4b.yaml, where the raw bf16
        # weights (16.02GB) exceed this card's total VRAM (15.93GB) on
        # their own. dtype="auto" is omitted when quantizing: bitsandbytes
        # manages compute precision for the quantized layers itself, and
        # passing both together is redundant/conflicting.
        load_in_8bit = self.config.extra.get("load_in_8bit", False)
        model_kwargs = dict(device_map="auto", token=False)
        if load_in_8bit:
            from transformers import BitsAndBytesConfig
            # Real bug caught before trusting any E2B-vs-E4B result (2026-08-02):
            # quantizing the WHOLE model - including vision_tower/embed_vision
            # (confirmed real module names via modeling_gemma4.py, not guessed) -
            # produced coherent-but-wrong output ("the image is largely blank",
            # low confidence) on images the same model reads correctly at bf16.
            # This is a known failure mode for naive int8 quantization of
            # multimodal models: the vision/projector path is more precision-
            # sensitive than the text decoder. Skipping these modules keeps
            # them in full precision while still 8-bit quantizing the LLM
            # backbone - the actual VRAM-saving target.
            # lm_head added after a second real bug: transformers normally
            # auto-detects and skips tied-weight output heads like lm_head
            # on its own, but supplying an explicit llm_int8_skip_modules
            # list REPLACES that default detection rather than extending
            # it - without lm_head listed explicitly here, generation
            # crashed with AttributeError: 'Parameter' object has no
            # attribute 'CB' (lm_head left as a bare, non-bitsandbytes-
            # wrapped Parameter that the quantized forward path still
            # tried to call).
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=["vision_tower", "embed_vision", "audio_tower", "embed_audio", "lm_head"],
            )
        else:
            model_kwargs["dtype"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            self.config.repo_id,
            **model_kwargs,
        ).eval()

        self._execution_meta = {
            "device_map": "auto",
            "max_memory": self.config.build_max_memory_map(),
            "vram_headroom_gb": self.config.vram_headroom_gb,
            "cpu_offload_limit_gb": self.config.cpu_offload_limit_gb,
        }
        self._oom_recovered = False

        self.model = model
        self.processor = processor
        # Previously left as the BaseLoader default (None) - real gap
        # found 2026-07-24 wiring restrict_output_charset support: every
        # OTHER loader in this project sets self.tokenizer = processor.
        # tokenizer, and _maybe_add_charset_logits_processor() (base_
        # loader.py) needs it. Brought in line rather than special-cased.
        self.tokenizer = processor.tokenizer

        # Vision instrumentation state (docs/GEMMA_INSTRUMENTATION_AND_
        # SENSOR_SURVEY.md Part 1) - always initialized so every method
        # below can assume these attributes exist, but hooks are only
        # actually REGISTERED (see enable_vision_debug_instrumentation())
        # when debug mode was requested before load. self._debug_mode is
        # set by core.classifier.build_classifier_loader() on the instance
        # BEFORE this method runs, when --debug is passed; absent for any
        # other caller (getattr default False), so this is a pure no-op
        # for every existing call site that doesn't opt in.
        self._vision_hooks: list = []
        self._vision_debug_enabled: bool = False
        self._captured_vision: dict[str, dict] = {}
        self._vision_hook_counts: dict[str, int] = {}
        self._vision_context: dict = {}
        if getattr(self, "_debug_mode", False):
            self.enable_vision_debug_instrumentation()

        return model, self.tokenizer, processor

    # -- vision instrumentation (--debug only) ------------------------------

    def enable_vision_debug_instrumentation(self) -> None:
        """
        Registers forward hooks on the three verified Gemma-4 vision
        submodules (_VISION_HOOK_TARGETS above) so classify() can be
        observed while it runs, with zero duplicate vision computation -
        the hooks fire as part of the existing model.generate() ->
        forward() call, never a second forward pass. See
        docs/GEMMA_INSTRUMENTATION_AND_SENSOR_SURVEY.md Part 1 for the
        verified module hierarchy and call-path evidence this relies on.

        Idempotent - safe to call more than once against the same loaded
        model (e.g. once automatically at load time, once manually) without
        double-registering hooks. A fresh initialize_model_and_tokenizer()
        call (a genuine reload) always starts self._vision_hooks empty
        again, so a load -> release -> reload cycle never accumulates
        hooks from the PREVIOUS model instance either - there is no
        previous instance left to accumulate on, each reload's hooks
        attach to a brand new self.model object.
        """
        if self.model is None:
            raise RuntimeError(
                f"{type(self).__name__}.enable_vision_debug_instrumentation() "
                "called before initialize_model_and_tokenizer() has set a real "
                "self.model."
            )
        if self._vision_hooks:
            self._vision_debug_enabled = True
            return

        inner = self.model.model  # Gemma4Model - see _VISION_HOOK_TARGETS's ".model." note
        modules = {
            "encoder": inner.vision_tower.encoder,
            "pooled": inner.vision_tower,
            "projected": inner.embed_vision,
        }
        for name, module in modules.items():
            handle = module.register_forward_hook(
                self._make_vision_hook(name, _VISION_HOOK_TARGETS[name])
            )
            self._vision_hooks.append(handle)
        self._vision_debug_enabled = True

    def _make_vision_hook(self, name: str, module_path: str):
        """
        Returns a forward-hook function that captures metadata + a
        detached CPU copy of the module's output, then returns nothing -
        a hook returning None leaves the real forward pass's output
        completely untouched, so this cannot alter classification
        behavior. Per the tensor-handling requirement: detach, move to
        CPU, and drop the GPU reference immediately - no tensor is ever
        retained on GPU or accumulated across images (self._captured_vision
        is reset at the start of every classify() call, see
        _reset_vision_capture()).
        """

        def _hook(module, args, output):
            source = output.last_hidden_state if hasattr(output, "last_hidden_state") else output
            original_device = str(source.device)
            tensor = source.detach().to("cpu")

            captured_at = datetime.now().isoformat(timespec="milliseconds")
            meta = {
                "module_path": module_path,
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": original_device,
                "size_bytes": tensor.element_size() * tensor.nelement(),
                "captured_at": captured_at,
            }
            self._captured_vision[name] = {"tensor": tensor, "meta": meta}
            self._vision_hook_counts[name] = self._vision_hook_counts.get(name, 0) + 1

            file_path = self._vision_context.get("file_path", "?")
            print(
                f"  [vision-hook:{name}] file={file_path} module={module_path} "
                f"shape={meta['shape']} dtype={meta['dtype']} device={meta['device']} "
                f"bytes={meta['size_bytes']} t={captured_at} "
                f"(fire #{self._vision_hook_counts[name]})"
            )
            # No return value - do not alter the tensor flowing downstream.

        return _hook

    def _reset_vision_capture(self, file_path: str) -> None:
        self._captured_vision = {}
        self._vision_hook_counts = {}
        self._vision_context = {"file_path": file_path}

    def _validate_vision_capture(self, file_path: str) -> None:
        """
        Warns (never raises - a debug-only observability check must not
        take down a real classification run) if any hook is missing,
        fired more than once, or produced a shape inconsistent with the
        model's own config. Expected token count is read from
        config.extra.image_token_budget (this pipeline's real setting,
        not a hardcoded literal); expected hidden dims are read from
        self.model.config's own vision_config/text_config at runtime,
        per docs/GEMMA_INSTRUMENTATION_AND_SENSOR_SURVEY.md §1.8's
        "introspect rather than hardcode" recommendation, so this still
        gives a real check if this pipeline's config ever points at a
        different Gemma variant.

        image_token_budget is a CAP, not a promise of an exact count -
        per docs/GEMMA_TOKEN_BUDGET_INVESTIGATION.md's Phase 4 finding,
        aspect-ratio-preserving resize/pooling means the real per-image
        pooled/projected token count is `<= image_token_budget`, and
        varies by image. Checked as an upper bound below, not equality -
        the original `!=` check here would have falsely flagged nearly
        every real image even after the call-site bug above is fixed.
        """
        expected_tokens = self.config.extra.get("image_token_budget", 280)
        vision_config = getattr(self.model.config, "vision_config", None)
        text_config = getattr(self.model.config, "text_config", None)
        expected_vision_hidden = getattr(vision_config, "hidden_size", None)
        expected_text_hidden = getattr(text_config, "hidden_size", None)

        problems = []
        for name in _VISION_HOOK_TARGETS:
            count = self._vision_hook_counts.get(name, 0)
            if count == 0:
                problems.append(f"'{name}' hook did not fire (expected exactly once)")
                continue
            if count > 1:
                problems.append(f"'{name}' hook fired {count} times (expected exactly once)")

            shape = self._captured_vision[name]["meta"]["shape"]
            if name in ("pooled", "projected") and len(shape) >= 2 and shape[-2] > expected_tokens:
                problems.append(
                    f"'{name}' token count {shape[-2]} > configured "
                    f"image_token_budget cap {expected_tokens}"
                )
            if name in ("encoder", "pooled") and expected_vision_hidden and shape[-1] != expected_vision_hidden:
                problems.append(
                    f"'{name}' hidden dim {shape[-1]} != model vision_config."
                    f"hidden_size {expected_vision_hidden}"
                )
            if name == "projected" and expected_text_hidden and shape[-1] != expected_text_hidden:
                problems.append(
                    f"'projected' hidden dim {shape[-1]} != model text_config."
                    f"hidden_size {expected_text_hidden}"
                )

        if problems:
            for p in problems:
                print(f"  WARNING: vision instrumentation ({file_path}): {p}")
        else:
            print(
                f"  [vision-hook] OK for {file_path}: encoder/pooled/projected "
                "each fired exactly once, shapes consistent with model config."
            )

    def vision_instrumentation_status(self) -> dict:
        """
        Read-only diagnostic snapshot for debugging the instrumentation
        itself - hook registration state, the real target module names,
        and the most recent classify() call's captured METADATA ONLY.
        Never includes tensor values or the tensors themselves, per the
        tensor-handling requirement (no logging/exposing tensor contents).
        """
        return {
            "enabled": self._vision_debug_enabled,
            "hooks_registered": len(self._vision_hooks),
            "targets": dict(_VISION_HOOK_TARGETS),
            "last_file": self._vision_context.get("file_path"),
            "fire_counts": dict(self._vision_hook_counts),
            "last_capture_meta": {
                name: entry["meta"] for name, entry in self._captured_vision.items()
            },
        }

    def _persist_vision_capture(self, file_path: str, result: ClassificationResult) -> Path | None:
        """
        Writes the current self._captured_vision (encoder/pooled/
        projected tensors + metadata) to a single per-image .pt file,
        alongside this call's classification result - lets a later
        analysis pass correlate Gemma's internal representations with
        its own routing decision, the whole point of building this.

        --debug ONLY - only ever called from classify() when
        self._vision_debug_enabled is True (see call site below). Never
        touches production data/buckets/*.csv or pipeline.db - writes
        exclusively under DEFAULT_VISION_DEBUG_CAPTURE_DIR, a dedicated
        debug-output directory with no other consumer.

        One file per image, named by stem, OVERWRITING on repeat
        classify() calls against the same file - appropriate for
        iterative interactive debug work (re-running --debug against the
        same test image should reflect the latest capture, not pile up
        stale copies), unlike a production sensor capture which would
        need append/versioning discipline.

        Returns the written path, or None if nothing was captured (e.g.
        a hook didn't fire - _validate_vision_capture() already warned
        about this separately; this method just skips writing rather
        than raising, since a debug capture failure must never take down
        classification itself).
        """
        if not self._captured_vision:
            return None

        DEFAULT_VISION_DEBUG_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        stem = Path(file_path).stem
        out_path = DEFAULT_VISION_DEBUG_CAPTURE_DIR / f"{stem}.pt"

        payload = {
            "file_path": file_path,
            "captured_at": datetime.now().isoformat(timespec="milliseconds"),
            "classification": {
                "category": result.category.value,
                "confidence": result.confidence,
                "reason": result.reason,
            },
            "vision": {
                name: {"tensor": entry["tensor"], "meta": entry["meta"]}
                for name, entry in self._captured_vision.items()
            },
        }
        torch.save(payload, out_path)
        print(f"  [vision-hook] capture saved: {out_path}")
        return out_path

    def classify(self, file_path: str, raw_image: Any) -> ClassificationResult:
        """
        Overrides BaseLoader.classify() only to reset/validate/persist
        vision-hook capture state around the unchanged call - when
        vision debug instrumentation isn't enabled this is exactly one
        boolean check before delegating straight to the base
        implementation, no other added cost, per the debug-gating
        requirement ("no runtime cost beyond a boolean check" when off).
        """
        if not self._vision_debug_enabled:
            return super().classify(file_path, raw_image)

        self._reset_vision_capture(file_path)
        result = super().classify(file_path, raw_image)
        self._validate_vision_capture(file_path)
        self._persist_vision_capture(file_path, result)
        return result

    def release(self) -> None:
        """
        Removes any forward hooks registered by
        enable_vision_debug_instrumentation() and clears captured state.
        core.row_extraction._release_model() (and any equivalent caller)
        invokes this BEFORE nulling self.model/tokenizer/processor - see
        BaseLoader.release()'s own docstring - so hook handles are removed
        while they're still valid, deterministic teardown rather than
        relying on GC of the model object to also drop the hooks.
        """
        for handle in self._vision_hooks:
            handle.remove()
        self._vision_hooks = []
        self._vision_debug_enabled = False
        self._captured_vision = {}
        self._vision_hook_counts = {}
        self._vision_context = {}

    def _build_prompt(self, task: str) -> str:
        if task != "classify":
            raise NotImplementedError(
                "GemmaLoader is currently scoped to classification only. "
                "Extraction role not yet assigned pending re-testing."
            )

        base_prompt = self.config.prompt_text
        if not base_prompt:
            raise ValueError(
                "config.prompt_text is empty — load it from "
                "config/prompts/classifier_classify_v1.txt before calling classify()."
            )
        return base_prompt

    def _build_system_content(self) -> str:
        # System prompt is config-driven, NOT hardcoded here (2026-07-30
        # fix, per Jon's direction, after a real bug: this used to be a
        # hardcoded classifier persona applied to EVERY call through this
        # loader regardless of task - harmless for gemma.yaml's actual
        # classification role, but silently injected into gemma_extract's
        # extraction calls too whenever a caller invokes _run_generate()
        # directly (e.g. benchmark/prompt_sweep.py, which bypasses
        # _build_prompt()'s task gate by design), making an OCR/structuring
        # call behave like "I am a strict router" instead. See docs/
        # CODE_MAP.md's "Architectural principle: config owns behavior,
        # loaders own mechanics" for the full story. Each model config now
        # sets its OWN extra.system_prompt (gemma.yaml keeps the classifier
        # wording, gemma_extract.yaml gets its own extraction-appropriate
        # one) - DELIBERATELY no hardcoded fallback here: a config that
        # doesn't set one gets no system message at all, not an
        # accidentally-inherited personality from whichever config
        # happened to define one first.
        #
        # Reasoning toggle: confirmed mechanism is a <|think|> token at
        # the START of the system prompt, not a generate() kwarg (see
        # gemma.yaml's reasoning_toggle_mechanism comment). Applied here
        # regardless of whether a system_prompt is configured, so the
        # toggle still works even for a config with none set.
        system_content = self.config.extra.get("system_prompt", "")
        if self.config.reasoning_enabled:
            system_content = "<|think|>" + system_content
        return system_content

    def _build_user_message(self, raw_image: Optional[Image.Image], prompt: str) -> dict:
        # Modality order matters per the model card: image content must
        # precede text content in the message, not follow it (this is
        # the reverse of how the Qwen/InternVL loaders in this project
        # structure their content lists - don't copy that pattern here).
        user_content = []
        if raw_image is not None:
            user_content.append({"type": "image", "image": raw_image})
        user_content.append({"type": "text", "text": prompt})
        return {"role": "user", "content": user_content}

    def _run_generate(self, raw_image: Optional[Image.Image], prompt: str) -> str:
        # raw_image=None path added 2026-08-10 for model_console (see
        # base_loader.py's text_only_supported field docstring) - Gemma4
        # is a vision fine-tune of a text-only backbone, and its chat
        # template/processor accept a content list with no image entry.
        # Confirmed by real generation call, not assumed - see
        # config/models/gemma.yaml's text_only_supported flag, only set
        # True after that confirmation.
        if raw_image is not None and raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")

        messages = []
        system_content = self._build_system_content()
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append(self._build_user_message(raw_image, prompt))

        return self._generate_from_messages(messages, raw_image)

    def _run_generate_with_history(
        self, history: list[dict], raw_image: Optional[Image.Image], prompt: str
    ) -> str:
        """
        2026-08-11, model_console conversation context (see base_loader.py's
        _run_generate_with_history docstring for the general contract).
        `history` is TEXT-ONLY (no image entries) - only the CURRENT turn's
        raw_image is ever passed to the two-step processor() call below,
        preserving the same "exactly one image per call" invariant
        _run_generate() already has. Shares _generate_from_messages() with
        _run_generate() rather than duplicating the apply_chat_template/
        gen_kwargs/decode logic.
        """
        if raw_image is not None and raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")

        messages = []
        system_content = self._build_system_content()
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.extend(history)
        messages.append(self._build_user_message(raw_image, prompt))

        return self._generate_from_messages(messages, raw_image)

    def _generate_from_messages(self, messages: list[dict], raw_image: Optional[Image.Image]) -> str:
        """
        Shared tail for _run_generate()/_run_generate_with_history():
        apply_chat_template -> processor() -> generate() -> decode. Takes
        the CURRENT turn's raw_image separately (not re-derived from
        `messages`) because Gemma's two-step pipeline needs it as a
        SINGULAR image passed directly to processor(images=...), not
        extracted from the message content list.
        """
        # Image token budget - real fix applied 2026-08-05, see
        # docs/GEMMA_TOKEN_BUDGET_INVESTIGATION.md. The installed
        # transformers version does NOT read a flat `image_seq_length=`
        # kwarg at call time (that name is only a Gemma4Processor
        # __init__-time default, unrelated to this per-call path) - the
        # real per-call control is `max_soft_tokens`, and it must be
        # passed nested under `images_kwargs`, not as a flat kwarg. The
        # old flat `image_seq_length=140` call was silently ignored,
        # meaning every classify() call actually ran at the processor's
        # own default cap (280), not the configured 140 - confirmed by
        # direct inspection of the installed
        # transformers.models.gemma4.processing_gemma4 source.
        #
        # Per direct confirmation this session: 280 is being kept as the
        # real, intended value, NOT reduced to 140 - dropping to 140
        # measurably crops how much of the image Gemma actually sees.
        # config/models/gemma.yaml (and gemma_12b_unified.yaml/
        # gemma_e4b.yaml, which mirror it) were updated from 140 to 280
        # to reflect this: previously "140" was a config value with no
        # real effect, since 280 is what was always actually happening.
        # This fix makes the config value load-bearing again, so a
        # config change now genuinely changes model behavior instead of
        # silently doing nothing.
        image_token_budget = self.config.extra.get("image_token_budget", 280)

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.config.reasoning_enabled,
        )
        if raw_image is not None:
            inputs = self.processor(
                text=text,
                images=raw_image,
                images_kwargs={"max_soft_tokens": image_token_budget},
                return_tensors="pt",
            ).to(self.model.device)
        else:
            # No images_kwargs/max_soft_tokens - that budget is
            # meaningless with nothing to encode.
            inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        # Deliberate deviation from the model card's recommended sampling
        # config (temperature=1.0, top_p=0.95, top_k=64): those defaults
        # are tuned for open-ended generation quality, not for the
        # deterministic, low-fabrication-risk behavior this pipeline
        # needs. do_sample stays governed by config/models/gemma.yaml
        # (currently false) rather than silently adopting the model
        # card's suggested values. Revisit only if greedy decoding
        # underperforms in actual testing.
        gen_kwargs = dict(
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.do_sample,
        )
        if self.config.do_sample:
            gen_kwargs.update(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )
        if self.config.repetition_penalty and self.config.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = self.config.repetition_penalty
        if self.config.no_repeat_ngram_size:
            gen_kwargs["no_repeat_ngram_size"] = self.config.no_repeat_ngram_size
        # None (default, every config as of this writing) leaves HF's own
        # default cache (DynamicCache) untouched - see base_loader.py's
        # GenerationConfig.cache_implementation docstring for why this
        # isn't defaulted to anything more aggressive, and
        # docs/CODE_MAP.md for what's been empirically verified to work
        # against THIS checkpoint's hybrid sliding-window/full-attention
        # layer pattern vs merely appearing in transformers' generic
        # ALL_CACHE_IMPLEMENTATIONS list.
        if self.config.cache_implementation:
            gen_kwargs["cache_implementation"] = self.config.cache_implementation
        self._maybe_add_charset_logits_processor(gen_kwargs)

        # Phase 3 inference telemetry (2026-08-12) - opt-in, written to
        # self.last_inference_telemetry (base_loader.py's BaseLoader.__init__)
        # for model_console/adapter.py's send_turn() to fold into its
        # meta dict. Kept local to this one call - never accumulated
        # across calls, always reflects only the most recent generate().
        prompt_tokens = input_len
        token_streamer = _FirstGeneratedTokenStreamer()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        generate_start = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs, streamer=token_streamer)
        generate_end = time.perf_counter()

        gen_token_count = outputs.shape[-1] - prompt_tokens
        last_token_id = outputs[0, -1].item() if gen_token_count > 0 else None
        eos_token_id = self.tokenizer.eos_token_id if self.tokenizer is not None else None
        if isinstance(eos_token_id, (list, tuple)):
            hit_eos = last_token_id in eos_token_id
        else:
            hit_eos = last_token_id == eos_token_id
        if hit_eos:
            stop_reason = "eos"
        elif gen_token_count >= self.config.max_new_tokens:
            stop_reason = "max_new_tokens"
        else:
            stop_reason = "other"

        prefill_time_s = (
            (token_streamer.first_token_time - generate_start)
            if token_streamer.first_token_time is not None else None
        )
        generation_time_s = (
            (generate_end - token_streamer.first_token_time)
            if token_streamer.first_token_time is not None
            else (generate_end - generate_start)
        )
        peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1e6, 1) if torch.cuda.is_available() else None
        final_vram_mb = round(torch.cuda.memory_allocated() / 1e6, 1) if torch.cuda.is_available() else None

        self.last_inference_telemetry = {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": gen_token_count,
            "vram_after_prefill_mb": token_streamer.vram_after_prefill_mb,
            "peak_vram_mb": peak_vram_mb,
            "final_vram_mb": final_vram_mb,
            "prefill_time_s": round(prefill_time_s, 4) if prefill_time_s is not None else None,
            "generation_time_s": round(generation_time_s, 4),
            "tokens_per_sec": (
                round(gen_token_count / generation_time_s, 2)
                if generation_time_s and gen_token_count > 0 else None
            ),
            "stop_reason": stop_reason,
            "cache_implementation": self.config.cache_implementation,
        }

        response = self.processor.decode(
            outputs[0][input_len:], skip_special_tokens=False
        )

        # processor.parse_response strips/separates the <|channel>thought
        # block per the model card, when thinking was enabled. When
        # thinking is disabled the channel tags may still be emitted
        # (empty thought block) except for E2B/E4B variants per the card -
        # parse_response handles both cases, so always route through it
        # rather than hand-rolling a strip.
        parsed = self.processor.parse_response(response)
        final_text = parsed.get("content", parsed) if isinstance(parsed, dict) else parsed
        return str(final_text).strip()

    # -- parsing -----------------------------------------------------------

    @staticmethod
    def _parse_kv_block(raw_output: str) -> dict[str, str]:
        """
        Parses 'key: value' lines. Ignores anything before the first
        recognized key (strips leaked reasoning/preamble) and stops at
        the first blank line after keys begin, so trailing commentary
        doesn't get absorbed into a field value.

        Tolerates an optional leading "-"/"*" markdown bullet marker
        before the key (2026-07-25, real bug found via a live
        classification run: Gemma formatted its ENTIRE response as a
        bullet list - "- category: dense_tabular_rows", "- confidence:
        1.0", etc. - and the original whitespace-only prefix regex
        rejected every single line, not just the ones that ended up
        "missing": all 8 fields were present and readable, but zero
        matched, so every field silently failed to parse and the
        image was routed to uncertain_review with a "missing required
        fields" error that named only the 3 REQUIRED keys, masking
        that this was a 100% parse failure, not a partial one).
        """
        result: dict[str, str] = {}
        lines = raw_output.splitlines()
        started = False
        for line in lines:
            match = re.match(r"^\s*[-*]?\s*([a-zA-Z_]+)\s*:\s*(.*)$", line)
            if match:
                key, value = match.group(1).lower(), match.group(2).strip()
                result[key] = value
                started = True
            elif started and line.strip() == "":
                break
        return result

    @staticmethod
    def _to_bool(value: str) -> bool | None:
        v = value.strip().lower()
        if v in ("true", "yes", "1"):
            return True
        if v in ("false", "no", "0"):
            return False
        return None

    @staticmethod
    def _to_float(value: str) -> float | None:
        try:
            return float(value.strip())
        except (ValueError, AttributeError):
            return None

    def _parse_classification(self, file_path: str, raw_output: str) -> ClassificationResult:
        fields = self._parse_kv_block(raw_output)

        missing = REQUIRED_CLASSIFY_KEYS - fields.keys()
        if missing:
            raise ValueError(
                f"Classifier output missing required fields {missing} for "
                f"{file_path}. Raw output: {raw_output[:200]!r}. "
                "This file should be routed to uncertain_review by the caller."
            )

        try:
            category = DocumentCategory(fields["category"].strip())
        except ValueError as e:
            raise ValueError(
                f"Classifier returned unrecognized category "
                f"{fields['category']!r} for {file_path}."
            ) from e

        confidence = self._to_float(fields["confidence"])
        if confidence is None:
            raise ValueError(f"Could not parse confidence from {fields['confidence']!r}")

        return ClassificationResult(
            file_path=file_path,
            category=category,
            confidence=confidence,
            text_density=self._to_float(fields.get("text_density", "")),
            handwriting=self._to_bool(fields.get("handwriting", "")),
            table_layout=self._to_bool(fields.get("table_layout", "")),
            faces=self._to_bool(fields.get("faces", "")),
            map_like=self._to_bool(fields.get("map_like", "")),
            reason=fields["reason"][:280],
            model=self.config.model_name,
            prompt_version=self.config.prompt_version,
        )

    def _parse_extraction(self, file_path: str, category: str, raw_output: str) -> ExtractionResult:
        raise NotImplementedError(
            "GemmaLoader is not currently assigned an extraction role. "
            "See config/pipeline.yaml — extraction buckets are routed "
            "to other models pending re-testing."
        )
