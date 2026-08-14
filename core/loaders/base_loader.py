"""
Shared loader interface for the genealogy extraction pipeline.

Every model (Qwen, Gemma, InternVL, LFM2.5-VL, etc.) gets its own
subclass living in core/loaders/. This base class defines the contract
they must satisfy and centralizes the things that were previously
duplicated/inconsistent across VectorDB-Plugin's loaders:

  - per-model generation config, loaded from YAML, not hardcoded
  - a declared method for toggling "reasoning" per-model, since not
    every model exposes this the same way (system prompt vs. a param
    vs. not supported at all)
  - a generation_config_hash so every output row can be traced back
    to the exact settings that produced it (audit trail requirement
    from the pipeline design)
  - output validated against core.schema before being returned

No fabrication-prevention logic lives here — that's the schema's job
and the anomaly-detection pass's job. This class only standardizes
*how* a model is invoked and *what shape* comes back out.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from core.schema import ExtractionResult, ClassificationResult


CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "models"


@dataclass
class GenerationConfig:
    """
    Mirrors what actually varies per model, based on what mattered in
    testing: repetition_penalty, sampling on/off, resolution/token
    ceilings, and a reasoning toggle whose mechanism differs per model.
    """
    model_name: str
    repo_id: str
    loader_class: str  # e.g. "QwenLoader" — used by the registry to dispatch

    # Optional - only needed when the processor/tokenizer must come from
    # a DIFFERENT repo than the model weights. Confirmed necessary for
    # olmOCR: it's a fine-tune of Qwen2.5-VL-7B-Instruct, and per the
    # official model card, AutoProcessor.from_pretrained() must load from
    # the original "Qwen/Qwen2.5-VL-7B-Instruct" repo, not the olmOCR
    # checkpoint repo itself (fine-tuning only touched model weights, the
    # vocabulary/processor were never retrained, so reusing the original
    # is correct - not a workaround). Defaults to repo_id (every other
    # loader's existing behavior) when not set.
    processor_repo_id: Optional[str] = None

    do_sample: bool = False
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 20
    repetition_penalty: float = 1.1
    presence_penalty: Optional[float] = None
    no_repeat_ngram_size: Optional[int] = None
    max_new_tokens: int = 512

    # Explicit generation terminator. When set, passed to generate() as
    # stop_strings=[stop_string] (requires tokenizer=... also passed at
    # call time - HF's stop_strings support needs the tokenizer to detect
    # the string across token boundaries). Exists because repetition_penalty
    # and no_repeat_ngram_size only discourage exact repeats - they give
    # the model no positive reason to stop, so unbounded fields (e.g.
    # subject_keywords) can drift into novel-but-ungrounded content once
    # repetition is blocked. A literal required terminator, paired with a
    # prompt instruction to emit it, gives generate() a hard structural
    # stop condition instead of relying on instruction-following alone.
    stop_string: Optional[str] = None

    # Dedicated token ceiling for the isolated subject_keywords call (see
    # Qwen3VLLoader._run_generate). Separate from max_new_tokens because
    # that budget needs to stay large enough for document_type/dates/
    # personal_names/place_names, while subject_keywords has shown
    # unbounded drift at every budget tried (44/128/282 keywords at
    # 250/512/1024 tokens respectively, scaling roughly linearly with
    # room given rather than naturally stopping) - only an external,
    # tight, field-specific ceiling reliably bounds it.
    keywords_max_new_tokens: int = 80

    # Beam search width. Every loader so far has used greedy (do_sample=
    # False) or sampling (do_sample=True) - Florence-2's official example
    # uses neither, it uses beam search (num_beams=3) as its recommended
    # decoding strategy instead. Defaults to 1 (no beam search, standard
    # greedy behavior) so this field is a no-op for every other loader
    # unless explicitly set.
    num_beams: int = 1

    # Beam-search-specific, meaningless when num_beams=1 - only applied
    # by loaders when beam search is actually active. length_penalty < 1
    # biases toward shorter completions; > 1 biases toward longer ones.
    # Added specifically to test against Florence-2's fabrication
    # pattern (generating increasingly elaborate invented prose past
    # what's actually on the document) - a direct lever against exactly
    # that failure mode, unlike repetition_penalty/no_repeat_ngram_size
    # which target exact-repetition loops, a different mechanism.
    length_penalty: float = 1.0
    early_stopping: bool = False

    # Optional, model-agnostic - passed to from_pretrained() as
    # attn_implementation=... only if set. Left None by default rather
    # than hardcoding "flash_attention_2" (as some model cards do in
    # their example code), since flash-attn isn't guaranteed installed
    # and a hardcoded requirement would hard-crash the load on any
    # machine without it. Set explicitly per-model-profile if desired.
    attn_implementation: Optional[str] = None

    # Resolution / token ceilings — model-specific, was previously
    # hardcoded inline (e.g. Qwen's min_pixels/max_pixels, LiquidVL's
    # max_image_tokens). Now lives in config so it's tunable without
    # touching loader code.
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None
    min_image_tokens: Optional[int] = None
    max_image_tokens: Optional[int] = None
    do_image_splitting: Optional[bool] = None

    # Reasoning toggle — mechanism varies per model. "system_prompt" means
    # the loader injects/removes a system message; "param" means a native
    # generate() kwarg exists; "unsupported" means the model has no
    # reasoning mode to toggle (most VLMs tested so far).
    reasoning_toggle_mechanism: str = "unsupported"  # "system_prompt" | "param" | "unsupported"
    reasoning_enabled: bool = False

    # VRAM headroom, in GB, deliberately left unused by device_map when
    # loading this model. Exists because "auto" device mapping will
    # otherwise claim nearly all available VRAM for weights, leaving no
    # room for KV-cache growth as prompt+output context accumulates
    # during a long extraction run, or for other processes running
    # alongside the batch. Applied via max_memory in loader init.
    vram_headroom_gb: float = 1.5
    cpu_offload_limit_gb: float = 48.0

    prompt_version: str = "v1"
    prompt_text: str = ""

    # Constrained decoding (2026-07-24) - masks generation to only allow
    # expected-script characters (Latin letters/accented Latin, digits,
    # this project's own output-format punctuation), forcing the model
    # to pick its best answer from what's actually plausible instead of
    # falling back to a high-probability-but-wrong-script token (real
    # evidence: multiple runs this session produced stray Cyrillic
    # fragments under low visual confidence). See
    # core/loaders/constrained_decoding.py for the full mechanism and
    # which loaders it applies to. Opt-in (default False) - a deliberate
    # behavior change like reasoning_enabled, not silently applied.
    restrict_output_charset: bool = False

    # Whether this model/loader combination can generate from text alone,
    # with no image passed to _run_generate(). Defaults False (safe) -
    # every loader is a VLM built to always receive an image, and most
    # never had this path exercised. False here is NOT a claim that the
    # underlying model architecture is incapable of text-only generation
    # (several are vision fine-tunes of a text-only backbone, e.g. the
    # Gemma/Qwen families, and likely could do it) - it means this
    # SPECIFIC loader's _run_generate() has not been updated to build a
    # text-only message/processor call AND had that path empirically
    # confirmed against the real model. Only flip to True for a model
    # after both: (1) _run_generate() has an explicit `if raw_image is
    # None` branch (no image content block, no images= kwarg to the
    # processor), and (2) a real generation call with raw_image=None has
    # actually been run and produced sensible output - same
    # "confirmed by real testing, not assumed" discipline as every other
    # capability flag in this project. See model_console/adapter.py,
    # which gates whether a chat turn without an image is even attempted
    # on this flag rather than a blanket assumption either way.
    text_only_supported: bool = False

    # The mirror-image flag of text_only_supported above, for a
    # different failure mode: whether this loader can accept an image
    # AT ALL, not whether it can run without one. Defaults True - every
    # loader until 2026-08-13 was a VLM with a real vision tower, so
    # accepting an image was always safe. core/loaders/text_llm_loader.py
    # (config/models/qwen_research_text.yaml - Qwen2.5-7B-Instruct, no
    # vision tower, no processor) is the first loader where this must be
    # False: its _run_generate()/_run_generate_with_history() already
    # raise a loud RuntimeError if raw_image is not None (defense at the
    # loader level, correct and unchanged), but that only surfaces AFTER
    # a real model load - model_console/adapter.py's send_turn() checks
    # this flag BEFORE that, so a user who attaches an image to a
    # text-only-no-vision model gets an immediate, friendly ValueError
    # instead of waiting through a multi-minute cold load first (see
    # adapter.py's model_supports_image_input(), used the same way
    # model_supports_text_only() already is by chat_tab.py).
    image_input_supported: bool = True

    # The model's real usable context window (input + output tokens
    # combined), in tokens. None (default) means "unknown" - callers
    # that need a budget (model_console's conversation-history trimming,
    # see adapter.py's _resolve_context_length()) fall back to the
    # tokenizer's own model_max_length if that looks sane (not one of
    # the nonsense huge sentinel values - e.g. 1000000000000000019884624838656 -
    # some HF tokenizers report when a model card never set a real one),
    # else a conservative hardcoded default. No loader currently reads
    # this field itself - it exists for external context-budgeting
    # logic, not generation mechanics. Set explicitly per-model-profile
    # from the real model card/config.json when known.
    context_length: Optional[int] = None

    # Which execution backend runs this model (2026-08-13, multi-runtime
    # integration - see the plan's "Multi-Runtime Model Integration"
    # addendum). "transformers" (default - every existing config,
    # byte-identical behavior) means the normal loader path:
    # LOADER_REGISTRY + core.model_residency in-process. "vllm" means
    # the model runs as a vLLM OpenAI-compatible server subprocess
    # managed by core/vllm_runtime.py (WSL backend only - no loader
    # class is ever instantiated for it; api/agent_main.py dispatches
    # on this field before touching ChatBackendAdapter). This field
    # exists to make the (runtime, checkpoint) combination explicit and
    # queryable: a checkpoint that fails or performs badly under one
    # runtime is a combination result, never automatically a "failed
    # model" - the E4B-under-plain-transformers misdiagnosis this
    # session is the motivating case. vLLM-only knobs reuse existing
    # fields: context_length -> --max-model-len, vram_headroom_gb ->
    # --gpu-memory-utilization (see core/vllm_runtime.py).
    runtime: str = "transformers"

    # Purely descriptive - not read by any loader mechanics. Lets a
    # config declare what it's FOR (e.g. "classifier", "extractor",
    # "research_text") without inventing a structured capability system
    # (the plan's own future "agent profiles" - system_prompt +
    # generation_overrides + capabilities + output_mode - is explicitly
    # NOT built here). Added 2026-08-12 for the research_text role
    # (config/models/gemma_e4b_research.yaml) so that role is a real,
    # queryable fact about a config rather than only inferrable from its
    # filename/system_prompt. None (default) means unset - every
    # existing config predates this field and is unaffected.
    role: Optional[str] = None

    # Optional cache_implementation string passed straight through to
    # generate() (e.g. "static", "offloaded", "quantized") - see
    # transformers.generation.configuration_utils.ALL_CACHE_IMPLEMENTATIONS
    # for the full set the installed version recognizes. None (default,
    # every existing config) means HF's own default (DynamicCache) -
    # unchanged behavior for every model that doesn't set this.
    # Deliberately NOT defaulted to anything more aggressive - quantized
    # KV cache specifically needs a backend package (optimum-quanto or
    # hqq) not currently installed in this project's environment, and a
    # model/runtime combination should be empirically confirmed to
    # accept a given value before it's set here, not assumed from this
    # field merely existing. Only GemmaLoader reads this today.
    cache_implementation: Optional[str] = None

    extra: dict[str, Any] = field(default_factory=dict)

    def build_max_memory_map(self) -> Optional[dict]:
        """
        Computes a max_memory dict for device_map="auto", reserving
        vram_headroom_gb on each visible GPU. Returns None if torch/CUDA
        isn't available (caller should just omit max_memory in that case
        rather than pass None through - check the return value).
        """
        try:
            import torch
        except ImportError:
            return None
        if not torch.cuda.is_available():
            return None

        max_memory = {}
        for i in range(torch.cuda.device_count()):
            total_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            usable_gb = max(total_gb - self.vram_headroom_gb, 1.0)
            max_memory[i] = f"{usable_gb:.1f}GiB"
        max_memory["cpu"] = f"{self.cpu_offload_limit_gb:.0f}GiB"
        return max_memory

    def content_hash(self) -> str:
        """
        Deterministic hash of every setting that affects generation output.
        Written into every ExtractionResult so a fabrication or quality
        issue found later can be traced to the exact config that produced
        it — this was impossible to do reliably in VectorDB-Plugin, where
        settings were shared across a whole model family.
        """
        payload = {
            "model_name": self.model_name,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "presence_penalty": self.presence_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "max_new_tokens": self.max_new_tokens,
            "stop_string": self.stop_string,
            "keywords_max_new_tokens": self.keywords_max_new_tokens,
            "num_beams": self.num_beams,
            "length_penalty": self.length_penalty,
            "early_stopping": self.early_stopping,
            "attn_implementation": self.attn_implementation,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "min_image_tokens": self.min_image_tokens,
            "max_image_tokens": self.max_image_tokens,
            "do_image_splitting": self.do_image_splitting,
            "reasoning_enabled": self.reasoning_enabled,
            "prompt_version": self.prompt_version,
            "restrict_output_charset": self.restrict_output_charset,
            "cache_implementation": self.cache_implementation,
        }
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]

    @classmethod
    def from_yaml(cls, path: Path) -> "GenerationConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # "extra" is handled separately below - must not also be picked
        # up as a plain known field, or it gets passed to the
        # constructor twice (once as the real field, once merged in).
        known_fields = {
            k: v for k, v in data.items()
            if k in cls.__dataclass_fields__ and k != "extra"
        }
        extra = dict(data.get("extra") or {})
        extra.update({
            k: v for k, v in data.items()
            if k not in cls.__dataclass_fields__
        })
        return cls(**known_fields, extra=extra)

    def to_yaml(self, path: Path) -> None:
        """
        Used by the settings UI: checkbox/text-field edits get written
        back here. Keeps config as the single source of truth that both
        the UI and the pipeline read from.
        """
        payload = {
            k: getattr(self, k)
            for k in self.__dataclass_fields__
            if k != "extra"
        }
        payload.update(self.extra)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False)


def load_model_config(model_name: str) -> GenerationConfig:
    path = CONFIG_DIR / f"{model_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"No config found for model '{model_name}' at {path}. "
            "Every model must have a YAML config in config/models/."
        )
    return GenerationConfig.from_yaml(path)


class BaseLoader(ABC):
    """
    Subclasses implement initialize_model_and_tokenizer() and the two
    generate_* methods. Nothing else should need overriding for a
    typical HF Transformers-based VLM.
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.processor = None

        # Opt-in per-call inference telemetry (Phase 3, 2026-08-12) -
        # None by default; a loader that implements capture (currently
        # only GemmaLoader) overwrites this at the end of its own
        # _generate_from_messages()-equivalent tail with a dict of
        # VRAM/timing/token-count/stop-reason fields. model_console/
        # adapter.py's send_turn() reads this (getattr default None) and
        # folds it into the meta dict it already returns, so a loader
        # that never sets this is completely unaffected - same opt-in
        # shape as GemmaLoader's existing _captured_vision_tensors debug
        # side-channel. Overwritten (not accumulated) on every call -
        # always reflects only the MOST RECENT generate() call.
        self.last_inference_telemetry: Optional[dict[str, Any]] = None

    @abstractmethod
    def initialize_model_and_tokenizer(self) -> tuple[Any, Any, Any]:
        ...

    @abstractmethod
    def _build_prompt(self, task: str) -> str:
        """
        task is "classify" or "extract". Loaders build the appropriate
        prompt, applying the reasoning toggle per config.reasoning_toggle_mechanism.
        """
        ...

    @abstractmethod
    def _run_generate(self, raw_image: Any, prompt: str) -> str:
        """Raw model call. Returns unvalidated text."""
        ...

    def _run_generate_with_history(
        self, history: list[dict[str, Any]], raw_image: Any, prompt: str
    ) -> str:
        """
        Additive, NON-abstract (2026-08-11, model_console conversation
        context) - default raises NotImplementedError, so every existing
        loader that doesn't override this keeps working exactly as
        before; nothing about ABC/dataclass instantiation requires this
        be implemented. _run_generate() itself is NEVER touched by
        adding this - one-shot pipeline callers (classify()/extract())
        are completely unaffected.

        `history` is a list of already role-tagged, TEXT-ONLY message
        dicts: [{"role": "user"|"assistant", "content": [{"type":
        "text", "text": "..."}]}, ...], oldest first, built generically
        by model_console/adapter.py (no loader-specific knowledge
        needed for a plain text content entry - every loader already
        produces this exact shape for the text portion of its own
        single-turn content list).

        TEXT-ONLY BY DESIGN, not an oversight: real investigation found
        Gemma's separate processor() call takes images= as a SINGULAR
        PIL Image (core/loaders/gemma_loader.py), not a list - passing
        multiple images across turns would need per-loader rework and
        is unconfirmed for the other loaders too. Restricting history to
        text keeps this additive: the current turn's raw_image (if any)
        is still handled exactly as it already is by each loader's own
        single-turn message-building code, appended as the LAST message
        - only one image is ever in play per call, same invariant as
        _run_generate() already has today.

        A loader that supports this overrides it by reusing its own
        existing "generate from a full messages list" tail (the part of
        _run_generate() after message construction - apply_chat_template
        -> generate -> decode), called with `history + [current_turn_message]`
        instead of just `[current_turn_message]`. See gemma_loader.py/
        qwen_loader.py/qwen3vl_loader.py/internvl_loader.py/
        lfm2_vl_loader.py/smolvlm2_loader.py for the six real
        implementations - each refactors _run_generate() into building
        the current-turn message + calling a shared private tail method,
        so this method and _run_generate() share that tail rather than
        duplicating gen_kwargs/decode logic twice per file.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support conversation history "
            "yet - _run_generate_with_history() has no override."
        )

    def release(self) -> None:
        """
        Hook for releasing resources beyond model/tokenizer/processor -
        default no-op, since most loaders keep everything as in-process
        HF objects that core.row_extraction._release_model() already
        handles correctly (set to None, gc.collect(), torch.cuda.
        empty_cache()). Override this ONLY when a loader owns something
        that plain None-ing + garbage collection won't actually release
        - e.g. MoondreamLoader's worker subprocess (see that class),
        which holds its own separate CUDA context in a different OS
        process entirely. _release_model() calls this BEFORE clearing
        model/tokenizer/processor, so an override can still reference
        them if needed during its own cleanup.
        """
        pass

    def apply_checkpoint(self, checkpoint_path: str) -> None:
        """
        Wraps self.model with a saved LoRA adapter (data/outputs/
        lora_checkpoints/epoch_N/, see training/train_lora.py) - the
        SAME mechanism training/test_lora_checkpoint.py already uses
        (PeftModel.from_pretrained(model, checkpoint_path)), shared here
        so any in-process loader can apply a checkpoint without its own
        copy of this logic. Added 2026-07-28 per Jon's direction, so
        debug_tools/workflow_gui.py's extraction tabs could offer a real
        checkpoint option, not a UI stub.

        Must be called AFTER initialize_model_and_tokenizer() has set a
        real self.model - raises clearly rather than silently no-op'ing
        if called too early, same "loud error over silent wrong
        behavior" discipline as _maybe_add_charset_logits_processor's
        own tokenizer check below.

        Not implemented for subprocess-backed loaders (MoondreamLoader,
        DeepseekVL2Loader, HunyuanOcrLoader) - self.model is always None
        for those (the real model lives in a separate worker process/
        venv entirely), so this raises rather than silently doing
        nothing. Applying a checkpoint to one of those would need the
        adapter loaded INSIDE that worker script instead, not built
        here since no LoRA training has targeted those models.
        """
        if self.model is None:
            raise RuntimeError(
                f"{type(self).__name__}.apply_checkpoint() called before "
                "initialize_model_and_tokenizer() (or this loader runs its "
                "model in a separate subprocess/venv, so self.model is never "
                "set here at all - see this method's own docstring)."
            )
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, checkpoint_path)

    def _maybe_add_charset_logits_processor(self, gen_kwargs: dict) -> None:
        """
        Mutates gen_kwargs IN PLACE, adding an AllowedCharsLogitsProcessor
        (core/loaders/constrained_decoding.py) to its "logits_processor"
        list when self.config.restrict_output_charset is True - a no-op
        otherwise (default False), so calling this unconditionally at
        the top of every qualifying loader's _run_generate() before its
        own model.generate(**inputs, **gen_kwargs) call is always safe -
        no `if` needed at each call site.

        Only meaningful for loaders that call transformers' generate()
        directly with a real logits_processor= kwarg - see constrained_
        decoding.py's module docstring for the real, confirmed-by-audit
        list of which loaders in this project qualify (most do) and
        which don't (ChandraLoader - a different package's own generate_
        hf(), no hook exposed; MoondreamLoader - a separate-venv
        subprocess whose public API has no token-level hook, though a
        private-internals hook is confirmed possible if ever needed).
        Loaders that don't qualify simply never call this method.

        Requires self.tokenizer to already be set (true for every
        qualifying loader once initialize_model_and_tokenizer() has run,
        which is always the case by the time _run_generate() executes) -
        raises clearly rather than silently skipping the mask if that
        invariant is somehow violated, since a silently-ignored opt-in
        flag would be a worse surprise than a loud error.
        """
        if not self.config.restrict_output_charset:
            return
        if self.tokenizer is None:
            raise RuntimeError(
                f"{type(self).__name__}: restrict_output_charset=True but "
                "self.tokenizer is not set - initialize_model_and_tokenizer() "
                "must run before _run_generate()."
            )
        from core.loaders.constrained_decoding import AllowedCharsLogitsProcessor
        gen_kwargs.setdefault("logits_processor", [])
        gen_kwargs["logits_processor"].append(AllowedCharsLogitsProcessor(self.tokenizer))

    def classify(self, file_path: str, raw_image: Any) -> ClassificationResult:
        prompt = self._build_prompt(task="classify")
        raw_output = self._run_generate(raw_image, prompt)
        return self._parse_classification(file_path, raw_output)

    def extract(self, file_path: str, category: str, raw_image: Any) -> ExtractionResult:
        prompt = self._build_prompt(task="extract")
        raw_output = self._run_generate(raw_image, prompt)
        return self._parse_extraction(file_path, category, raw_output)

    @abstractmethod
    def _parse_classification(self, file_path: str, raw_output: str) -> ClassificationResult:
        """
        Parse model's raw text/JSON into a validated ClassificationResult.
        Must raise (not silently coerce) on malformed output so the
        pipeline can route the failure to uncertain_review rather than
        write a guessed classification.
        """
        ...

    @abstractmethod
    def _parse_extraction(self, file_path: str, category: str, raw_output: str) -> ExtractionResult:
        """Same contract as above, for extraction output."""
        ...
