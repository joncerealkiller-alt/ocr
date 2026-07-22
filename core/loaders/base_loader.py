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
