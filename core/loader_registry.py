"""
Central registry mapping loader_class names (as declared in
config/models/*.yaml) to their implementing classes.

Both the real pipeline (classifier.py, extractor.py) and the isolated
Model Assessment tool (model_assessment.py) import from here, so
adding a new model loader only requires one registration point.
"""

from __future__ import annotations

from core.loaders.base_loader import BaseLoader, GenerationConfig, load_model_config

from core.loaders.gemma_loader import GemmaLoader
from core.loaders.gemma_unified_loader import Gemma4UnifiedLoader
from core.loaders.chameleon_loader import ChameleonLoader
from core.loaders.qwen_loader import QwenLoader
from core.loaders.qwen3vl_loader import Qwen3VLLoader
from core.loaders.smolvlm2_loader import SmolVLM2Loader
from core.loaders.granite_vision_loader import GraniteVisionLoader
from core.loaders.florence_loader import FlorenceLoader
from core.loaders.glm_ocr_loader import GlmOcrLoader
from core.loaders.olmocr_loader import OlmOcrLoader
from core.loaders.internvl_loader import InternVLLoader
from core.loaders.chandra_loader import ChandraLoader
from core.loaders.pixtral_loader import PixtralLoader
from core.loaders.got_ocr2_loader import GotOcr2Loader
from core.loaders.moondream_loader import MoondreamLoader
from core.loaders.deepseek_vl2_loader import DeepseekVL2Loader
from core.loaders.hunyuan_ocr_loader import HunyuanOcrLoader
from core.loaders.lfm2_vl_loader import Lfm2VlLoader
from core.loaders.granite_vision_4_1_loader import GraniteVision41Loader
from core.loaders.nanonets_ocr2_loader import NanonetsOcr2Loader
from core.loaders.text_llm_loader import TextLLMLoader
from core.loaders.minicpm_v_loader import MinicpmVLoader

LOADER_REGISTRY = {
    "GemmaLoader": GemmaLoader,
    "Gemma4UnifiedLoader": Gemma4UnifiedLoader,
    "ChameleonLoader": ChameleonLoader,
    "QwenLoader": QwenLoader,
    "Qwen3VLLoader": Qwen3VLLoader,
    "SmolVLM2Loader": SmolVLM2Loader,
    "GraniteVisionLoader": GraniteVisionLoader,
    "FlorenceLoader": FlorenceLoader,
    "GlmOcrLoader": GlmOcrLoader,
    "OlmOcrLoader": OlmOcrLoader,
    "InternVLLoader": InternVLLoader,
    "ChandraLoader": ChandraLoader,
    "PixtralLoader": PixtralLoader,
    "GotOcr2Loader": GotOcr2Loader,
    "MoondreamLoader": MoondreamLoader,
    "DeepseekVL2Loader": DeepseekVL2Loader,
    "HunyuanOcrLoader": HunyuanOcrLoader,
    "Lfm2VlLoader": Lfm2VlLoader,
    "GraniteVision41Loader": GraniteVision41Loader,
    "NanonetsOcr2Loader": NanonetsOcr2Loader,
    "TextLLMLoader": TextLLMLoader,
    "MinicpmVLoader": MinicpmVLoader,
}


def validate_model_assignment(
    model_name: str, *, require_vision: bool = False, context: str = "",
) -> GenerationConfig:
    """
    Loads model_name's config and checks it's actually usable as a
    pipeline/console assignment: exists, enabled, has a registered
    loader_class, and (if require_vision) can accept an image. Raises
    ValueError with a specific, actionable message rather than letting
    an unknown/disabled/wrong-capability model fail later as a confusing
    load error or - worse - silently run with the wrong model. `context`
    is a short label (e.g. "pipeline.yaml buckets.map_land_record.model")
    prepended to the error so a config-validation failure points straight
    at the offending config line, not just the model name.
    """
    prefix = f"{context}: " if context else ""
    try:
        cfg = load_model_config(model_name)
    except FileNotFoundError:
        raise ValueError(
            f"{prefix}assignment references unknown model '{model_name}' - "
            f"no config/models/{model_name}.yaml exists."
        )

    if not cfg.enabled:
        raise ValueError(
            f"{prefix}assignment references disabled model '{model_name}' "
            f"(config/models/{model_name}.yaml has enabled: false)."
        )

    if cfg.loader_class not in LOADER_REGISTRY:
        raise ValueError(
            f"{prefix}model '{model_name}' declares loader_class="
            f"{cfg.loader_class!r}, which is not registered in "
            f"LOADER_REGISTRY. Known loaders: {sorted(LOADER_REGISTRY)}."
        )

    if require_vision and not cfg.image_input_supported:
        raise ValueError(
            f"{prefix}model '{model_name}' is assigned to a vision stage "
            f"but its config sets image_input_supported: false (text-only "
            f"model)."
        )

    return cfg


def build_loader(model_name: str, *, debug: bool = False) -> BaseLoader:
    """
    The one real "config -> registry -> loader" dispatch: load the
    model's YAML config, look up its loader_class in LOADER_REGISTRY,
    instantiate and initialize it. Callers that need to set config
    fields (e.g. classifier.py's prompt_text) before load should build
    the GenerationConfig via load_model_config() themselves and pass it
    to the loader_cls directly instead - this helper is for the common
    "just give me a ready-to-use loader for this model" case
    (semantic_stages.py, benchmark scripts, ad hoc tooling), so that
    call sequence isn't copy-pasted at every one of them.
    """
    cfg = validate_model_assignment(model_name)
    loader_cls = LOADER_REGISTRY[cfg.loader_class]
    loader = loader_cls(cfg)
    loader._debug_mode = debug
    loader.initialize_model_and_tokenizer()
    return loader
