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


def is_vllm_model(cfg: GenerationConfig) -> bool:
    """
    True for a config served by core/vllm_runtime.py's subprocess
    rather than a BaseLoader subclass - config.runtime == "vllm" is the
    authoritative signal (set in ~5 config/models/*.yaml files as of
    2026-08-15: qwen25_vl_7b_awq, gemma_12b_w4a16, gemma_e4b_w4a16,
    internvl3_5_8b_awq, minicpm_v_gptq). Every such config's
    loader_class is the literal string "vllm" - a documented sentinel
    (see those YAMLs' own comments) that is NEVER looked up in
    LOADER_REGISTRY; api/agent_main.py branches on config.runtime
    BEFORE any loader_class dispatch, and this function exists so
    validate_model_assignment() below mirrors that exact same branch
    instead of treating "vllm" as an unregistered/broken loader_class
    (which it would otherwise look like - a real bug this fixes: every
    vLLM-runtime model failed validate_model_assignment() before this,
    since "vllm" was never a real LOADER_REGISTRY key).
    """
    return cfg.runtime == "vllm"


def is_llamacpp_model(cfg: GenerationConfig) -> bool:
    """
    True for a config served by core/llamacpp_runtime.py's Windows-side
    llama-server subprocess (2026-09-01, two-system architecture
    benchmark) - config.runtime == "llamacpp", loader_class sentinel
    "llamacpp", exact same pattern as is_vllm_model() above and exempted
    from LOADER_REGISTRY lookup for the exact same reason.
    """
    return cfg.runtime == "llamacpp"


def validate_model_assignment(
    model_name: str, *, require_vision: bool = False, context: str = "",
) -> GenerationConfig:
    """
    Loads model_name's config and checks it's actually usable as a
    pipeline/console assignment: exists, enabled, has a registered
    loader_class (or is a valid vLLM-runtime config - see
    is_vllm_model()), and (if require_vision) can accept an image.
    Raises ValueError with a specific, actionable message rather than
    letting an unknown/disabled/wrong-capability model fail later as a
    confusing load error or - worse - silently run with the wrong
    model. `context` is a short label (e.g. "pipeline.yaml
    buckets.map_land_record.model") prepended to the error so a
    config-validation failure points straight at the offending config
    line, not just the model name.
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

    if not is_vllm_model(cfg) and not is_llamacpp_model(cfg) \
            and cfg.loader_class not in LOADER_REGISTRY:
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
    if is_vllm_model(cfg):
        raise ValueError(
            f"build_loader(): {model_name!r} is a runtime='vllm' config - it has "
            "no BaseLoader subclass at all (served by core/vllm_runtime.py's "
            "subprocess instead - see api/agent_main.py's runtime dispatch). "
            "Use core.vllm_runtime.vllm_runtime / model_console.adapter."
            "ChatBackendAdapter(backend='remote') instead of build_loader() for "
            "this model."
        )
    if is_llamacpp_model(cfg):
        raise ValueError(
            f"build_loader(): {model_name!r} is a runtime='llamacpp' config - it "
            "has no BaseLoader subclass (served by core/llamacpp_runtime.py's "
            "llama-server subprocess instead). Use model_console.adapter."
            "ChatBackendAdapter(backend='local') instead of build_loader() for "
            "this model."
        )
    loader_cls = LOADER_REGISTRY[cfg.loader_class]
    loader = loader_cls(cfg)
    loader._debug_mode = debug
    loader.initialize_model_and_tokenizer()
    return loader
