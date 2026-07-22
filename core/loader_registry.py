"""
Central registry mapping loader_class names (as declared in
config/models/*.yaml) to their implementing classes.

Both the real pipeline (classifier.py, extractor.py) and the isolated
Model Assessment tool (model_assessment.py) import from here, so
adding a new model loader only requires one registration point.
"""

from __future__ import annotations

from core.loaders.gemma_loader import GemmaLoader
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

LOADER_REGISTRY = {
    "GemmaLoader": GemmaLoader,
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
}
