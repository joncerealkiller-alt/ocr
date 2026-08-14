"""
Generic tool framework for the local multimodal agent (Phase 1 of
docs/... architecture — see the plan this was built from:
"Local Multimodal Genealogy Agent — Architecture Document").

This package wraps EXISTING pipeline stages (classification, extraction,
preprocessing) as tools behind a declarative registry. It does not
reimplement any of them — core.document_classification,
core.extractor, core.image_preprocessing stay the source of truth.

Modules:
    errors.py    - ToolError, typed error classes
    schema.py    - ToolSpec, ToolCallBlock, ToolResultBlock
    registry.py  - TOOL_REGISTRY + @register_tool decorator
    dispatcher.py - dispatch()/dispatch_many() entry points
    tools.py     - the initial tool set (classify_document, preprocess_image,
                    extract_fields)
"""

from core.agent_tools.registry import TOOL_REGISTRY, register_tool
from core.agent_tools.dispatcher import dispatch, dispatch_many
from core.agent_tools.schema import ToolSpec, ToolCallBlock, ToolResultBlock
from core.agent_tools.errors import ToolError

# Importing tools.py registers the initial tool set as a side effect,
# mirroring how core/loader_registry.py's LOADER_REGISTRY is populated.
import core.agent_tools.tools  # noqa: F401

__all__ = [
    "TOOL_REGISTRY",
    "register_tool",
    "dispatch",
    "dispatch_many",
    "ToolSpec",
    "ToolCallBlock",
    "ToolResultBlock",
    "ToolError",
]
