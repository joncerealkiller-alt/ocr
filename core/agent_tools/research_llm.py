"""
Shared access point for the dedicated text-only research LLM
(core/loaders/text_llm_loader.py, config/models/qwen_research_text.yaml
- quantized Qwen2.5-7B-Instruct, no vision tower). Extracted 2026-08-13
from core/agent_tools/web_research_agent.py once planner.py also needed
it for the outer final-answer synthesis step - Jon's principle: "keep
Gemma as the VLM front end, but if it's a research task that's text
only, no vision required, handing back to Gemma doesn't make sense for
synthesis." Applies wherever a generation step in the agent loop is
KNOWN to be text-only (no image involved) and is a synthesis/judgment
task rather than a vision task - currently: web_research_agent's
refine/synthesize calls, and planner.py's tool-evidence final-answer
step when no image is attached to the turn.

Borrowed via core.model_residency.residency.borrow() - never held
resident independently, same "one authoritative GPU owner" discipline
extract_fields_tool already follows (core/agent_tools/tools.py). Use
`borrowed()` (not `generate()`) when a caller needs MORE THAN ONE
generation in the same logical step (e.g. refine then synthesize) -
one borrow covers both calls instead of releasing Gemma, loading this
model, restoring Gemma, then repeating the whole cycle a second time
for the very next call (a real, confirmed inefficiency: the original
single-call-per-borrow version did exactly that for web_research_agent's
two internal calls every single turn).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator

from core.loaders.base_loader import load_model_config
from core.model_residency import residency

MODEL_NAME = "qwen_research_text"


@contextmanager
def borrowed() -> Iterator[Callable[[str], str]]:
    """
    Borrows the shared GPU residency slot ONCE and yields a
    generate(prompt_text) -> str callable valid for the lifetime of the
    `with` block - call it as many times as needed before the block
    exits; the chat model is restored only once, after the last call.
    """
    config = load_model_config(MODEL_NAME)
    with residency.borrow(MODEL_NAME, config) as loader:
        yield lambda prompt_text: loader._run_generate(None, prompt_text)


def generate(prompt_text: str) -> str:
    """Convenience wrapper for a single generation - borrows for just this one call."""
    with borrowed() as gen:
        return gen(prompt_text)
