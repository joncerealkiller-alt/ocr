"""
Shared request-loop helper for subprocess worker scripts (see
core/loaders/subprocess_loader_base.py for the loader-side half of this
pattern, and its docstring for why this exists at all - some models
need a pinned dependency version incompatible with this project's main
environment, so they run in a separate venv and talk over stdin/stdout).

A new worker script (one per distinct model, since each model's actual
"how do I call this thing" API differs - moondream's model.query(),
some other model's model.generate(), etc.) only needs to supply:

    load_fn() -> model
        Called once at startup. Do all imports INSIDE this function
        (not at module level) so a worker for a CPU-light model doesn't
        pay torch's import cost before it's needed, and so any import
        error surfaces as part of the normal startup failure path
        rather than crashing before argument parsing.

    query_fn(model, image: PIL.Image, question: str, extra: dict) -> str
        Called once per request. `extra` holds whatever additional
        fields the loader's extra_request_fields() sent beyond image_
        path/question (already resolved to a real PIL Image by the time
        query_fn sees it - this helper handles the image_path -> Image
        translation). Return the answer text.

Then call run_worker(load_fn, query_fn) as the script's __main__ body -
see _moondream_worker.py for the reference implementation.

Wire protocol implemented here (must match SubprocessLoaderBase exactly
- this file and that one are two halves of one contract):
    Request:  {"image_path": str, "question": str, ...extra fields}
    Response: {"answer": str} or {"error": str}
    "READY" printed to stdout once, after load_fn() returns, before the
    request loop starts.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable


def run_worker(
    load_fn: Callable[[], object],
    # query_fn's second arg is a PIL.Image.Image at runtime - typed as
    # Any rather than a forward-referenced "PIL.Image.Image" string,
    # since PIL is deliberately NOT imported at module level (see this
    # module's docstring: keeps a worker for a CPU-light model from
    # paying an import cost it doesn't need before it's needed).
    query_fn: Callable[[object, Any, str, dict], str],
) -> None:
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <repo_id> <revision> [extra args...]",
              file=sys.stderr)
        sys.exit(2)

    from PIL import Image  # deferred - see module docstring

    model = load_fn()

    print("READY", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            image_path = request.pop("image_path")
            question = request.pop("question")
            with Image.open(image_path) as raw_image:
                if raw_image.mode != "RGB":
                    raw_image = raw_image.convert("RGB")
                answer = query_fn(model, raw_image, question, request)
            response = {"answer": answer.strip() if isinstance(answer, str) else answer}
        except Exception as e:
            response = {"error": f"{type(e).__name__}: {e}"}

        print(json.dumps(response), flush=True)
