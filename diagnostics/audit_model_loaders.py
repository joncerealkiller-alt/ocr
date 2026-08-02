"""
Loader audit (2026-07-28, per Jon's direction): checks every config/
models/*.yaml against LIVE Hugging Face Hub metadata, catching drift
between what a config file claims (repo_id, expected architecture) and
what the Hub actually has right now - exactly the class of bug that
caused the real qwen2b.yaml failure (Qwen/Qwen2.5-VL-2B-Instruct
doesn't exist - confirmed 404 - while every other loader's repo_id
checked out fine).

Pure metadata lookups (huggingface_hub.HfApi().model_info()) - never
downloads weights, never loads a model, no GPU/inference involved, so
CLAUDE.md's nvidia-smi gate doesn't apply to this script.

For each config, checks:
  - Repo exists (catches stale/mistyped/deprecated repo_ids). If not,
    searches the Hub for same-author models with a similar name and
    lists the closest candidates, sorted by name-similarity to the
    configured repo_id.
  - Architecture/model_type reported by the Hub, shown alongside what
    THIS PROJECT'S loader class actually imports (core/loaders/*.py) -
    informational, not auto-flagged as a mismatch unless the repo
    lookup itself failed, since several loaders here deliberately use
    a DIFFERENT-but-compatible class than the Hub's own suggestion
    (e.g. GraniteVisionLoader on a llava_next-architecture repo,
    OlmOcrLoader repackaging Qwen2.5-VL) - flagging those as "wrong"
    would be a false positive against a deliberate, documented choice.
  - trust_remote_code: the Hub's transformersInfo.custom_class field is
    a reliable signal (non-null = the repo's own modeling code isn't a
    standard transformers class, so trust_remote_code=True is needed)
    - compared against LOADER_TRUST_REMOTE_CODE below, a static map of
      which loader .py files actually pass trust_remote_code=True
      today (grepped directly from source, not assumed).
  - Processor class the Hub expects (transformersInfo.processor).
  - Recommended dtype: the config.json's own "torch_dtype" field, if
    present (informational - every loader here already uses
    torch_dtype="auto" plus its own bf16-cast logic, not a hardcoded
    dtype driven by this).
  - Context/image limits: max_position_embeddings and any vision_config
    image_size found in config.json, for future benchmarking reference.
  - Gated repos: flagged separately (needs license acceptance/token,
    not a broken config).

Self-healing (2026-07-28, per Jon's direction: "if the audit finds
something unambiguous, propose a patch rather than just reporting
it"): ONLY proposes (never silently applies) a repo_id fix, and ONLY
when there is exactly one same-author, same-base-name candidate on the
Hub - anything with multiple plausible candidates or requiring a
judgment call is reported for a human to decide, not guessed at.

Usage:
    python diagnostics/audit_model_loaders.py
    python diagnostics/audit_model_loaders.py --model qwen2b   # just one
"""

from __future__ import annotations

import argparse
import difflib
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Windows console default codepage (cp1252) can't render the ✓/⚠/✗
# status icons below - same fix already applied in
# training/test_lora_checkpoint.py for the same reason.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "config" / "models"

# Which loader .py files ACTUALLY pass trust_remote_code=True today -
# grepped directly from core/loaders/*.py (2026-07-28), not assumed from
# comments. Several loaders' docstrings mention trust_remote_code (e.g.
# got_ocr2_loader.py, florence_loader.py, internvl_loader.py) only to
# explain why the CONFIGURED repo_id does NOT need it (a native "-hf"/
# transformers-integrated conversion instead of the original community
# repo) - those are correctly False here, matching their own actual code.
LOADER_TRUST_REMOTE_CODE = {
    "GemmaLoader": False,
    "QwenLoader": False,
    "Qwen3VLLoader": False,
    "SmolVLM2Loader": False,
    "GraniteVisionLoader": False,
    "FlorenceLoader": False,
    "GlmOcrLoader": False,
    "OlmOcrLoader": False,
    "InternVLLoader": False,
    "ChandraLoader": False,
    "PixtralLoader": False,
    "GotOcr2Loader": False,
    "MoondreamLoader": True,  # via core/loaders/_moondream_worker.py's subprocess call
}

# What each loader actually imports as its model class - for the
# informational "Architecture" line, sourced directly from each
# core/loaders/*.py file (2026-07-28), not guessed.
LOADER_MODEL_CLASS = {
    "GemmaLoader": "AutoModelForCausalLM",
    "QwenLoader": "Qwen2_5_VLForConditionalGeneration",
    "Qwen3VLLoader": "Qwen3VLForConditionalGeneration",
    "SmolVLM2Loader": "AutoModelForImageTextToText",
    "GraniteVisionLoader": "AutoModelForImageTextToText",
    "FlorenceLoader": "Florence2ForConditionalGeneration",
    "GlmOcrLoader": "GlmOcrForConditionalGeneration",
    "OlmOcrLoader": "Qwen2_5_VLForConditionalGeneration",
    "InternVLLoader": "AutoModelForImageTextToText",
    "ChandraLoader": "AutoModelForImageTextToText",
    "PixtralLoader": "LlavaForConditionalGeneration",
    "GotOcr2Loader": "AutoModelForImageTextToText",
    "MoondreamLoader": "AutoModelForCausalLM (subprocess)",
}


@dataclass
class AuditResult:
    model_name: str
    yaml_path: Path
    repo_id: str
    loader_class: str
    status: str = "ok"  # "ok" | "warning" | "error"
    findings: list[str] = field(default_factory=list)
    suggested_repo_id: str | None = None  # set only when self-heal has ONE clear candidate


def _read_yaml_fields(path: Path) -> dict:
    """
    Minimal, dependency-light field extraction - this project's own
    yaml files are flat key: value (see config/models/*.yaml), so a
    real yaml parser isn't required just to pull repo_id/loader_class/
    model_name, and avoids adding a new dependency to a pure-audit
    script. Falls back to the full yaml module if simple parsing is
    insufficient (kept simple deliberately; every field this script
    needs is a plain top-level scalar in every existing config).
    """
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _closest_candidates(api, repo_id: str, limit: int = 5) -> list[str]:
    """
    Searches the Hub for same-author models with a similar base name -
    e.g. Qwen/Qwen2.5-VL-2B-Instruct (404) -> searches author="Qwen",
    "Qwen2.5-VL" -> finds the 3B/7B/32B/72B siblings. Sorted by string
    similarity to the original repo_id so the most likely intended
    match (same family, different size suffix) surfaces first.
    """
    if "/" not in repo_id:
        return []
    author, name = repo_id.split("/", 1)
    # Strip a trailing size/variant token (2B, 3B, -Instruct, etc.) down
    # to a base search term likely to still match sibling sizes -
    # crude but effective: just use the part before the first size-like
    # token (a run of digits followed by 'B' or 'b').
    import re
    m = re.search(r"\d+[Bb]", name)
    search_term = name[:m.start()].rstrip("-_") if m else name
    try:
        candidates = list(api.list_models(search=search_term, author=author, limit=limit * 4))
    except Exception:
        return []
    ids = [c.id for c in candidates if c.id.lower() != repo_id.lower()]
    ids.sort(key=lambda cid: difflib.SequenceMatcher(None, cid.lower(), repo_id.lower()).ratio(), reverse=True)
    return ids[:limit]


def audit_one(api, yaml_path: Path) -> AuditResult:
    fields = _read_yaml_fields(yaml_path)
    model_name = fields.get("model_name", yaml_path.stem)
    repo_id = fields.get("repo_id", "")
    loader_class = fields.get("loader_class", "")

    result = AuditResult(model_name=model_name, yaml_path=yaml_path, repo_id=repo_id, loader_class=loader_class)

    if not repo_id:
        result.status = "error"
        result.findings.append("No repo_id set in this config at all.")
        return result

    from huggingface_hub.errors import RepositoryNotFoundError, GatedRepoError, HFValidationError

    try:
        info = api.model_info(repo_id, expand=["config", "transformersInfo", "gated"])
    except RepositoryNotFoundError:
        result.status = "error"
        result.findings.append(f"Repo not found: {repo_id!r} (404 - stale, mistyped, or deprecated).")
        candidates = _closest_candidates(api, repo_id)
        if candidates:
            result.findings.append("Closest available on the Hub (same author, similar name):")
            for c in candidates:
                result.findings.append(f"    - {c}")
            if len(candidates) == 1:
                result.suggested_repo_id = candidates[0]
        else:
            result.findings.append("No similarly-named repos found under the same author.")
        return result
    except GatedRepoError:
        result.status = "warning"
        result.findings.append(
            f"Repo {repo_id!r} is GATED - requires accepting a license/terms on huggingface.co "
            f"and an authenticated token to actually download. Not a broken config, but will fail "
            f"without HF_TOKEN + accepted access."
        )
        return result
    except HFValidationError as e:
        result.status = "error"
        result.findings.append(f"Invalid repo_id {repo_id!r}: {e}")
        return result
    except Exception as e:
        result.status = "error"
        result.findings.append(f"Lookup failed ({type(e).__name__}): {e}")
        return result

    cfg = info.config or {}
    ti = info.transformers_info

    # --- Architecture (informational) ---
    hub_model_type = cfg.get("model_type", "?")
    hub_architectures = cfg.get("architectures", [])
    loader_model_class = LOADER_MODEL_CLASS.get(loader_class, "?")
    result.findings.append(
        f"Architecture: Hub reports model_type={hub_model_type!r}, architectures={hub_architectures} "
        f"| this loader imports {loader_model_class!r}"
    )

    # --- trust_remote_code ---
    hub_needs_custom_code = bool(ti and ti.custom_class)
    loader_uses_trust_remote_code = LOADER_TRUST_REMOTE_CODE.get(loader_class)
    if hub_needs_custom_code and not loader_uses_trust_remote_code:
        result.status = "warning"
        result.findings.append(
            f"WARNING: Hub reports custom modeling code (custom_class={ti.custom_class!r}) - "
            f"this repo needs trust_remote_code=True, but {loader_class} does not currently pass it. "
            f"Will likely fail to load or fall back to a generic (wrong) architecture."
        )
    elif hub_needs_custom_code and loader_uses_trust_remote_code:
        result.findings.append(
            f"trust_remote_code: required (custom_class={ti.custom_class!r}) and correctly set by {loader_class}."
        )
    else:
        result.findings.append("trust_remote_code: not required (standard transformers class).")

    # --- Processor class ---
    if ti and ti.processor:
        result.findings.append(f"Processor: Hub expects {ti.processor!r}.")
    else:
        result.findings.append("Processor: not declared in Hub metadata (no processor_config.json found).")

    # --- Recommended dtype ---
    torch_dtype = cfg.get("torch_dtype")
    if torch_dtype:
        result.findings.append(f"Recommended dtype (config.json): {torch_dtype!r}.")

    # --- Context/image limits (for future benchmarking) ---
    limits = []
    if cfg.get("max_position_embeddings"):
        limits.append(f"max_position_embeddings={cfg['max_position_embeddings']}")
    vision_cfg = cfg.get("vision_config") or {}
    if isinstance(vision_cfg, dict) and vision_cfg.get("image_size"):
        limits.append(f"vision_config.image_size={vision_cfg['image_size']}")
    if limits:
        result.findings.append("Limits: " + ", ".join(limits))

    # --- Quantization (informational - only PixtralLoader hardcodes bitsandbytes today) ---
    if loader_class == "PixtralLoader":
        result.findings.append(
            "Quantization: PixtralLoader hardcodes bitsandbytes 4-bit (load_in_4bit=True) in its "
            "own from_pretrained() call, independent of this yaml - not independently re-verified "
            "against current bitsandbytes/transformers versions by this audit."
        )

    if result.status == "ok" and len(result.findings) <= 1:
        result.findings = ["All checks passed."]

    return result


def print_report(results: list[AuditResult]):
    print("Loader Audit Report")
    print("=" * 60)
    for r in results:
        icon = {"ok": "✓", "warning": "⚠", "error": "✗"}[r.status]
        print(f"{icon} {r.model_name}  (repo_id={r.repo_id!r}, loader={r.loader_class})")
        for line in r.findings:
            print(f"    {line}")
        print()

    proposals = [r for r in results if r.suggested_repo_id]
    if proposals:
        print("Suggested changes:")
        print("-" * 60)
        for r in proposals:
            rel_path = r.yaml_path.relative_to(PROJECT_ROOT)
            print(f"{rel_path}")
            print(f"- repo_id: {r.repo_id}")
            print(f"+ repo_id: {r.suggested_repo_id}")
            print()

    n_error = sum(1 for r in results if r.status == "error")
    n_warning = sum(1 for r in results if r.status == "warning")
    n_ok = sum(1 for r in results if r.status == "ok")
    print(f"Summary: {n_ok} OK, {n_warning} warning(s), {n_error} error(s), out of {len(results)} config(s).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, default=None,
                         help="Audit just this one model (matches config/models/<model>.yaml). "
                              "Default: audit every yaml in config/models/.")
    args = parser.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()

    if args.model:
        yaml_paths = [MODELS_DIR / f"{args.model}.yaml"]
        if not yaml_paths[0].exists():
            print(f"ERROR: {yaml_paths[0]} not found.")
            sys.exit(1)
    else:
        yaml_paths = sorted(MODELS_DIR.glob("*.yaml"))

    results = [audit_one(api, p) for p in yaml_paths]
    print_report(results)


if __name__ == "__main__":
    main()
