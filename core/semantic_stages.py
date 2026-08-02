"""
Modular runner for POST-BUCKET semantic classification stages - Gemma
calls that share ONE resident model load, chained one after another.
Jon's Phase 3 design ("adding future prompts requires minimal code
changes"), applied concretely: a new stage is a SemanticStage entry
(input CSV, prompt file, output CSV, expected fields), not a new copy
of the loop/loading logic.

Deliberately separate from core/classifier.py (Stage 2, bucket
routing) rather than a modification of it or a copy-pasted variant:
- core/classifier.py is untouched, not imported for its run() logic.
- Every stage here only ever READS its input CSV (e.g.
  data/buckets/dense_tabular_rows.csv) and writes to a DIFFERENT
  output CSV - there is no code path in this module that opens a
  bucket CSV for writing. Per Jon's standing rule: an existing working
  file gets touched only on a copy, never the original, unless told
  otherwise - here that's satisfied structurally (no write access to
  the original at all), not just by convention.

Parsing reuses core.loaders.gemma_loader.GemmaLoader._parse_kv_block -
the SAME key:value-line parser classify()/_parse_classification()
already depend on (see that module's own docstring for why key:value
was chosen over JSON: smaller models degrade more predictably that
way) - not reimplemented here.

Usage as a library:
    loader = load_gemma_loader()
    for stage in STAGES:
        run_stage(loader, stage)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Callable

from PIL import Image

from core.loaders.base_loader import load_model_config
from core.loader_registry import LOADER_REGISTRY
from core.loaders.gemma_loader import GemmaLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class SemanticStage:
    name: str
    input_csv: Path
    output_csv: Path
    prompt_path: Path
    output_fields: list[str]              # parsed response keys to persist as CSV columns, in order
    input_filter: Callable[[dict], bool] = field(default=lambda row: True)
    # e.g. skip rows a prior stage already marked low-confidence/error


def load_gemma_loader(model_name: str = "gemma") -> GemmaLoader:
    """
    Loads Gemma ONCE. Callers run every stage against this same loader
    instance - matching Jon's "stay resident in VRAM, chain multiple
    semantic calls, unload once at the end" design. Separate from
    core.classifier.build_classifier_loader() (which hard-codes the
    bucket-routing prompt into config.prompt_text at load time) since
    this runner passes each stage's own prompt per-call instead - one
    loaded model, many different prompts across stages/images.
    """
    model_cfg = load_model_config(model_name)
    loader_cls = LOADER_REGISTRY.get(model_cfg.loader_class)
    if loader_cls is None:
        raise ValueError(
            f"No loader registered for loader_class={model_cfg.loader_class!r}. "
            f"Known loaders: {list(LOADER_REGISTRY.keys())}"
        )
    loader = loader_cls(model_cfg)
    loader.initialize_model_and_tokenizer()
    return loader


def run_stage(loader: GemmaLoader, stage: SemanticStage) -> Path:
    """
    Runs ONE stage end to end: reads stage.input_csv, calls
    loader._run_generate() once per row (NOT loader.classify() - that
    method's parsing is hard-coded to the bucket-routing
    ClassificationResult schema; see gemma_loader.py's
    _parse_classification), parses the response, writes
    stage.output_csv. Never opens stage.input_csv for writing.

    The prompt is passed through string.Template.safe_substitute()
    with each image's own width/height (safe_substitute, not the
    stricter substitute, so a stage's prompt is free to not reference
    $width/$height at all without erroring - most won't). Using
    Template rather than str.format() is deliberate: a prompt with
    unrelated literal curly braces (e.g. a JSON example in its
    instructions) would make .format() raise or silently misfire;
    Template only ever touches $-prefixed placeholders, leaving
    everything else in the prompt text alone.
    """
    if not stage.input_csv.exists():
        raise FileNotFoundError(
            f"Stage {stage.name!r} input not found: {stage.input_csv} - "
            f"has the prior stage that produces it actually been run?"
        )

    prompt_template = Template(stage.prompt_path.read_text(encoding="utf-8"))

    with open(stage.input_csv, "r", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if stage.input_filter(r)]

    stage.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file_path"] + stage.output_fields + ["error"]

    print(f"\n=== Stage: {stage.name} ({len(rows)} file(s)) ===")
    print(f"    input:  {stage.input_csv}")
    print(f"    output: {stage.output_csv}")

    with open(stage.output_csv, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(rows, 1):
            file_path = row["file_path"]
            print(f"[{i}/{len(rows)}] {file_path}")
            out_row = {"file_path": file_path, "error": ""}

            try:
                with Image.open(file_path) as raw_image:
                    raw_image = raw_image.convert("RGB")
                    w, h = raw_image.size
                    prompt = prompt_template.safe_substitute(width=w, height=h)
                    raw_output = loader._run_generate(raw_image, prompt)

                fields = GemmaLoader._parse_kv_block(raw_output)
                missing = [k for k in stage.output_fields if k not in fields]
                for key in stage.output_fields:
                    out_row[key] = fields.get(key, "")
                if missing:
                    out_row["error"] = f"missing fields: {missing}"
                    print(f"  -> WARNING missing fields {missing} (raw: {raw_output[:150]!r})")
                else:
                    headline_field = stage.output_fields[0]
                    print(f"  -> {headline_field}={out_row[headline_field]!r}")
            except Exception as e:
                out_row["error"] = str(e)[:300]
                print(f"  -> FAILED: {e}")

            writer.writerow(out_row)

    print(f"Stage {stage.name!r} done. Output: {stage.output_csv}")
    return stage.output_csv
