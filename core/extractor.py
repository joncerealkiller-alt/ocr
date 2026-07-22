"""
Stage 4 of the pipeline: run each bucket CSV (produced by classifier.py)
through its configured extraction model, validate against schema, run
anomaly checks across the batch, and write final output.

Usage:
    python -m core.extractor

Reads config/pipeline.yaml to know which model/prompt handles which
bucket. Skips uncertain_review.csv entirely (human review queue, no
automated extraction). Skips any bucket with model: null.

Anomaly checks (see config/pipeline.yaml anomaly_checks) run AFTER all
extraction in a bucket completes, since they need the full batch to
compute things like "does this name share any n-grams with anything
else in the run." Flagged records are NOT dropped — they're written
to data/outputs/anomaly_flags.csv for review, while the extraction
itself still goes to the main output. This mirrors the project's
stated approach: don't silently discard, surface for audit.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import yaml
import torch
from PIL import Image
from pydantic import ValidationError

from core.loaders.base_loader import load_model_config
from core.loader_registry import LOADER_REGISTRY
from core.schema import DocumentCategory, ExtractionResult, ConfidenceLevel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUCKET_DIR = PROJECT_ROOT / "data" / "buckets"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

# Loaders are model-specific but expensive to load (GPU memory), so we
# cache one instance PER MODEL across buckets - not per (model, prompt)
# pair. Multiple buckets routinely share the same model with different
# prompts (all 6 extraction buckets currently use qwen3b), and the
# prompt text is just swapped on the already-loaded instance rather
# than triggering a full reload. Caching by (model, prompt) was an
# earlier bug here: it created a fresh model load - and fresh VRAM
# allocation - for every bucket even when the underlying weights were
# identical, which is why VRAM climbed bucket over bucket instead of
# staying flat.
_loader_cache: dict[str, Any] = {}


def load_pipeline_config() -> dict:
    with open(PROJECT_ROOT / "config" / "pipeline.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_or_build_loader(model_name: str, prompt_file: str):
    prompt_path = PROJECT_ROOT / prompt_file
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_path}. This bucket's extraction "
            f"prompt hasn't been written yet."
        )

    if model_name in _loader_cache:
        loader = _loader_cache[model_name]
        loader.config.prompt_text = prompt_path.read_text(encoding="utf-8")
        return loader

    model_cfg = load_model_config(model_name)
    model_cfg.prompt_text = prompt_path.read_text(encoding="utf-8")

    loader_cls = LOADER_REGISTRY.get(model_cfg.loader_class)
    if loader_cls is None:
        raise ValueError(
            f"No loader registered for loader_class={model_cfg.loader_class!r}. "
            f"Known loaders: {list(LOADER_REGISTRY.keys())}"
        )

    loader = loader_cls(model_cfg)
    loader.initialize_model_and_tokenizer()
    _loader_cache[model_name] = loader
    return loader


def read_bucket_file_paths(bucket_csv: Path) -> list[str]:
    if not bucket_csv.exists():
        return []
    with open(bucket_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row["file_path"] for row in reader if row.get("file_path")]


def extraction_result_to_row(result: ExtractionResult) -> dict:
    return {
        "file_path": result.file_path,
        "category": result.category.value,
        "document_type": result.document_type or "",
        "personal_names": "; ".join(
            f"{n.value}|{n.confidence.value}" for n in result.personal_names
        ),
        "place_names": "; ".join(
            f"{p.value}|{p.confidence.value}" for p in result.place_names
        ),
        "visible_dates": "; ".join(
            f"{d.value}|{d.confidence.value}" for d in result.visible_dates
        ),
        "subject_keywords": ", ".join(result.subject_keywords),
        "raw_model_output_len": result.raw_model_output_len,
        "model": result.model,
        "prompt_version": result.prompt_version,
        "generation_config_hash": result.generation_config_hash,
        "device_map": result.device_map or "",
        "max_memory": str(result.max_memory) if result.max_memory else "",
        "vram_headroom_gb": result.vram_headroom_gb if result.vram_headroom_gb is not None else "",
        "cpu_offload_limit_gb": result.cpu_offload_limit_gb if result.cpu_offload_limit_gb is not None else "",
        "oom_recovered": result.oom_recovered,
    }


OUTPUT_FIELDS = [
    "file_path", "category", "document_type", "personal_names",
    "place_names", "visible_dates", "subject_keywords",
    "raw_model_output_len", "model", "prompt_version",
    "generation_config_hash", "device_map", "max_memory",
    "vram_headroom_gb", "cpu_offload_limit_gb", "oom_recovered",
]

FAILED_FIELDS = ["file_path", "category", "model", "error"]


def run_anomaly_checks(results: list[ExtractionResult], checks_cfg: list[dict]) -> list[dict]:
    """
    Runs across a completed bucket's results. Returns flag rows, does
    not mutate results. Kept intentionally simple (no external NLP
    deps) - character-trigram overlap for the n-gram check, mean/stdev
    for the length-outlier check, exact-value counting for repeated
    entries.
    """
    flags: list[dict] = []
    enabled = {c["name"] for c in checks_cfg if c.get("enabled")}

    def trigrams(s: str) -> set[str]:
        s = s.lower()
        return {s[i:i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}

    all_name_values = [
        n.value for r in results for n in r.personal_names
        if n.confidence == ConfidenceLevel.CONFIRMED
    ]
    all_trigram_sets = [trigrams(v) for v in all_name_values]

    if "no_ngram_overlap" in enabled:
        for r in results:
            for n in r.personal_names:
                if n.confidence != ConfidenceLevel.CONFIRMED:
                    continue
                own_tg = trigrams(n.value)
                overlaps_any = any(
                    own_tg & other for other in all_trigram_sets if other is not own_tg
                )
                if not overlaps_any and len(all_trigram_sets) > 1:
                    flags.append({
                        "file_path": r.file_path,
                        "flag_type": "no_ngram_overlap",
                        "detail": f"Name {n.value!r} shares no character trigrams with "
                                  f"any other CONFIRMED name in this batch.",
                        "severity": "medium",
                    })

    if "length_outlier" in enabled and len(results) > 3:
        lengths = [r.raw_model_output_len for r in results]
        mean = sum(lengths) / len(lengths)
        variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
        stdev = variance ** 0.5
        for r in results:
            if stdev > 0 and abs(r.raw_model_output_len - mean) > 2 * stdev:
                flags.append({
                    "file_path": r.file_path,
                    "flag_type": "length_outlier",
                    "detail": f"Output length {r.raw_model_output_len} deviates >2 stdev "
                              f"from batch mean {mean:.0f} (stdev {stdev:.0f}).",
                    "severity": "low",
                })

    if "repeated_entry" in enabled:
        for r in results:
            seen: dict[str, int] = {}
            for n in r.personal_names:
                seen[n.value] = seen.get(n.value, 0) + 1
            for value, count in seen.items():
                if count >= 3:
                    flags.append({
                        "file_path": r.file_path,
                        "flag_type": "repeated_entry",
                        "detail": f"Name {value!r} appears {count} times within a single "
                                  f"record - possible loop degeneration.",
                        "severity": "high",
                    })

    return flags


def check_uncertain_gate() -> bool:
    """
    Returns True if it's safe to proceed with extraction. Per project
    design: extraction should not run against buckets while
    uncertain_review.csv still has unreviewed rows, since those rows
    represent classification ambiguity that hasn't been resolved by a
    human yet - proceeding anyway risks the same kind of silent
    contamination the review step exists to prevent.
    """
    uncertain_csv = BUCKET_DIR / "uncertain_review.csv"
    if not uncertain_csv.exists():
        return True
    with open(uncertain_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        pending = [r for r in reader if r.get("file_path") and not r.get("error")]
    if pending:
        print(f"\nBLOCKED: {len(pending)} row(s) still pending in "
              f"uncertain_review.csv.")
        print("Run 'python review_uncertain.py' to clear the queue before "
              "extraction.")
        print("(Rows with a logged error, e.g. file-not-found, don't count "
              "against this gate - those are pipeline failures, not "
              "classification ambiguity, and won't block extraction.)")
        return False
    return True


def _check_header_matches(path: Path, expected_fields: list[str]) -> None:
    """
    Guards against the exact corruption seen in practice: this file was
    created by an older pipeline version with fewer output fields (e.g.
    before device_map/max_memory/oom_recovered were added to
    ExtractionResult), then appended to by a newer version whose rows
    have more columns than the file's header. pandas/Excel would then
    misalign every value in those extra columns against the wrong
    header entirely - a silent data-corruption bug, not just an
    accumulation-of-duplicates annoyance.

    Raises with a clear fix instruction rather than attempting to
    auto-migrate the file, since silently rewriting someone's existing
    extraction data is a worse failure mode than making them re-run.
    """
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            existing_header = next(reader)
        except StopIteration:
            return
    if existing_header != expected_fields:
        raise RuntimeError(
            f"\nSCHEMA MISMATCH: {path} has an outdated header from a "
            f"previous pipeline version.\n"
            f"  File header:     {existing_header}\n"
            f"  Expected header: {expected_fields}\n"
            f"Appending now would misalign columns and corrupt the file.\n"
            f"Fix: delete {path} (and its sibling files in the same "
            f"directory, since they're usually regenerated together) "
            f"and re-run. This does not delete your source images or "
            f"bucket classifications, only the extraction output.\n"
        )


def run() -> None:
    if not check_uncertain_gate():
        sys.exit(1)

    pipeline_cfg = load_pipeline_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "extracted.csv"
    failed_path = OUTPUT_DIR / "failed_extraction.csv"
    flags_path = OUTPUT_DIR / "anomaly_flags.csv"

    _check_header_matches(output_path, OUTPUT_FIELDS)
    _check_header_matches(failed_path, FAILED_FIELDS)
    _check_header_matches(flags_path, ["file_path", "flag_type", "detail", "severity"])

    out_f = open(output_path, "a", newline="", encoding="utf-8")
    out_writer = csv.DictWriter(out_f, fieldnames=OUTPUT_FIELDS)
    if output_path.stat().st_size == 0:
        out_writer.writeheader()

    failed_f = open(failed_path, "a", newline="", encoding="utf-8")
    failed_writer = csv.DictWriter(failed_f, fieldnames=FAILED_FIELDS)
    if failed_path.stat().st_size == 0:
        failed_writer.writeheader()

    flags_f = open(flags_path, "a", newline="", encoding="utf-8")
    flags_writer = csv.DictWriter(flags_f, fieldnames=["file_path", "flag_type", "detail", "severity"])
    if flags_path.stat().st_size == 0:
        flags_writer.writeheader()

    for category in DocumentCategory:
        if category == DocumentCategory.UNCERTAIN:
            continue  # human review queue, no automated extraction

        bucket_cfg = pipeline_cfg["buckets"].get(category.value)
        if not bucket_cfg or not bucket_cfg.get("model"):
            continue  # bucket explicitly has no assigned model yet

        bucket_csv = BUCKET_DIR / f"{category.value}.csv"
        file_paths = read_bucket_file_paths(bucket_csv)
        if not file_paths:
            continue

        print(f"\n=== {category.value} ({len(file_paths)} files, "
              f"model={bucket_cfg['model']}) ===")

        try:
            loader = get_or_build_loader(bucket_cfg["model"], bucket_cfg["prompt_file"])
        except (FileNotFoundError, ValueError) as e:
            print(f"  -> SKIPPING bucket '{category.value}': {e}")
            for file_path in file_paths:
                failed_writer.writerow({
                    "file_path": file_path,
                    "category": category.value,
                    "model": bucket_cfg["model"],
                    "error": f"Bucket skipped, could not build loader: {e}"[:300],
                })
            continue

        bucket_results: list[ExtractionResult] = []

        for i, file_path in enumerate(file_paths, 1):
            print(f"[{i}/{len(file_paths)}] {file_path}")
            loader._oom_recovered = False  # reset per-image, don't let a
                                             # prior image's recovery leak
                                             # into this one's record

            def attempt_extract():
                with Image.open(file_path) as raw_image:
                    return loader.extract(file_path, category.value, raw_image)

            try:
                try:
                    result = attempt_extract()
                except torch.cuda.OutOfMemoryError:
                    print(f"  -> CUDA OOM, clearing cache and retrying once...")
                    torch.cuda.empty_cache()
                    loader._oom_recovered = True
                    result = attempt_extract()  # second failure propagates normally

                bucket_results.append(result)
                out_writer.writerow(extraction_result_to_row(result))
            except (ValueError, ValidationError) as e:
                print(f"  -> FAILED (parse/schema): {e}")
                failed_writer.writerow({
                    "file_path": file_path,
                    "category": category.value,
                    "model": bucket_cfg["model"],
                    "error": str(e)[:300],
                })
            except torch.cuda.OutOfMemoryError as e:
                # Reached only if the retry above also OOM'd - genuinely
                # too large for available memory even after cache clear
                # and CPU-offload headroom. Log and move on rather than
                # losing the rest of the batch.
                print(f"  -> FAILED (CUDA OOM, retry also failed): {e}")
                failed_writer.writerow({
                    "file_path": file_path,
                    "category": category.value,
                    "model": bucket_cfg["model"],
                    "error": f"CUDA OOM (retry also failed): {str(e)[:230]}",
                })
                torch.cuda.empty_cache()

        flags = run_anomaly_checks(bucket_results, pipeline_cfg["anomaly_checks"])
        for flag in flags:
            flags_writer.writerow(flag)
        if flags:
            print(f"  {len(flags)} anomaly flag(s) written for this bucket.")

    out_f.close()
    failed_f.close()
    flags_f.close()

    print(f"\nDone.")
    print(f"  Extracted records: {output_path}")
    print(f"  Failed extractions: {failed_path}")
    print(f"  Anomaly flags: {flags_path}")


if __name__ == "__main__":
    run()
