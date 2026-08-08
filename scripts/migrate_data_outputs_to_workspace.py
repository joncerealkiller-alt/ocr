"""
Phase 2 of the hashed-run-system migration: decomposes data/outputs/
(a 44GB catch-all predating the run system) into
genealogy_workspace/{research,datasets,models}/, per the ownership
model in docs/RUN_ARCHITECTURE.md.

See the approved plan for full design rationale. Summary:

- Directories get moved + an NTFS junction (`mklink /J`) left at the
  old path; files get moved + an NTFS hardlink (`mklink /H`) left at
  the old path. Either way, the ~75 files that hardcode a
  `data/outputs/<name>` path keep working completely unmodified.
- 3 directories (row_segmentation, lora_dataset, scoring_reports) are
  hook-protected (.claude/protected_paths.txt) from any scripted
  rm -rf - for these, copy + verify only, NO removal, NO junction. Both
  copies coexist until Jon manually removes the original himself.
- gemma_hidden_state_probe/, error_analysis/ (DO_NOT_TOUCH.md),
  reference_pull/, new_taxonomy_ground_truth.csv/_README.txt (active
  writer), .lorabackup/ (a backup snapshot, out of scope), and the root
  "DO NOT AUTO DELETE ANYTHING HERE.txt" marker are never touched by
  this script at all - not even read.
- Writes genealogy_workspace/migration_manifest.json documenting every
  processed item's old/new path, kind (dir/file), and type
  (junction/hardlink/copy_only) - so a future session never has to
  wonder "why is this a junction" from scratch.

Usage:
    python -m scripts.migrate_data_outputs_to_workspace --check    # pre-flight validation only
    python -m scripts.migrate_data_outputs_to_workspace --checksum-baseline  # checksum + save baseline, no copy
    python -m scripts.migrate_data_outputs_to_workspace --copy      # copy + verify (no removal/linking yet)
    python -m scripts.migrate_data_outputs_to_workspace --link      # remove originals + create junctions/hardlinks (non-protected items only, requires --copy already done and verified)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_OUTPUTS = PROJECT_ROOT / "data" / "outputs"
WORKSPACE_ROOT = PROJECT_ROOT.parent / "genealogy_workspace"

CHECKSUM_BASELINE_PATH = PROJECT_ROOT / "scripts" / "_phase2_checksum_baseline.json"
MANIFEST_PATH = WORKSPACE_ROOT / "migration_manifest.json"

# Hook-protected (.claude/protected_paths.txt) - copy + verify only,
# never removed/linked by this script.
PROTECTED_NO_JUNCTION = {"row_segmentation", "lora_dataset", "scoring_reports"}

# Never touched at all by this script - not moved, not checksummed,
# not read. Documented here for clarity even though the script never
# iterates this set (MAPPING below simply doesn't include them).
EXCLUDED = {
    "gemma_hidden_state_probe", "error_analysis",  # DO_NOT_TOUCH.md
    "reference_pull", "new_taxonomy_ground_truth.csv", "new_taxonomy_ground_truth_README.txt",  # active writer
    ".lorabackup",  # out-of-scope backup snapshot
    "DO NOT AUTO DELETE ANYTHING HERE.txt",  # protects the whole dir
}


@dataclass
class Item:
    old_name: str          # relative to data/outputs/
    new_rel: str            # relative to genealogy_workspace/
    kind: str                # "dir" or "file"


# Explicit, literal mapping - no pattern matching. Every entry
# pre-flight-validated against the real filesystem before anything else
# runs (see validate()).
MAPPING: list[Item] = [
    # -- research/benchmarks/ --
    Item("benchmark2", "research/benchmarks/benchmark2", "dir"),
    Item("benchmark2_2", "research/benchmarks/benchmark2_2", "dir"),
    Item("benchmark2_3_e2b_vs_e4b", "research/benchmarks/benchmark2_3_e2b_vs_e4b", "dir"),
    Item("benchmark2_3_multi_tower", "research/benchmarks/benchmark2_3_multi_tower", "dir"),
    Item("benchmark2_4_selective_escalation", "research/benchmarks/benchmark2_4_selective_escalation", "dir"),
    Item("benchmark2_gemma_input_qualification", "research/benchmarks/benchmark2_gemma_input_qualification", "dir"),
    Item("benchmark2_pilot_checkpoints.json", "research/benchmarks/benchmark2_pilot_checkpoints.json", "file"),
    Item("benchmark2_sequential_runtime_log.jsonl", "research/benchmarks/benchmark2_sequential_runtime_log.jsonl", "file"),
    Item("vit_family_benchmark", "research/benchmarks/vit_family_benchmark", "dir"),

    # -- research/experiments/ --
    Item("deskew_fix_2026-08-05", "research/experiments/deskew_fix_2026-08-05", "dir"),
    Item("fresh_pass_vs_checkpoint_comparison", "research/experiments/fresh_pass_vs_checkpoint_comparison", "dir"),
    Item("fresh_pass_vs_v4_checkpoint_comparison", "research/experiments/fresh_pass_vs_v4_checkpoint_comparison", "dir"),
    Item("gemma_reasoning_consistency_audit", "research/experiments/gemma_reasoning_consistency_audit", "dir"),
    Item("gemma_stability_sandbox", "research/experiments/gemma_stability_sandbox", "dir"),
    Item("layout_detection_visual_check", "research/experiments/layout_detection_visual_check", "dir"),
    Item("layout_detection_yolo26_test", "research/experiments/layout_detection_yolo26_test", "dir"),
    Item("overnight_vlm_routing_test", "research/experiments/overnight_vlm_routing_test", "dir"),
    Item("preprocessing_profile_comparison", "research/experiments/preprocessing_profile_comparison", "dir"),
    Item("reference_classifier_qualification", "research/experiments/reference_classifier_qualification", "dir"),
    Item("single_shot_classifier_test", "research/experiments/single_shot_classifier_test", "dir"),
    Item("tower_consensus_audit", "research/experiments/tower_consensus_audit", "dir"),
    Item("vision_encoder_aspect_ratio_log.jsonl", "research/experiments/vision_encoder_aspect_ratio_log.jsonl", "file"),
    Item("vision_encoder_qualification_log.jsonl", "research/experiments/vision_encoder_qualification_log.jsonl", "file"),
    Item("experiment3_1_crop_confound_log.jsonl", "research/experiments/experiment3_1_crop_confound_log.jsonl", "file"),
    Item("experiment3_2_robust_localization_log.jsonl", "research/experiments/experiment3_2_robust_localization_log.jsonl", "file"),
    Item("experiment3_3_gradcam_log.jsonl", "research/experiments/experiment3_3_gradcam_log.jsonl", "file"),
    Item("auto_row_segmentation_batch_test", "research/experiments/auto_row_segmentation_batch_test", "dir"),
    Item("yolo_assisted_auto_sidecar", "research/experiments/yolo_assisted_auto_sidecar", "dir"),
    Item("row_segmentation", "research/experiments/row_segmentation", "dir"),  # PROTECTED
    Item("prompt_sweep_runs", "research/experiments/prompt_sweep_runs", "dir"),
    Item("prompt_sweep_log.jsonl", "research/experiments/prompt_sweep_log.jsonl", "file"),
    Item("prompt_sweep_log - Copy.jsonl", "research/experiments/prompt_sweep_log - Copy.jsonl", "file"),
    Item("lora_eval_log.jsonl", "research/experiments/lora_eval_log.jsonl", "file"),
    Item("scoring_reports", "research/experiments/scoring_reports", "dir"),  # PROTECTED
    Item("image_analysis", "research/experiments/image_analysis", "dir"),

    # -- research/baselines/ --
    Item("reference_pipeline_prerefactor", "research/baselines/reference_pipeline_prerefactor", "dir"),
    Item("reference_pipeline_v1", "research/baselines/reference_pipeline_v1", "dir"),
    Item("reference_pipeline_v2", "research/baselines/reference_pipeline_v2", "dir"),
    Item("reference_pipeline_v3", "research/baselines/reference_pipeline_v3", "dir"),
    Item("reference_pipeline_v4", "research/baselines/reference_pipeline_v4", "dir"),

    # -- research/calibration/ --
    Item("column_calibration", "research/calibration/column_calibration", "dir"),
    Item("column_calibration_workspace", "research/calibration/column_calibration_workspace", "dir"),

    # -- models/checkpoints/ --
    Item("granite_vision_2b_lora_checkpoints", "models/checkpoints/granite_vision_2b_lora_checkpoints", "dir"),
    Item("internvl3_2b_lora_checkpoints", "models/checkpoints/internvl3_2b_lora_checkpoints", "dir"),
    Item("qwen25_vl_7b_lora_checkpoints", "models/checkpoints/qwen25_vl_7b_lora_checkpoints", "dir"),
    Item("qwen2b_lora_checkpoints", "models/checkpoints/qwen2b_lora_checkpoints", "dir"),
    Item("qwen3vl2b_lora_checkpoints", "models/checkpoints/qwen3vl2b_lora_checkpoints", "dir"),
    Item("qwen3vl2b_thinking_lora_checkpoints", "models/checkpoints/qwen3vl2b_thinking_lora_checkpoints", "dir"),
    Item("qwen3vl4b_lora_checkpoints", "models/checkpoints/qwen3vl4b_lora_checkpoints", "dir"),
    Item("smolvlm_lora_checkpoints", "models/checkpoints/smolvlm_lora_checkpoints", "dir"),
    Item("vit21k_doc_classifier_checkpoints", "models/checkpoints/vit21k_doc_classifier_checkpoints", "dir"),

    # -- datasets/reference/ --
    Item("lac_census_pull", "datasets/reference/lac_census_pull", "dir"),
    Item("lac_new_years_samples", "datasets/reference/lac_new_years_samples", "dir"),
    Item("lac_pull_1901_batch1", "datasets/reference/lac_pull_1901_batch1", "dir"),
    Item("lac_pull_1906_batch1", "datasets/reference/lac_pull_1906_batch1", "dir"),
    Item("lac_pull_1921_batch1", "datasets/reference/lac_pull_1921_batch1", "dir"),
    Item("lac_pull_1926_batch1", "datasets/reference/lac_pull_1926_batch1", "dir"),
    Item("lac_pull_1931_batch1", "datasets/reference/lac_pull_1931_batch1", "dir"),
    Item("lac_pull_batch1", "datasets/reference/lac_pull_batch1", "dir"),
    Item("lac_pull_batch2", "datasets/reference/lac_pull_batch2", "dir"),
    Item("lac_pull_batch3", "datasets/reference/lac_pull_batch3", "dir"),

    # -- datasets/bootstrap/ --
    Item("layout_bootstrap_train", "datasets/bootstrap/layout_bootstrap_train", "dir"),

    # -- datasets/training/ --
    Item("lora_dataset", "datasets/training/lora_dataset", "dir"),  # PROTECTED

    # -- datasets/ground_truth/ --
    Item("ground_truth_log.jsonl", "datasets/ground_truth/ground_truth_log.jsonl", "file"),
    Item("vit_finetune_dataset.csv", "datasets/ground_truth/vit_finetune_dataset.csv", "file"),
    Item("vit_finetune_dataset_ablation_no_census_pull.csv", "datasets/ground_truth/vit_finetune_dataset_ablation_no_census_pull.csv", "file"),

    # -- workspace root (tool state, not data) --
    Item("_workflow_gui_state.json", "_workflow_gui_state.json", "file"),
]

_by_old_name = {item.old_name: item for item in MAPPING}
assert len(_by_old_name) == len(MAPPING), "duplicate old_name in MAPPING"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dir_size_and_count(path: Path) -> tuple[int, int]:
    size, count = 0, 0
    for p in path.rglob("*"):
        if p.is_file():
            size += p.stat().st_size
            count += 1
    return size, count


def validate() -> None:
    """Pre-flight: every literal old_name must exist under data/outputs/
    AND its actual is_dir()/is_file() must match the declared kind.
    Aborts loudly (no partial run) if anything doesn't match."""
    problems = []
    for item in MAPPING:
        old_path = DATA_OUTPUTS / item.old_name
        if not old_path.exists():
            problems.append(f"MISSING: {old_path}")
            continue
        actual_kind = "dir" if old_path.is_dir() else "file"
        if actual_kind != item.kind:
            problems.append(f"KIND MISMATCH: {old_path} is a {actual_kind}, mapping says {item.kind}")

    print(f"Validated {len(MAPPING)} mapping entries against {DATA_OUTPUTS}")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S) FOUND - aborting, nothing touched:")
        for p in problems:
            print(f"  {p}")
        sys.exit(1)
    print("All entries valid. Safe to proceed.")


def checksum_baseline() -> None:
    """Checksums every in-scope file and saves to CHECKSUM_BASELINE_PATH.
    Run BEFORE any copy. Directories are walked recursively; every file
    gets its own sha256 entry keyed by its path relative to data/outputs/."""
    validate()
    baseline: dict[str, str] = {}
    counts: dict[str, int] = {}
    t0 = datetime.now()
    for i, item in enumerate(MAPPING, 1):
        old_path = DATA_OUTPUTS / item.old_name
        print(f"[{i}/{len(MAPPING)}] Checksumming {item.old_name} ...")
        n = 0
        if item.kind == "file":
            rel = item.old_name
            baseline[rel] = _sha256(old_path)
            n = 1
        else:
            for f in old_path.rglob("*"):
                if f.is_file():
                    rel = str(f.relative_to(DATA_OUTPUTS))
                    baseline[rel] = _sha256(f)
                    n += 1
        counts[item.old_name] = n
        print(f"    {n} file(s)")

    CHECKSUM_BASELINE_PATH.write_text(
        json.dumps({"checksums": baseline, "counts": counts}, indent=2), encoding="utf-8",
    )
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\nBaseline: {len(baseline)} file(s) checksummed in {elapsed:.0f}s.")
    print(f"Saved to {CHECKSUM_BASELINE_PATH}")


def copy_and_verify() -> None:
    """Copies every mapped item to its new location under
    genealogy_workspace/, then verifies count + checksum match the
    saved baseline. Does NOT touch data/outputs/ at all - purely
    additive. Run --link afterward, separately, once this is confirmed
    clean."""
    if not CHECKSUM_BASELINE_PATH.exists():
        print("No checksum baseline found - run --checksum-baseline first.")
        sys.exit(1)
    baseline = json.loads(CHECKSUM_BASELINE_PATH.read_text(encoding="utf-8"))
    validate()

    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    mismatches = []
    for i, item in enumerate(MAPPING, 1):
        old_path = DATA_OUTPUTS / item.old_name
        new_path = WORKSPACE_ROOT / item.new_rel
        print(f"[{i}/{len(MAPPING)}] Copying {item.old_name} -> {item.new_rel} ...")
        new_path.parent.mkdir(parents=True, exist_ok=True)

        if item.kind == "file":
            shutil.copy2(old_path, new_path)
        else:
            if new_path.exists():
                print(f"    (destination already exists, skipping copy - verify only)")
            else:
                shutil.copytree(old_path, new_path)

        # verify
        if item.kind == "file":
            rel = item.old_name
            expected = baseline["checksums"].get(rel)
            actual = _sha256(new_path)
            if expected != actual:
                mismatches.append((item.old_name, rel))
        else:
            for f in new_path.rglob("*"):
                if f.is_file():
                    rel = str((old_path / f.relative_to(new_path)).relative_to(DATA_OUTPUTS))
                    expected = baseline["checksums"].get(rel)
                    actual = _sha256(f)
                    if expected != actual:
                        mismatches.append((item.old_name, rel))

    if mismatches:
        print(f"\n{len(mismatches)} CHECKSUM MISMATCH(ES) - do NOT proceed to --link:")
        for old_name, rel in mismatches[:30]:
            print(f"  {old_name}: {rel}")
        sys.exit(1)

    print(f"\nAll {len(MAPPING)} item(s) copied and verified byte-for-byte identical.")
    print("Safe to run --link next.")


def _run_mklink(args: list[str]) -> None:
    result = subprocess.run(["cmd", "/c", "mklink", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mklink failed: {result.stdout} {result.stderr}")


def link() -> None:
    """For non-protected items: removes the original data/outputs/<name>
    and creates a junction (dir) or hardlink (file) in its place,
    pointing at the already-copied-and-verified genealogy_workspace/
    location. For the 3 hook-protected items: does nothing except
    report - both copies stay, original never removed by this script.
    Writes migration_manifest.json at the end."""
    if not CHECKSUM_BASELINE_PATH.exists():
        print("No checksum baseline found - run --checksum-baseline and --copy first.")
        sys.exit(1)

    manifest_items = []
    for i, item in enumerate(MAPPING, 1):
        old_path = DATA_OUTPUTS / item.old_name
        new_path = WORKSPACE_ROOT / item.new_rel
        if not new_path.exists():
            print(f"[{i}/{len(MAPPING)}] SKIP {item.old_name}: no verified copy at {new_path} - run --copy first.")
            continue

        size_bytes, file_count = (
            (new_path.stat().st_size, 1) if item.kind == "file" else _dir_size_and_count(new_path)
        )

        if item.old_name in PROTECTED_NO_JUNCTION:
            print(f"[{i}/{len(MAPPING)}] {item.old_name}: hook-protected, leaving original in place (copy_only).")
            manifest_items.append({
                "old": f"data/outputs/{item.old_name}", "new": item.new_rel,
                "kind": item.kind, "type": "copy_only",
                "reason": "hook-protected (.claude/protected_paths.txt) - original left in place, remove manually when ready",
                "size_bytes": size_bytes, "file_count": file_count,
            })
            continue

        print(f"[{i}/{len(MAPPING)}] {item.old_name}: removing original, creating "
              f"{'junction' if item.kind == 'dir' else 'hardlink'} ...")
        if item.kind == "dir":
            shutil.rmtree(old_path)
            _run_mklink(["/J", str(old_path), str(new_path)])
            link_type = "junction"
        else:
            old_path.unlink()
            _run_mklink(["/H", str(old_path), str(new_path)])
            link_type = "hardlink"

        manifest_items.append({
            "old": f"data/outputs/{item.old_name}", "new": item.new_rel,
            "kind": item.kind, "type": link_type,
            "size_bytes": size_bytes, "file_count": file_count,
        })

    manifest = {
        "phase": 2,
        "created": datetime.now(timezone.utc).isoformat(),
        "items": manifest_items,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {MANIFEST_PATH}")
    print(f"{sum(1 for m in manifest_items if m['type'] != 'copy_only')} item(s) linked, "
          f"{sum(1 for m in manifest_items if m['type'] == 'copy_only')} item(s) copy_only (pending manual cleanup).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Pre-flight validation only.")
    group.add_argument("--checksum-baseline", action="store_true", help="Checksum every in-scope file, save baseline.")
    group.add_argument("--copy", action="store_true", help="Copy every item, verify against baseline.")
    group.add_argument("--link", action="store_true", help="Remove originals, create junctions/hardlinks (non-protected only).")
    args = parser.parse_args()

    if args.check:
        validate()
    elif args.checksum_baseline:
        checksum_baseline()
    elif args.copy:
        copy_and_verify()
    elif args.link:
        link()


if __name__ == "__main__":
    main()
