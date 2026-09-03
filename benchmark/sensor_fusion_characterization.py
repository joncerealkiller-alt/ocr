"""
Sensor Fusion Characterization (2026-08-03).

Extends benchmark/tower_ensemble_characterization.py's ground-truth-free
discipline across ALL THREE recorded sensor families, not just the
vision towers - per Jon's framing: Stage 2 (core/decision_engine.py) is
becoming an evidence-fusion engine, and this pass characterizes how the
sensors it now has access to relate to each other, so that a future
policy pass reasons over real relationships instead of assumptions.

THREE SENSOR FAMILIES, per-image:
  semantic  - <name>_tower_consensus.json (8 encoders: winner, margin,
              consensus category - core/decision_engine.py)
  physical  - <name>_analysis.json (table/page detection, blur,
              contrast, illumination, ink fraction, stroke/text
              measurements - core/image_analysis.py)
  metadata  - data/pipeline.db `images` row (source_type, page_number,
              identity_hash vs current_hash, bucket, stage history)

NO GROUND TRUTH, STILL. Same reasoning as tower_ensemble_
characterization.py: the only labels that will ever exist in production
are the human-review queue's sparse, biased sample (see the project
memory this session recorded on that point) - never a comprehensive
corpus-wide label set. Every measurement below is an objective,
computable-today property (correlation, co-occurrence, group
differences) - never a correctness claim. "Document profile" groupings
near the bottom are DESCRIPTIVE BUCKETING for characterization, not a
proposed routing policy - no threshold here is presented as a decision
rule.

Usage:
    python -m benchmark.sensor_fusion_characterization
"""
from __future__ import annotations

import itertools
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from core.pipeline_db import PipelineDatabase, DEFAULT_DB_PATH

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_CONSENSUS_ORDER = {"complete_disagreement": 0, "split": 1, "majority": 2, "unanimous": 3}


def _roi_block(analysis: dict) -> dict:
    """Same ROI-preference as core/image_analysis.py's own print
    statement: table region if detected, else page, else frame."""
    regions = analysis.get("regions") or {}
    return regions.get("table") or regions.get("page") or regions.get("frame") or {}


def load_joined_records(db_path: Path = DEFAULT_DB_PATH) -> list[dict]:
    """One dict per image with whatever sensor data is actually present -
    missing sidecars simply leave those fields absent, never faked."""
    db = PipelineDatabase(db_path)
    images = db.list_images()
    records = []

    for image in images:
        working_path = Path(image["working_path"])
        rec = {
            "working_path": str(working_path),
            "source_type": image["source_type"],
            "page_number": image["page_number"],
            "bucket": image["bucket"],
            "tower_consensus_category": image["tower_consensus_category"],
            "current_stage": image["current_stage"],
            "status": image["status"],
            "hash_changed": (
                image["current_hash"] is not None
                and image["current_hash"] != image["identity_hash"]
            ),
        }

        analysis_path = working_path.with_name(f"{working_path.stem}_analysis.json")
        if analysis_path.exists():
            try:
                analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
                roi = _roi_block(analysis)
                rec["physical"] = {
                    "table_confidence": analysis.get("table_confidence"),
                    "page_confidence": analysis.get("page_confidence"),
                    "page_method": analysis.get("page_method"),
                    "has_table_boundary": analysis.get("table_boundary") is not None,
                    "has_page_boundary": analysis.get("page_boundary") is not None,
                    "blur_laplacian_var": analysis.get("blur_laplacian_var"),
                    "noise_residual_std": analysis.get("noise_residual_std"),
                    "abs_deskew_angle": abs(analysis["deskew_angle_deg"])
                    if analysis.get("deskew_angle_deg") is not None else None,
                    "ruling_lines_vertical": analysis.get("ruling_lines_vertical"),
                    "ruling_lines_horizontal": analysis.get("ruling_lines_horizontal"),
                    "roi_contrast": roi.get("contrast_p5_p95_spread"),
                    "roi_illumination": roi.get("illumination_unevenness"),
                    "roi_ink_fraction": roi.get("ink_fraction"),
                    "roi_stroke_width_px": roi.get("stroke_width_px"),
                    "roi_text_height_px": roi.get("text_height_px"),
                }
            except Exception:
                pass

        tc_path = working_path.with_name(f"{working_path.stem}_tower_consensus.json")
        if tc_path.exists():
            try:
                tc = json.loads(tc_path.read_text(encoding="utf-8"))
                towers = tc.get("towers", {})
                margins = [t["margin"] for t in towers.values() if t.get("margin") is not None]
                votes = [t["winner"] for t in towers.values()]
                rec["semantic"] = {
                    "consensus_category": tc.get("consensus_category"),
                    "avg_margin": sum(margins) / len(margins) if margins else None,
                    "n_distinct_votes": len(set(votes)) if votes else None,
                    "votes": votes,
                    "top_bucket": Counter(votes).most_common(1)[0][0] if votes else None,
                }
            except Exception:
                pass

        stage_outputs = db.get_stage_outputs(image["id"])
        rec["n_stage_events"] = len(stage_outputs)
        rec["stages_seen"] = sorted(set(s["stage"] for s in stage_outputs))

        records.append(rec)

    return records


# -- numeric extraction for correlation ------------------------------------

_NUMERIC_FIELDS = [
    ("physical", "table_confidence"),
    ("physical", "page_confidence"),
    ("physical", "blur_laplacian_var"),
    ("physical", "noise_residual_std"),
    ("physical", "abs_deskew_angle"),
    ("physical", "roi_contrast"),
    ("physical", "roi_illumination"),
    ("physical", "roi_ink_fraction"),
    ("physical", "roi_stroke_width_px"),
    ("physical", "roi_text_height_px"),
    ("semantic", "avg_margin"),
    ("semantic", "n_distinct_votes"),
]


def _numeric_value(rec: dict, family: str, field: str):
    block = rec.get(family)
    if not block:
        return None
    return block.get(field)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx == 0 or sy == 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / (sx * sy)


def correlation_matrix(records: list[dict]) -> dict[tuple[str, str], float | None]:
    result = {}
    for (f1, n1), (f2, n2) in itertools.combinations(_NUMERIC_FIELDS, 2):
        pairs = [
            (_numeric_value(r, f1, n1), _numeric_value(r, f2, n2))
            for r in records
        ]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if len(pairs) < 10:
            result[(f"{f1}.{n1}", f"{f2}.{n2}")] = None
            continue
        xs, ys = zip(*pairs)
        result[(f"{f1}.{n1}", f"{f2}.{n2}")] = pearson(list(xs), list(ys))
    return result


def group_means_by_consensus_category(records: list[dict]) -> dict:
    """Physical measurement means, grouped by semantic consensus_category
    - is semantic ambiguity accompanied by any distinctive physical
    signature, or does it look physically like everything else?"""
    groups = defaultdict(lambda: defaultdict(list))
    for r in records:
        sem = r.get("semantic")
        phys = r.get("physical")
        if not sem or not phys:
            continue
        cat = sem.get("consensus_category")
        if cat is None:
            continue
        for _, field in _NUMERIC_FIELDS:
            if _ == "physical":
                pass
        for field in ["table_confidence", "page_confidence", "blur_laplacian_var",
                      "roi_contrast", "roi_illumination", "roi_ink_fraction",
                      "abs_deskew_angle"]:
            val = phys.get(field)
            if val is not None:
                groups[cat][field].append(val)
    return {
        cat: {field: round(sum(vals) / len(vals), 4) for field, vals in fields.items()}
        for cat, fields in groups.items()
    }


def group_means_by_boundary_pair(records: list[dict], pairs: list[tuple[str, str]]) -> dict:
    """Physical measurement means for images whose top semantic vote pair
    (winner, runner-up bucket across the corpus - reuses the dominant
    confusions found in tower_ensemble_characterization.py) matches one
    of the given dominant boundary pairs, vs. the corpus-wide baseline."""
    baseline = defaultdict(list)
    by_pair = {p: defaultdict(list) for p in pairs}

    for r in records:
        sem, phys = r.get("semantic"), r.get("physical")
        if not sem or not phys:
            continue
        votes = sem.get("votes") or []
        vote_counts = Counter(votes)
        top2 = [b for b, _ in vote_counts.most_common(2)]
        this_pair = tuple(sorted(top2)) if len(top2) == 2 else None

        for field in ["table_confidence", "roi_contrast", "roi_ink_fraction", "blur_laplacian_var"]:
            val = phys.get(field)
            if val is None:
                continue
            baseline[field].append(val)
            if this_pair in by_pair:
                by_pair[this_pair][field].append(val)

    def summarize(d):
        return {f: round(sum(v) / len(v), 4) for f, v in d.items() if v}

    return {
        "baseline (all images)": summarize(baseline),
        **{f"{p[0]} <-> {p[1]}": summarize(by_pair[p]) for p in pairs},
    }


def coverage_report(records: list[dict]) -> dict:
    n = len(records)
    has_physical = sum(1 for r in records if "physical" in r)
    has_semantic = sum(1 for r in records if "semantic" in r)
    has_both = sum(1 for r in records if "physical" in r and "semantic" in r)
    has_bucket = sum(1 for r in records if r["bucket"] is not None)
    source_types = Counter(r["source_type"] for r in records)
    hash_changed = sum(1 for r in records if r["hash_changed"])
    return {
        "total_images": n,
        "has_physical_sensor": has_physical,
        "has_semantic_sensor": has_semantic,
        "has_both": has_both,
        "has_classification": has_bucket,
        "source_type_distribution": dict(source_types),
        "pixels_modified_since_acquisition": hash_changed,
    }


def page_method_vs_consensus(records: list[dict]) -> dict:
    """Contingency: does 'no page boundary detected at all' (an unusual
    capture, physically) coincide with semantic disagreement?"""
    table = defaultdict(Counter)
    for r in records:
        sem, phys = r.get("semantic"), r.get("physical")
        if not sem or not phys:
            continue
        method = phys.get("page_method") or "none"
        cat = sem.get("consensus_category")
        if cat is not None:
            table[method][cat] += 1
    return {method: dict(counts) for method, counts in table.items()}


def hash_changed_vs_semantic(records: list[dict]) -> dict:
    """Does whether preprocessing has touched this image's pixels
    (hash_changed) relate to semantic margin/consensus at all?"""
    groups = defaultdict(list)
    cat_groups = defaultdict(Counter)
    for r in records:
        sem = r.get("semantic")
        if not sem or sem.get("avg_margin") is None:
            continue
        groups[r["hash_changed"]].append(sem["avg_margin"])
        if sem.get("consensus_category"):
            cat_groups[r["hash_changed"]][sem["consensus_category"]] += 1
    return {
        "avg_margin_by_hash_changed": {
            str(k): round(sum(v) / len(v), 4) for k, v in groups.items() if v
        },
        "consensus_category_by_hash_changed": {
            str(k): dict(v) for k, v in cat_groups.items()
        },
    }


def document_profiles(records: list[dict]) -> dict:
    """
    DESCRIPTIVE bucketing only (not a routing policy) combining signals
    from all three families, using simple percentile-based cutoffs
    computed FROM this corpus, purely to characterize how often each
    profile shape occurs and how much the families overlap in flagging
    it - not to prescribe what should happen to any of them.
    """
    complete = [r for r in records if "physical" in r and "semantic" in r]
    if not complete:
        return {}

    margins = [r["semantic"]["avg_margin"] for r in complete if r["semantic"].get("avg_margin") is not None]
    blurs = [r["physical"]["blur_laplacian_var"] for r in complete if r["physical"].get("blur_laplacian_var") is not None]
    margins.sort()
    blurs.sort()

    def pct(sorted_vals, p):
        if not sorted_vals:
            return None
        idx = min(len(sorted_vals) - 1, int(p * len(sorted_vals)))
        return sorted_vals[idx]

    margin_low = pct(margins, 0.25)
    blur_low = pct(blurs, 0.10)

    profiles = Counter()
    overlap = Counter()
    for r in complete:
        sem, phys = r["semantic"], r["physical"]
        cat = sem.get("consensus_category")
        margin = sem.get("avg_margin")
        blur = phys.get("blur_laplacian_var")
        no_boundary = not phys.get("has_page_boundary") and not phys.get("has_table_boundary")
        n_votes = sem.get("n_distinct_votes")

        tags = []
        if cat == "unanimous" and margin is not None and margin_low is not None and margin > margin_low:
            tags.append("straightforward")
        if cat in ("split", "complete_disagreement"):
            tags.append("ambiguous")
        if (blur is not None and blur_low is not None and blur < blur_low) or no_boundary:
            tags.append("unusual_capture")
        if n_votes is not None and n_votes >= 6:
            tags.append("novel_high_vote_spread")
        if r["bucket"] == "uncertain_review":
            tags.append("already_review_queue")

        if not tags:
            tags.append("unremarkable")
        for t in tags:
            profiles[t] += 1
        overlap[tuple(sorted(tags))] += 1

    return {
        "profile_counts": dict(profiles),
        "profile_combinations": {" + ".join(k): v for k, v in overlap.most_common(20)},
        "thresholds_used_for_this_characterization_only": {
            "margin_low (25th pct of avg_margin)": margin_low,
            "blur_low (10th pct of blur_laplacian_var)": blur_low,
        },
    }


def main() -> None:
    records = load_joined_records()
    n = len(records)
    print(f"Loaded {n} images from {DEFAULT_DB_PATH}.\n")

    print("=" * 78)
    print("[A] EVIDENCE")
    print("=" * 78)

    print("\n--- Coverage: which sensor families are actually present ---")
    cov = coverage_report(records)
    for k, v in cov.items():
        print(f"  {k}: {v}")

    print("\n--- Cross-family correlations (Pearson r, |r| >= 0.15 shown) ---")
    corr = correlation_matrix(records)
    notable = sorted(
        ((k, v) for k, v in corr.items() if v is not None and abs(v) >= 0.15),
        key=lambda kv: -abs(kv[1]),
    )
    for (a, b), r in notable:
        print(f"  {a:<30} <-> {b:<30} r={r:+.3f}")
    if not notable:
        print("  (no pair reached |r| >= 0.15)")

    print("\n--- Physical measurement means, grouped by semantic consensus category ---")
    by_cat = group_means_by_consensus_category(records)
    for cat in ["unanimous", "majority", "split", "complete_disagreement"]:
        if cat in by_cat:
            print(f"  {cat}:")
            for field, val in by_cat[cat].items():
                print(f"    {field:<22} {val}")

    print("\n--- Physical measurement means, by dominant semantic boundary pair ---")
    dominant_pairs = [
        ("dense_tabular_rows", "printed_document"),
        ("mixed_text_image", "printed_document"),
        ("genealogy_chart", "website_screenshot"),
        ("handwritten_ledger", "printed_document"),
    ]
    by_pair = group_means_by_boundary_pair(records, dominant_pairs)
    for label, vals in by_pair.items():
        print(f"  {label}:")
        for field, val in vals.items():
            print(f"    {field:<22} {val}")

    print("\n--- Page-detection method vs semantic consensus category (contingency) ---")
    pm = page_method_vs_consensus(records)
    for method, counts in pm.items():
        total = sum(counts.values())
        parts = ", ".join(f"{k}={v}({100*v/total:.0f}%)" for k, v in counts.items())
        print(f"  page_method={method:<15} n={total:<6} {parts}")

    print("\n--- Pixels-modified-since-acquisition vs semantic signal ---")
    hc = hash_changed_vs_semantic(records)
    for k, v in hc.items():
        print(f"  {k}: {v}")

    print("\n--- Descriptive document profiles (all 3 families combined) ---")
    profiles = document_profiles(records)
    if profiles:
        print("  Counts (an image can carry multiple tags):")
        for tag, count in profiles["profile_counts"].items():
            print(f"    {tag:<28} {count}")
        print("  Most common tag combinations:")
        for combo, count in profiles["profile_combinations"].items():
            print(f"    {combo:<50} {count}")
        print(f"  (characterization-only cutoffs used: "
              f"{profiles['thresholds_used_for_this_characterization_only']})")

    print(f"\nTotal images: {n}")


if __name__ == "__main__":
    main()
