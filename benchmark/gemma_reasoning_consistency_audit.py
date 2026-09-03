"""
Gemma classification-vs-reasoning consistency audit (2026-08-04). For
every image classified in the fresh Stage 5 run, checks whether the
free-text "reason" field is semantically consistent with the predicted
category - an internal-consistency check, not an accuracy benchmark
(core/schema.py's ClassificationResult.reason is capped at one
sentence, 40 words - a real constraint on how much self-contradiction
can physically fit in the field, noted explicitly in the report rather
than glossed over).

MECHANICAL FIRST PASS, not a claim of true LLM-level semantic judgment:
each category has a hand-curated set of strong marker phrases (drawn
from config/taxonomy.yaml's classifier_guidance text plus domain
knowledge). A reason gets tiered by which categories' markers it hits:

  Contradictory     - hits ANOTHER category's markers, not its own
  Consistent        - hits its OWN category's markers (and no strong
                       contradicting marker from elsewhere)
  Ambiguous         - hits markers from 2+ DIFFERENT categories,
                       including possibly its own
  Weak              - hits no category's markers at all (generic/vague)
  Self-Contradictory - a same-text negation/affirmation pattern for the
                       same concept (checked separately, rare given the
                       one-sentence cap)

This mechanical pass is then VALIDATED (not assumed accurate) against a
real human/Claude-read random sample - see the validation section of
the delivered report for the measured agreement rate.

Usage:
    python -m benchmark.gemma_reasoning_consistency_audit
"""
from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUCKET_DIR = PROJECT_ROOT / "data" / "buckets"
DB_PATH = PROJECT_ROOT / "data" / "pipeline.db"
DISAGREEMENTS_CSV = PROJECT_ROOT / "data" / "outputs" / "tower_consensus_audit" / "20260804T132128Z" / "disagreements.csv"
MISCLASSIFICATIONS_CSV = PROJECT_ROOT / "data" / "outputs" / "reference_pipeline_v3" / "misclassifications.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "gemma_reasoning_consistency_audit"

ALL_BUCKETS = [
    "dense_tabular_rows", "genealogy_chart", "handwritten_ledger", "map_land_record",
    "printed_document", "mixed_text_image", "portrait_photo", "website_screenshot",
    "photo_collage", "casual_photo", "cemetery_photo",
]

# Hand-curated strong marker phrases per category - deliberately narrow
# (precision over recall): a hit here is meant to be a confident signal
# that the TEXT is describing that category's defining trait, not a
# loose word-association. Drawn from config/taxonomy.yaml's
# classifier_guidance text plus the concept list the task itself named
# ("photograph, portrait, website, table, manifest, map, ledger,
# certificate, document").
MARKERS = {
    "dense_tabular_rows": [
        "repeated rows", "repeating rows", "many rows", "dozens of rows",
        "manifest", "census", "passenger manifest", "fixed column",
        "structured data in", "tabular structure", "repeated fields",
        "repeated columns",
    ],
    "handwritten_ledger": [
        "handwritten", "handwriting", "handwritten ledger", "handwritten record",
        # bare "ledger" deliberately excluded - Gemma's own dense_tabular_rows
        # guidance routinely says "characteristic of a manifest or ledger" as
        # a paired illustrative phrase, not a genuine competing signal
        # (confirmed empirically: this was the single largest false-Ambiguous
        # driver after "tabular structure" before this fix).
    ],
    "printed_document": [
        "typed prose", "printed prose", "printed document", "typed document",
        "printed record", "single-entity", "letter", "certificate",
        "typed page", "printed page", "book chapter", "printed text",
    ],
    "portrait_photo": [
        "portrait", "photograph of a person", "photograph depicting a person",
        "photograph of people", "photograph depicting people",
    ],
    "map_land_record": [
        "map", "survey", "geographic", "satellite", "aerial view",
        "land survey", "township", "plat", "land record",
    ],
    "mixed_text_image": [
        "combination of text and", "text and a photograph", "text and an illustration",
        "photograph and text",
    ],
    "genealogy_chart": [
        "family tree", "pedigree", "lineage diagram", "genealogy chart",
    ],
    "website_screenshot": [
        "website", "webpage", "web page", "screenshot of a", "browser",
        "user interface", "hyperlink", "search results", "catalog",
        "database record",
    ],
    "photo_collage": [
        "collage", "album page", "multiple photographs", "photo gallery",
        "gallery of photos",
    ],
    "casual_photo": [
        "food", "photograph of an object", "candid photograph", "tray of",
    ],
    "cemetery_photo": [
        "headstone", "gravestone", "grave marker", "cemetery", "tombstone",
    ],
}

SELF_CONTRADICTION_PATTERNS = [
    (r"\bdoes not\b.{0,60}\brepeat", r"\brepeat"),
    (r"\bnot a\b.{0,40}\b(document|photo|map|chart)", r"\bis a\b.{0,40}\b(document|photo|map|chart)"),
]


# Negation cues checked in the ~4 words immediately preceding a marker
# phrase - Gemma's own guidance text frequently uses NEGATED comparisons
# to justify a DIFFERENT category ("without a tabular structure",
# "does not repeat the same row structure" is literally in printed_
# document's own classifier_guidance) - a marker match inside a negated
# span is not a genuine cross-category signal and must not count as one.
# Found empirically: "tabular structure" alone drove 429/496 of the
# initial (pre-fix) Ambiguous flags, almost entirely via this pattern.
_NEGATION_CUES = ["not", "without", "isn't", "doesn't", "does not", "no ", "n't"]
_NEGATION_WINDOW_CHARS = 30
# "rather than X (or Y)" - a comparative negation whose scope can extend
# well past a short fixed window (e.g. "...typical of an archival
# record rather than a repeating tabular structure OR a website
# interface" - "website interface" is ~50 chars past "rather than").
# Found empirically reading all 35 initial Contradictory hits directly:
# most of the printed_document false positives were this exact pattern,
# missed by the fixed-window check above. If "rather than"/"instead of"
# appears ANYWHERE before the marker in the same sentence, treat it as
# negated - a whole-sentence field (one sentence, 40-word cap per
# core/schema.py) makes this safe; it won't bleed into an unrelated
# earlier clause the way it might in a longer, multi-sentence text.
_COMPARATIVE_NEGATION_CUES = ["rather than", "instead of", "not a ", "not the "]


def _find_markers(text: str, category: str) -> list[str]:
    text_lower = text.lower()
    hits = []
    for marker in MARKERS.get(category, []):
        idx = text_lower.find(marker)
        if idx == -1:
            continue
        window = text_lower[max(0, idx - _NEGATION_WINDOW_CHARS):idx]
        if any(cue in window for cue in _NEGATION_CUES):
            continue  # negated mention - not a genuine signal for this category
        preceding_text = text_lower[:idx]
        if any(cue in preceding_text for cue in _COMPARATIVE_NEGATION_CUES):
            continue  # "rather than X" / "instead of X" - X is being ruled OUT
        hits.append(marker)
    return hits


# mixed_text_image is DEFINED as text + a photo/illustration together
# (config/taxonomy.yaml's own classifier_guidance: "a combination of
# substantial text AND a photograph/illustration"). A reason describing
# THAT photo component using portrait_photo/printed_document language
# (e.g. "a portrait photograph combined with substantial text") is
# correctly describing one half of a legitimately mixed image, not
# contradicting the category - found empirically: this pattern was the
# single largest remaining false-positive source after the negation
# fixes above, reading all 35 initial hits directly (10/10 of the
# mixed_text_image cases were this, zero were genuine contradictions).
_MIXED_TEXT_IMAGE_EXPECTED_COMPONENTS = {"portrait_photo", "printed_document"}


def _classify_reason(bucket: str, reason: str) -> tuple[str, dict]:
    text_lower = reason.lower()
    hits_by_category = {}
    for category in ALL_BUCKETS:
        hits = _find_markers(reason, category)
        if hits:
            hits_by_category[category] = hits

    if bucket == "mixed_text_image":
        for expected in _MIXED_TEXT_IMAGE_EXPECTED_COMPONENTS:
            hits_by_category.pop(expected, None)

    own_hits = hits_by_category.get(bucket, [])
    other_hits = {c: h for c, h in hits_by_category.items() if c != bucket}

    if len(hits_by_category) >= 2:
        tier = "Ambiguous" if own_hits else "Contradictory"
    elif other_hits and not own_hits:
        tier = "Contradictory"
    elif own_hits:
        tier = "Consistent"
    else:
        tier = "Weak"

    return tier, {"own_hits": own_hits, "other_hits": other_hits}


def _check_self_contradiction(reason: str) -> bool:
    text_lower = reason.lower()
    for neg_pattern, pos_pattern in SELF_CONTRADICTION_PATTERNS:
        neg_match = re.search(neg_pattern, text_lower)
        if neg_match and re.search(pos_pattern, text_lower[neg_match.end():]):
            return True
    return False


def _load_bucket_rows() -> list[dict]:
    rows = []
    for bucket in ALL_BUCKETS:
        path = BUCKET_DIR / f"{bucket}.csv"
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("file_path"):
                    row["bucket"] = bucket
                    rows.append(row)
    return rows


def _load_processing_profiles() -> dict[str, str]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT working_path, processing_profile FROM images")
    return {row["working_path"]: row["processing_profile"] for row in cur.fetchall()}


def _load_tower_disagreements() -> set[str]:
    if not DISAGREEMENTS_CSV.exists():
        return set()
    with open(DISAGREEMENTS_CSV, newline="", encoding="utf-8") as f:
        return {row["working_path"] for row in csv.DictReader(f)}


def _load_human_ground_truth() -> dict[str, str]:
    if not MISCLASSIFICATIONS_CSV.exists():
        return {}
    with open(MISCLASSIFICATIONS_CSV, newline="", encoding="utf-8") as f:
        return {row["file_path"]: row.get("correct_category", "") for row in csv.DictReader(f)}


def main() -> None:
    rows = _load_bucket_rows()
    print(f"{len(rows)} classified images loaded.")

    profiles = _load_processing_profiles()
    disagreeing = _load_tower_disagreements()
    human_gt = _load_human_ground_truth()

    tier_counts = Counter()
    self_contra_count = 0
    enriched = []

    for row in rows:
        bucket = row["bucket"]
        reason = row.get("reason", "") or ""
        tier, hit_detail = _classify_reason(bucket, reason)
        is_self_contra = _check_self_contradiction(reason)
        if is_self_contra:
            tier = "Self-Contradictory"
            self_contra_count += 1
        tier_counts[tier] += 1

        fp = row["file_path"]
        enriched.append({
            "file_path": fp,
            "bucket": bucket,
            "confidence": row.get("confidence", ""),
            "reason": reason,
            "tier": tier,
            "own_hits": ";".join(hit_detail["own_hits"]),
            "other_hits": ";".join(f"{c}:{','.join(h)}" for c, h in hit_detail["other_hits"].items()),
            "processing_profile": profiles.get(fp, ""),
            "tower_disagrees": fp in disagreeing,
            "human_ground_truth": human_gt.get(fp, ""),
        })

    print("\n=== [A] Counts by consistency tier ===")
    for tier in ["Consistent", "Weak", "Contradictory", "Self-Contradictory", "Ambiguous"]:
        n = tier_counts.get(tier, 0)
        print(f"  {tier:<20} {n:>5} ({100*n/len(rows):.1f}%)")

    print("\n=== [C] Contradictory cases by bucket ===")
    contra_by_bucket = Counter(r["bucket"] for r in enriched if r["tier"] == "Contradictory")
    bucket_totals = Counter(r["bucket"] for r in enriched)
    for bucket, n in contra_by_bucket.most_common():
        total = bucket_totals[bucket]
        print(f"  {bucket:<20} {n}/{total} ({100*n/total:.1f}%)")

    print("\n=== Contradictory cases: confidence distribution ===")
    contra_confidences = [float(r["confidence"]) for r in enriched if r["tier"] == "Contradictory" and r["confidence"]]
    if contra_confidences:
        import statistics
        print(f"  n={len(contra_confidences)} mean={statistics.mean(contra_confidences):.3f} "
              f"median={statistics.median(contra_confidences):.3f} "
              f"min={min(contra_confidences):.3f} max={max(contra_confidences):.3f}")
    all_confidences = [float(r["confidence"]) for r in enriched if r["confidence"]]
    import statistics
    print(f"  (all images: mean={statistics.mean(all_confidences):.3f})")

    print("\n=== Contradictory cases: tower agreement ===")
    contra_rows = [r for r in enriched if r["tier"] == "Contradictory"]
    contra_tower_disagree = sum(1 for r in contra_rows if r["tower_disagrees"])
    print(f"  {contra_tower_disagree}/{len(contra_rows)} contradictory cases ALSO have tower disagreement")
    overall_disagree_rate = len(disagreeing) / len(rows)
    contra_disagree_rate = contra_tower_disagree / len(contra_rows) if contra_rows else 0
    print(f"  baseline tower-disagreement rate (all images): {100*overall_disagree_rate:.1f}%")
    print(f"  tower-disagreement rate WITHIN contradictory cases: {100*contra_disagree_rate:.1f}%")

    print("\n=== Contradictory cases: human ground truth overlap ===")
    contra_with_gt = [r for r in contra_rows if r["human_ground_truth"]]
    print(f"  {len(contra_with_gt)}/{len(contra_rows)} contradictory cases have a human label")
    for r in contra_with_gt:
        match = "MATCHES predicted bucket" if r["human_ground_truth"] == r["bucket"] else "DIFFERS from predicted bucket"
        print(f"    {Path(r['file_path']).name}: predicted={r['bucket']} human={r['human_ground_truth']} ({match})")

    print(f"\nSelf-contradictory (mechanical negation-pattern check): {self_contra_count}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "full_audit.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(enriched[0].keys()))
        writer.writeheader()
        writer.writerows(enriched)
    print(f"\nFull per-image audit written to {out_path}")

    # write out the Contradictory set specifically for manual review
    contra_path = OUTPUT_DIR / "contradictory_cases.csv"
    with open(contra_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(enriched[0].keys()))
        writer.writeheader()
        writer.writerows(contra_rows)
    print(f"Contradictory-tier cases written to {contra_path}")


if __name__ == "__main__":
    main()
