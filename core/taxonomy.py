"""
Single loader for config/taxonomy.yaml - the project's document
taxonomy, treated as data rather than hardcoded UI/pipeline constants.
Built 2026-08-04 after debug_tools/review_uncertain.py's ASSIGNABLE_
BUCKETS list was found stale relative to core/schema.py's DocumentCategory
enum (missing WEBSITE_SCREENSHOT, added 2026-08-01 but never propagated
to that UI) - a category/UI mismatch this module exists to make
structurally impossible going forward: any tool wanting the list of
assignable categories reads it from here, not from its own local list.

Two distinct concepts, kept apart per Jon's own framing: "Taxonomy ->
Categories (current buckets) + Subtypes (future Gemma pass)". Categories
are what DocumentCategory already validates in production (Gemma's
actual output, bucket CSVs, pipeline_db.py's images.bucket column) -
this module does NOT replace that enum or relax its validation; it adds
display/UI metadata on top of the SAME set of values, and layers the
not-yet-production subtype concept underneath each category.

CONSISTENCY, not duplication: load_taxonomy() raises if config/
taxonomy.yaml's category ids and core/schema.py's DocumentCategory
values ever drift apart (missing either direction) - the exact class of
bug this module exists to prevent from recurring silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from core.schema import DocumentCategory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TAXONOMY_PATH = PROJECT_ROOT / "config" / "taxonomy.yaml"

# The one DocumentCategory value that's a routing sentinel, not a real
# assignable destination - config/taxonomy.yaml's own entry for it is
# enabled: false, but it's also excluded here from the "every
# DocumentCategory must appear" consistency check's assignability
# assumption (it MUST appear in the file, just never as a button).
ROUTING_SENTINEL_ID = DocumentCategory.UNCERTAIN.value


@dataclass
class Subtype:
    id: str
    display_name: str
    sort_order: int = 0
    description: str = ""
    # Not populated by config/taxonomy.yaml today (no subtype defines a
    # color yet) - present for structural parity with Category so
    # core/review_modes.py's Option conversion works identically for
    # both, without the UI needing to special-case subtypes.
    color: str | None = None
    # Classifier-facing disambiguation text (2026-08-04) - see Category's
    # own field of the same name for the full reasoning. None until the
    # planned second Gemma pass (subtype classification) exists and
    # someone writes/tunes real guidance per subtype.
    classifier_guidance: str | None = None


@dataclass
class Category:
    id: str
    display_name: str
    enabled: bool = True
    sort_order: int = 0
    description: str = ""
    color: str | None = None
    keyboard_shortcut: str | None = None
    supports_subtypes: bool = False
    subtypes: list[Subtype] = field(default_factory=list)
    # Classifier-facing disambiguation text (2026-08-04) - DELIBERATELY
    # separate from `description` above. `description` is a short UI
    # tooltip string ("shown nowhere yet, reserved for a future tooltip
    # panel"). `classifier_guidance` is the real, hand-tuned prompt text
    # Gemma is shown to disambiguate this category from OTHERS (e.g.
    # dense_tabular_rows's guidance explicitly says "do NOT use this...
    # belongs in printed_document instead" - relationship-level nuance,
    # not a one-line summary). Conflating the two would either bloat UI
    # tooltips with classifier-tuning prose, or dilute the classifier's
    # accuracy by feeding it a generic description instead of the tuned
    # text. None means "not currently offered as a choice in the live
    # classifier prompt" - see categories_for_classifier_prompt().
    classifier_guidance: str | None = None

    def subtype(self, subtype_id: str) -> Subtype | None:
        for s in self.subtypes:
            if s.id == subtype_id:
                return s
        return None


@dataclass
class Taxonomy:
    categories: list[Category]

    def get_category(self, category_id: str) -> Category | None:
        for c in self.categories:
            if c.id == category_id:
                return c
        return None

    def assignable_categories(self) -> list[Category]:
        """Every enabled category, sorted for display - the list every
        review-mode UI should build its primary-bucket buttons from
        instead of a local hardcoded list."""
        return sorted((c for c in self.categories if c.enabled), key=lambda c: c.sort_order)

    def subtypes_for(self, category_id: str) -> list[Subtype]:
        """Sorted subtype list for one category - empty if that category
        doesn't support subtypes or has none defined. This is what Mode
        3 (Subtype Annotation) reads to generate ITS buttons, the same
        way assignable_categories() drives Mode 1/2's buttons."""
        category = self.get_category(category_id)
        if category is None or not category.supports_subtypes:
            return []
        return sorted(category.subtypes, key=lambda s: s.sort_order)

    def categories_for_classifier_prompt(self) -> list[Category]:
        """Every category with classifier_guidance defined, sorted -
        DELIBERATELY not the same filter as assignable_categories()
        (which excludes disabled categories like uncertain_review, a UI
        concern). uncertain_review DOES belong in the classifier's own
        choice set - it's genuinely one of the options Gemma can pick -
        so this filters on "has tuned guidance text" instead of
        "enabled." A newly-added category (photo_collage, casual_photo,
        cemetery_photo as of 2026-08-04) with no classifier_guidance yet
        is correctly excluded here - it exists for review/ground-truth
        purposes but isn't offered to the live classifier until someone
        writes and validates real guidance text for it."""
        return sorted(
            (c for c in self.categories if c.classifier_guidance),
            key=lambda c: c.sort_order,
        )

    def render_classifier_category_block(self) -> str:
        """Renders the "- id: guidance text" bullet list the classifier
        prompt's category-choice section is built from - the mechanical
        part of core/classifier.py's prompt that CAN be fully templated
        without losing any tuned disambiguation nuance, since it just
        concatenates each category's own classifier_guidance verbatim.
        Global, cross-category rules that aren't naturally one
        category's own text (e.g. the "FIRST, check for website_
        screenshot before anything else" pre-check) stay as fixed prose
        in the prompt template itself, not generated here."""
        lines = [
            f"- {c.id}: {c.classifier_guidance}"
            for c in self.categories_for_classifier_prompt()
        ]
        return "\n".join(lines)


def _parse_subtype(raw: dict) -> Subtype:
    return Subtype(
        id=raw["id"],
        display_name=raw.get("display_name", raw["id"]),
        sort_order=raw.get("sort_order", 0),
        description=raw.get("description", ""),
        color=raw.get("color"),
        classifier_guidance=raw.get("classifier_guidance"),
    )


def _parse_category(raw: dict) -> Category:
    return Category(
        id=raw["id"],
        display_name=raw.get("display_name", raw["id"]),
        enabled=raw.get("enabled", True),
        sort_order=raw.get("sort_order", 0),
        description=raw.get("description", ""),
        color=raw.get("color"),
        keyboard_shortcut=raw.get("keyboard_shortcut"),
        supports_subtypes=raw.get("supports_subtypes", False),
        subtypes=[_parse_subtype(s) for s in raw.get("subtypes", [])],
        classifier_guidance=raw.get("classifier_guidance"),
    )


def _check_consistency_with_document_category(taxonomy: "Taxonomy", path: Path) -> None:
    """
    The whole point of this module: catch a taxonomy.yaml/DocumentCategory
    drift at LOAD TIME, not silently (a stale UI missing a real bucket -
    the exact bug that motivated building this module - is a silent
    failure by design unless something actively checks for it).
    """
    taxonomy_ids = {c.id for c in taxonomy.categories}
    document_category_ids = {d.value for d in DocumentCategory}

    missing_from_taxonomy = document_category_ids - taxonomy_ids
    if missing_from_taxonomy:
        raise ValueError(
            f"{path} is missing categor{'y' if len(missing_from_taxonomy) == 1 else 'ies'} "
            f"present in core/schema.py's DocumentCategory: {sorted(missing_from_taxonomy)}. "
            f"Add {'it' if len(missing_from_taxonomy) == 1 else 'them'} to config/taxonomy.yaml."
        )

    extra_in_taxonomy = taxonomy_ids - document_category_ids
    if extra_in_taxonomy:
        raise ValueError(
            f"{path} defines categor{'y' if len(extra_in_taxonomy) == 1 else 'ies'} not present "
            f"in core/schema.py's DocumentCategory: {sorted(extra_in_taxonomy)}. Either add "
            f"{'it' if len(extra_in_taxonomy) == 1 else 'them'} to DocumentCategory (if this is a "
            f"real production category) or remove {'it' if len(extra_in_taxonomy) == 1 else 'them'} "
            f"from config/taxonomy.yaml."
        )


def load_taxonomy(path: Path = DEFAULT_TAXONOMY_PATH, validate: bool = True) -> Taxonomy:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    taxonomy = Taxonomy(categories=[_parse_category(c) for c in raw.get("categories", [])])
    if validate:
        _check_consistency_with_document_category(taxonomy, path)
    return taxonomy
