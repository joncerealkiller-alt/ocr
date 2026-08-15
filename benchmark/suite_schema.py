"""
Benchmark-suite definition schema for the Model Console Benchmark tab
(2026-08-15).

Suites are YAML files under config/benchmark_suites/ - same
"config owns the data, code owns the mechanics" convention as
config/models/*.yaml + core/loaders/base_loader.py's GenerationConfig
(see docs/CODE_MAP.md). A suite defines WHAT to run (ordered test
cases, each with a prompt/optional image/expected answer/scorer) and
WHICH settings to run it under (generation_settings, applied as
model_console/adapter.py's build_config() overrides - the exact same
mechanism chat_tab.py already uses for per-session temperature/top_p
overrides, not a second settings system).

No Python behavior lives in a suite YAML - only data. Scoring LOGIC
lives in benchmark/scorers.py; a suite only names which registered
scorer to use per case, per this project's existing "config → registry
→ code" pattern (core/loader_registry.py's LOADER_REGISTRY is the
precedent this mirrors).

Versioning (task requirement #12): `version` is a required field.
Changing a suite's prompts, expected answers, image assets, scorer
choices, or generation_settings MUST bump `version` - the suite_id used
for eligibility/eligibility-persistence purposes is `f"{suite_id}_v{version}"`
(see BenchmarkSuite.qualified_id), so a run recorded against
"vision_baseline_v1" can never be silently compared to a
"vision_baseline_v2" run - benchmark_tab.py's compare view checks this
explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUITES_DIR = PROJECT_ROOT / "config" / "benchmark_suites"


@dataclass
class BenchmarkCase:
    case_id: str
    category: str          # free-text grouping, e.g. "instruction_following"
    prompt: str
    # Relative to PROJECT_ROOT (e.g. "data/_test_fixtures/fake_test_doc.png").
    # None for text-only cases.
    image_path: Optional[str] = None
    scorer: str = "manual"          # a key in benchmark.scorers.SCORER_REGISTRY
    scorer_args: dict[str, Any] = field(default_factory=dict)
    # Free-text, shown in the UI next to the raw output - NOT fed to the
    # scorer (scorer_args carries whatever structured form the scorer
    # itself needs, which may differ from how it reads on screen).
    expected_display: str = ""

    def resolve_image_path(self) -> Optional[Path]:
        return (PROJECT_ROOT / self.image_path) if self.image_path else None


@dataclass
class BenchmarkSuite:
    suite_id: str
    version: str
    display_name: str
    description: str
    # "text" | "vision" - determines eligibility (see model_console/
    # benchmark_tab.py's eligible_models_for_suite()): "vision" requires
    # image_input_supported=True; "text" requires text_only_supported=True
    # (every case in a text suite is sent with no image attached).
    required_capability: str
    # Applied as build_config()'s `overrides` dict - i.e. these fields
    # are set directly on the GenerationConfig instance before load, the
    # same mechanism model_console's per-session override panel already
    # uses. Deliberately controls determinism: every suite here sets
    # do_sample: False (greedy decoding - no seed needed, see the
    # generation_settings default below and docs/RUN_ARCHITECTURE.md's
    # "controlled conditions" language in the task this suite schema was
    # built for).
    generation_settings: dict[str, Any]
    cases: list[BenchmarkCase]

    @property
    def qualified_id(self) -> str:
        """The comparison-safety key (task #12): two runs are only
        ever compared as "the same benchmark" if this string matches
        exactly - a version bump on either side of a suite YAML edit
        changes this and correctly breaks that equality."""
        return f"{self.suite_id}_v{self.version}"


def list_suite_files() -> list[Path]:
    if not SUITES_DIR.exists():
        return []
    return sorted(SUITES_DIR.glob("*.yaml"))


def load_suite(path_or_stem: str | Path) -> BenchmarkSuite:
    """Accepts either a bare filename stem (e.g. "text_baseline_v1",
    matching config/benchmark_suites/text_baseline_v1.yaml) or a full
    Path. Raises FileNotFoundError / ValueError with a specific message
    on a missing file or malformed schema, rather than letting a KeyError
    from a missing required field surface as a confusing traceback deep
    in the UI thread."""
    path = Path(path_or_stem)
    if not path.suffix:
        path = SUITES_DIR / f"{path_or_stem}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No benchmark suite file at {path}.")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    required_top = ["suite_id", "version", "display_name", "description",
                     "required_capability", "generation_settings", "cases"]
    missing = [k for k in required_top if k not in data]
    if missing:
        raise ValueError(f"{path}: missing required suite field(s): {missing!r}")
    if data["required_capability"] not in ("text", "vision"):
        raise ValueError(
            f"{path}: required_capability must be 'text' or 'vision', got "
            f"{data['required_capability']!r}"
        )

    cases = []
    for i, raw_case in enumerate(data["cases"]):
        for k in ("case_id", "category", "prompt"):
            if k not in raw_case:
                raise ValueError(f"{path}: cases[{i}] missing required field {k!r}")
        cases.append(BenchmarkCase(
            case_id=raw_case["case_id"],
            category=raw_case["category"],
            prompt=raw_case["prompt"],
            image_path=raw_case.get("image_path"),
            scorer=raw_case.get("scorer", "manual"),
            scorer_args=raw_case.get("scorer_args") or {},
            expected_display=raw_case.get("expected_display", ""),
        ))
    if not cases:
        raise ValueError(f"{path}: suite has zero cases.")

    return BenchmarkSuite(
        suite_id=data["suite_id"],
        version=str(data["version"]),
        display_name=data["display_name"],
        description=data["description"],
        required_capability=data["required_capability"],
        generation_settings=dict(data["generation_settings"]),
        cases=cases,
    )


def list_suites() -> list[BenchmarkSuite]:
    """All suites under config/benchmark_suites/, individually. A
    malformed suite file is skipped with a printed warning rather than
    aborting the whole list - one bad suite file must not make every
    other suite unselectable in the UI (same tolerance model_console/
    chat_tab.py's load_console_prompts() already applies to
    config/console_prompts/*.yaml)."""
    suites = []
    for path in list_suite_files():
        try:
            suites.append(load_suite(path))
        except (FileNotFoundError, ValueError) as e:
            print(f"[benchmark.suite_schema] WARNING: skipping malformed suite {path}: {e}")
    return suites
