"""
Initial tool set (Phase 1 of the agent architecture). Each tool wraps an
existing pipeline stage — it does not reimplement classification,
preprocessing, or extraction logic that already exists.

GPU-cost tools (extract_fields) go through the shared
core.model_residency.residency manager, NOT core/extractor.py's own
loader cache — see extract_fields_tool's docstring and
core/model_residency.py's module docstring for why: extractor.py's
cache is scoped to its own standalone-process batch pipeline, and
reusing it from this process would create a second, independently-owned
GPU-resident model alongside whatever model_console's chat session
already has loaded.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from pydantic import BaseModel, Field

from core.agent_tools.errors import ToolError, ToolErrorClass
from core.agent_tools.registry import register_tool
from core.agent_tools.schema import CostHint, SideEffect
from core.document_classification import classify_document as _classify_document
from core.image_preprocessing import apply_profile as _apply_profile


# ---------------------------------------------------------------------------
# classify_document — rule-based, cheap, no GPU (core/document_classification.py)
# ---------------------------------------------------------------------------

class ClassifyDocumentArgs(BaseModel):
    file_path: str


class ClassifyDocumentResult(BaseModel):
    doc_type: str
    confidence: float
    features: dict
    scores: dict


@register_tool(
    name="classify_document",
    description=(
        "Classify a census/ledger page image against known structural templates "
        "(aspect ratio, ruling-line/column count, row count). Rule-based, no GPU. "
        "Returns doc_type='unknown' honestly when no template matches well enough "
        "rather than forcing a guess."
    ),
    input_schema=ClassifyDocumentArgs,
    output_schema=ClassifyDocumentResult,
    side_effects=SideEffect.READ,
    cost_hint=CostHint.CHEAP,
)
def classify_document_tool(args: ClassifyDocumentArgs) -> ClassifyDocumentResult:
    path = Path(args.file_path)
    if not path.exists():
        raise ToolError(ToolErrorClass.NOT_FOUND, f"No such file: {path}")
    with Image.open(path) as img:
        result = _classify_document(img)
    return ClassifyDocumentResult(
        doc_type=result.doc_type,
        confidence=result.confidence,
        features=result.features,
        scores=result.scores,
    )


# ---------------------------------------------------------------------------
# preprocess_image — OpenCV/PIL geometry+contrast profile (core/image_preprocessing.py)
# ---------------------------------------------------------------------------

class PreprocessImageArgs(BaseModel):
    file_path: str
    profile_name: str = Field(description="Profile name from core/image_preprocessing.py's apply_profile")
    output_path: str


class PreprocessImageResult(BaseModel):
    output_path: str
    profile_name: str


@register_tool(
    name="preprocess_image",
    description=(
        "Apply a named OpenCV/PIL preprocessing profile (contrast, sharpen, "
        "denoise, etc.) to an image and write the result to output_path. "
        "Geometry and pixel-level correction only — never re-derives content."
    ),
    input_schema=PreprocessImageArgs,
    output_schema=PreprocessImageResult,
    side_effects=SideEffect.WRITE,
    cost_hint=CostHint.CHEAP,
)
def preprocess_image_tool(args: PreprocessImageArgs) -> PreprocessImageResult:
    src = Path(args.file_path)
    if not src.exists():
        raise ToolError(ToolErrorClass.NOT_FOUND, f"No such file: {src}")
    try:
        with Image.open(src) as img:
            processed = _apply_profile(img, args.profile_name)
            processed.save(args.output_path)
    except KeyError as e:
        raise ToolError(
            ToolErrorClass.INVALID_ARGS, f"Unknown preprocessing profile: {e}"
        ) from e
    return PreprocessImageResult(output_path=args.output_path, profile_name=args.profile_name)


# ---------------------------------------------------------------------------
# extract_fields — GPU stage, wraps core/extractor.py's loader + ExtractionResult
# ---------------------------------------------------------------------------

class ExtractFieldsArgs(BaseModel):
    file_path: str
    category: str = Field(description="core.schema.DocumentCategory value")


class ExtractFieldsResult(BaseModel):
    file_path: str
    category: str
    document_type: str | None = None
    personal_names: list[dict] = Field(default_factory=list)
    place_names: list[dict] = Field(default_factory=list)
    visible_dates: list[dict] = Field(default_factory=list)
    subject_keywords: list[str] = Field(default_factory=list)
    model: str
    prompt_version: str


@register_tool(
    name="extract_fields",
    description=(
        "Extract NEW information from the CURRENTLY ATTACHED image by actually "
        "running a vision model on it. Use this when the user wants to know "
        "what's IN this image right now. Do NOT use this to check whether "
        "something has been seen/recorded BEFORE, in a PAST document, or in an "
        "EARLIER session — that question is answered by "
        "genealogy_framework_lookup instead, which is instant and does not "
        "re-run extraction. Run the configured extraction model (per "
        "config/pipeline.yaml's bucket "
        "routing) against a single image and return a schema-validated "
        "ExtractionResult (names/places/dates, each with a mandatory confidence "
        "level). REQUIRES a 'category' argument - judge it yourself from what "
        "you can see in the image, choosing EXACTLY one of: dense_tabular_rows, "
        "handwritten_ledger, printed_document, portrait_photo, map_land_record, "
        "mixed_text_image, genealogy_chart, website_screenshot, photo_collage, "
        "casual_photo, cemetery_photo. Do NOT call classify_document for this - "
        "that tool's doc_type output (a census-template name like "
        "'canada_census_1921') is a DIFFERENT taxonomy and will NOT work here. "
        "Touches the GPU — the "
        "dispatcher serializes this against other GPU tool calls, AND "
        "(2026-08-13) this tool borrows the shared GPU residency slot for its "
        "duration, temporarily displacing whatever chat model was resident and "
        "restoring it afterward - see core/model_residency.py."
    ),
    input_schema=ExtractFieldsArgs,
    output_schema=ExtractFieldsResult,
    side_effects=SideEffect.READ,
    cost_hint=CostHint.GPU,
)
def extract_fields_tool(args: ExtractFieldsArgs) -> ExtractFieldsResult:
    # Deliberately does NOT call core.extractor.get_or_build_loader() -
    # that function owns extractor.py's OWN _loader_cache, meant only
    # for extractor.py's standalone-process batch pipeline (core/
    # extractor.py::run(), always its own OS process - see model_console/
    # app.py's docstring). Calling it from THIS process (agent mode,
    # running inside model_console) would create a second, independent
    # GPU-resident loader alongside whatever model_console's
    # ChatBackendAdapter already has resident - the exact two-owners bug
    # core/model_residency.py's module docstring documents in full.
    # Instead, this builds its own config the same way get_or_build_
    # loader() does internally (load_model_config + prompt_text swap)
    # and asks the ONE shared residency manager for a loader, which
    # transparently displaces and later restores whatever else was
    # resident.
    from core.extractor import load_pipeline_config
    from core.loaders.base_loader import load_model_config
    from core.model_residency import residency

    path = Path(args.file_path)
    if not path.exists():
        raise ToolError(ToolErrorClass.NOT_FOUND, f"No such file: {path}")

    pipeline_cfg = load_pipeline_config()
    bucket_cfg = pipeline_cfg["buckets"].get(args.category)
    if not bucket_cfg or not bucket_cfg.get("model"):
        raise ToolError(
            ToolErrorClass.INVALID_ARGS,
            f"Category {args.category!r} has no extraction model configured "
            f"in config/pipeline.yaml.",
        )

    model_name = bucket_cfg["model"]
    prompt_path = Path(__file__).resolve().parent.parent.parent / bucket_cfg["prompt_file"]
    if not prompt_path.exists():
        raise ToolError(
            ToolErrorClass.EXECUTION_FAILED,
            f"Prompt file not found: {prompt_path} - this bucket's extraction "
            f"prompt hasn't been written yet.",
        )

    try:
        model_cfg = load_model_config(model_name)
        model_cfg.prompt_text = prompt_path.read_text(encoding="utf-8")
        with residency.borrow(model_name, model_cfg) as loader:
            with Image.open(path) as raw_image:
                result = loader.extract(str(path), args.category, raw_image)
    except Exception as e:  # noqa: BLE001 - OOM/parse/etc all surfaced as ToolError
        raise ToolError(ToolErrorClass.EXECUTION_FAILED, str(e)) from e

    # Phase 5: every successful extraction is also recorded as durable
    # genealogy memory (core/genealogy_memory.py), not just returned to
    # this one turn - a name/place/date found once should be findable
    # again in a later session via genealogy_framework_lookup, without
    # re-running extraction. Best-effort: a memory-write failure must
    # never fail the extraction itself (the tool's actual job), so this
    # is caught and printed, not raised.
    try:
        from core.genealogy_memory import memory as genealogy_memory
        # runtime passed explicitly (2026-08-13 provenance audit): a
        # discovery is provenanced by CHECKPOINT + RUNTIME, not model
        # name alone - model_cfg.runtime is the execution backend this
        # extraction actually ran under.
        genealogy_memory.record_extraction_result(result, runtime=model_cfg.runtime)
    except Exception as e:  # noqa: BLE001
        print(f"[core.agent_tools.tools] WARNING: extract_fields succeeded but "
              f"failed to record discoveries to genealogy memory: {e}")

    return ExtractFieldsResult(
        file_path=result.file_path,
        category=result.category.value,
        document_type=result.document_type,
        personal_names=[n.model_dump() for n in result.personal_names],
        place_names=[p.model_dump() for p in result.place_names],
        visible_dates=[d.model_dump() for d in result.visible_dates],
        subject_keywords=result.subject_keywords,
        model=result.model,
        prompt_version=result.prompt_version,
    )


# ---------------------------------------------------------------------------
# genealogy_framework_lookup — Phase 5, core/genealogy_memory.py's discoveries
# ---------------------------------------------------------------------------

class GenealogyLookupArgs(BaseModel):
    query: str = Field(description="Substring to search for, e.g. a name or place")
    entity_type: str | None = Field(
        default=None,
        description="Optional filter: 'personal_name', 'place_name', or 'visible_date'",
    )


class GenealogyLookupResult(BaseModel):
    matches: list[dict] = Field(default_factory=list)
    match_count: int


@register_tool(
    name="genealogy_framework_lookup",
    description=(
        "THE tool for any question phrased as 'have we seen X before', 'has X "
        "come up before', 'is X already recorded', or similar — checking PAST "
        "records, not the current image. Instant, cheap, CPU-only, no GPU "
        "involved, and does NOT re-run extraction on anything. Searches "
        "previously recorded genealogy discoveries (names, places, dates found "
        "by earlier extract_fields calls, across ALL past sessions, not just "
        "this conversation) for a match. If the answer might already be in "
        "past records, try THIS tool BEFORE extract_fields — extract_fields "
        "re-analyzes the current image from scratch and is slower; only use it "
        "once you actually need NEW information the past records don't have. "
        "Arguments: args must be "
        "{\"query\": \"<the name or place to look for>\"} — the field is named "
        "'query', NOT 'substring' or 'search_term' or anything else."
    ),
    input_schema=GenealogyLookupArgs,
    output_schema=GenealogyLookupResult,
    side_effects=SideEffect.READ,
    cost_hint=CostHint.CHEAP,
)
def genealogy_framework_lookup_tool(args: GenealogyLookupArgs) -> GenealogyLookupResult:
    from core.genealogy_memory import memory as genealogy_memory

    if args.entity_type is not None and args.entity_type not in (
        "personal_name", "place_name", "visible_date",
    ):
        raise ToolError(
            ToolErrorClass.INVALID_ARGS,
            f"entity_type must be one of 'personal_name', 'place_name', "
            f"'visible_date', or omitted - got {args.entity_type!r}.",
        )

    matches = genealogy_memory.search(args.query, entity_type=args.entity_type)
    return GenealogyLookupResult(matches=matches, match_count=len(matches))


# ---------------------------------------------------------------------------
# web_search — Phase 7, the FIRST tool that leaves the local machine.
# Gated by core.network_gate.network_gate - OFF by default, must be
# explicitly enabled via the chat UI checkbox before this tool can do
# anything. See core/network_gate.py's module docstring for the full
# reasoning.
# ---------------------------------------------------------------------------

class WebSearchArgs(BaseModel):
    query: str = Field(description="The search query")


class WebSearchResultItem(BaseModel):
    title: str
    url: str
    snippet: str


class WebSearchResult(BaseModel):
    results: list[WebSearchResultItem] = Field(default_factory=list)
    result_count: int


@register_tool(
    name="web_search",
    description=(
        "Search the public web via DuckDuckGo and return the top results "
        "(title/url/snippet). This is the ONLY tool that leaves the local "
        "machine — it is gated behind an explicit 'Internet access' toggle "
        "in the chat UI, OFF by default. If internet access is currently "
        "disabled, this tool fails immediately with a clear message — do "
        "NOT retry it, tell the user to enable internet access instead. "
        "Use this only for genuinely external/current information (e.g. "
        "\"what year did X happen\", general historical context) — it "
        "cannot see the user's own documents or prior extractions, use "
        "genealogy_framework_lookup for that. Arguments: args must be "
        "{\"query\": \"<what to search for>\"} — the field is named "
        "'query' and is REQUIRED, do not call this tool with empty args."
    ),
    input_schema=WebSearchArgs,
    output_schema=WebSearchResult,
    side_effects=SideEffect.READ,
    # SLOW, not GPU - this is exactly the case CostHint.SLOW's own
    # docstring anticipated ("cheap on GPU but slow wall-clock, e.g.
    # network") back in Phase 1, before any network tool existed yet.
    cost_hint=CostHint.SLOW,
)
def web_search_tool(args: WebSearchArgs) -> WebSearchResult:
    from core.network_gate import network_gate

    if not network_gate.enabled:
        raise ToolError(
            ToolErrorClass.EXECUTION_FAILED,
            "Internet access is currently disabled. This is an explicit, "
            "user-controlled setting (OFF by default) - it will not become "
            "enabled by retrying. If the user wants a web search performed, "
            "tell them to enable 'Internet access' in the chat UI first.",
        )

    import requests
    from bs4 import BeautifulSoup

    try:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": args.query},
            headers={"User-Agent": "Mozilla/5.0 (genealogy_pipeline local agent)"},
            timeout=10,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise ToolError(ToolErrorClass.EXECUTION_FAILED, f"web_search request failed: {e}") from e

    # Parse from raw bytes, not resp.text - resp.text's charset
    # auto-detection was found unreliable against this endpoint's
    # response (BeautifulSoup's own encoding sniffing on resp.content
    # handles it correctly; confirmed live 2026-08-13, both actually
    # decode the same UTF-8 bytes correctly - resp.content is used here
    # simply to avoid depending on requests' encoding guess at all).
    soup = BeautifulSoup(resp.content, "html.parser")
    items: list[WebSearchResultItem] = []
    for result in soup.select(".result")[:5]:
        title_el = result.select_one(".result__title")
        snippet_el = result.select_one(".result__snippet")
        url_el = result.select_one(".result__url")
        if title_el is None:
            continue
        items.append(WebSearchResultItem(
            title=title_el.get_text(strip=True),
            url=url_el.get_text(strip=True) if url_el else "",
            snippet=snippet_el.get_text(strip=True) if snippet_el else "",
        ))

    return WebSearchResult(results=items, result_count=len(items))
