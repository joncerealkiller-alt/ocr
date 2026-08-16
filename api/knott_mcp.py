"""
Knott genealogy MCP server - READ-ONLY proof of concept (2026-08-16).

Exposes three tools over MCP streamable-HTTP so an external MCP client
(the Unsloth Desktop PoC - see the 2026-08-15 reconnaissance report and
docs/TWO_STAGE_HINT_FREE_PROPOSAL.md's ecosystem) can query the Knott
extraction system's EXISTING data. Deliberately:

  - READ-ONLY: every tool only reads persisted artifacts (benchmark
    store, two-stage extraction JSONs, ground_truth_log.jsonl). No tool
    mutates pipeline state, runs models, or touches the GPU.
  - Zero pipeline changes: imports only stable read paths.
  - Reversible: delete this file + remove the MCP server entry in the
    client; nothing else references it.

Run:
    python api/knott_mcp.py          # serves http://127.0.0.1:8977/mcp

Port 8977 chosen to avoid this project's existing ports (8001 WSL
backend, 8502 vLLM, 8888 Unsloth backend).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP

from core.workspace_context import WorkspaceContext

GT_LOG = PROJECT_ROOT / "data" / "outputs" / "ground_truth_log.jsonl"
WORKSPACE = WorkspaceContext.resolve().workspace_root
EXPERIMENTS_DIR = WORKSPACE / "research" / "experiments"
BENCHMARK_RUNS_DIR = WORKSPACE / "research" / "model_console" / "benchmark_runs"

mcp_server = FastMCP(
    name="knott-genealogy",
    instructions=(
        "Read-only tools over the Knott genealogy extraction system: list "
        "extraction/benchmark runs, inspect two-stage extraction results and "
        "the fields where the two independent model readers disagreed, and "
        "look up human ground-truth labels. All values carry provenance "
        "(file paths, models, timestamps) - report them with your answers. "
        "These tools cannot run models or modify anything."
    ),
)


@mcp_server.tool()
def search_extraction_runs() -> str:
    """List available extraction experiment runs and benchmark runs, with
    their locations, models, and dates. Use this first to find a run name
    to pass to the other tools."""
    out: dict = {"extraction_experiments": [], "benchmark_runs": []}
    if EXPERIMENTS_DIR.is_dir():
        for exp in sorted(EXPERIMENTS_DIR.iterdir()):
            if not exp.is_dir():
                continue
            legs = []
            for leg in sorted(exp.iterdir()):
                if leg.is_dir():
                    jsons = list(leg.glob("*_twostage_extraction.json"))
                    if jsons:
                        legs.append({"leg": leg.name, "result_json": str(jsons[0])})
            out["extraction_experiments"].append({"experiment": exp.name, "path": str(exp), "legs": legs})
    index = BENCHMARK_RUNS_DIR / "index.json"
    if index.is_file():
        with open(index, "r", encoding="utf-8") as f:
            # newest-first index maintained by model_console/benchmark_store.py
            out["benchmark_runs"] = json.load(f)[:20]
        out["benchmark_runs_note"] = "showing newest 20; full store at " + str(BENCHMARK_RUNS_DIR)
    return json.dumps(out, indent=1)


@mcp_server.tool()
def get_extraction_disagreements(result_json_path: str) -> str:
    """Inspect a two-stage extraction result JSON (path from
    search_extraction_runs). Returns, per row, the fields where the two
    INDEPENDENT model readers disagreed (field_agreement=false), with
    both readings: stage 1's raw read and stage 2's recorded value, plus
    stage 2's confidence. Disagreed fields are the review-queue items;
    agreed fields are listed by name only."""
    path = Path(result_json_path)
    # Confine reads to the workspace/experiments area - this is a
    # read-only tool but should still not be a generic file reader.
    try:
        path.resolve().relative_to(WORKSPACE.resolve())
    except ValueError:
        return json.dumps({"error": f"path must be inside the workspace ({WORKSPACE})"})
    if not path.is_file():
        return json.dumps({"error": f"no file at {path}"})
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data["rows"] if isinstance(data, dict) and "rows" in data else data
    report = []
    for r in rows:
        s1 = {}
        for line in (r.get("stage1_raw_output") or "").split("\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                s1[k.strip()] = v.strip()
        fa = {k: (v if isinstance(v, bool) else str(v).lower() == "true")
              for k, v in (r.get("field_agreement") or {}).items()}
        disagreed = []
        for col, agree in fa.items():
            if agree:
                continue
            field = (r.get("fields") or {}).get(col) or {}
            disagreed.append({
                "column": col,
                "stage1_read": s1.get(col),
                "stage2_value": field.get("value"),
                "stage2_confidence": field.get("confidence"),
            })
        report.append({
            "row_index": r.get("row_index"),
            "models": r.get("model"),
            "disagreed_fields": disagreed,
            "agreed_fields": sorted(k for k, v in fa.items() if v),
        })
    return json.dumps({"source": str(path), "rows": report}, indent=1)


@mcp_server.tool()
def query_ground_truth(image_name_contains: str, row_index: int | None = None) -> str:
    """Look up human ground-truth labels from the labeling log. Filter by a
    substring of the source image name (e.g. '1931_174' or '1921_022')
    and optionally a specific row index. Returns the LATEST label per
    (row, column) - the log is append-only and later labels supersede
    earlier ones - including the labeler's status (readable/illegible/
    blank) and condition notes."""
    if not GT_LOG.is_file():
        return json.dumps({"error": f"no ground-truth log at {GT_LOG}"})
    latest: dict = {}
    with open(GT_LOG, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if image_name_contains.lower() not in rec.get("source_image_path", "").lower():
                continue
            if row_index is not None and rec.get("row_index") != row_index:
                continue
            latest[(rec.get("row_index"), rec.get("column"))] = rec
    results = [
        {"row_index": k[0], "column": k[1], "status": v.get("status"),
         "value": v.get("value"), "notes": v.get("notes"), "labeled_at": v.get("timestamp")}
        for k, v in sorted(latest.items(), key=lambda kv: (kv[0][0] or 0, kv[0][1] or ""))
    ]
    return json.dumps({"filter": image_name_contains, "row_index": row_index,
                        "labels": results, "count": len(results)}, indent=1)


if __name__ == "__main__":
    mcp_server.settings.host = "127.0.0.1"
    mcp_server.settings.port = 8977
    print(f"[knott_mcp] read-only MCP server on http://127.0.0.1:8977/mcp")
    mcp_server.run(transport="streamable-http")
