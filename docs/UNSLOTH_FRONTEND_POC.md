# Unsloth Desktop as Knott's interactive frontend — reconnaissance + PoC results

**Dates:** recon 2026-08-15, PoC executed 2026-08-16 (Fable session).
**Verdict: VIABLE — Option B (agent integration via a Knott MCP server) validated
end-to-end, with a hard minimum-capability bar on the chat model.**

## What was verified (not inferred)

Unsloth Studio 0.1.800-beta = Tauri/WebView2 shell + local FastAPI backend
(port 8888, JWT via `/api/auth/desktop-login`; encrypted credential storage).
Its public OpenAPI spec (343 routes) confirmed:

- **MCP is first-class**: `/api/mcp/servers/` CRUD + `/test` probe + `/import`
  (Claude-style config). HTTP/streamable-HTTP transport only in the server
  registry (`McpServerCreate: {display_name, url, headers, is_enabled,
  use_oauth}`). Servers persist in `~/.unsloth/studio/studio.db` (`mcp_servers`
  table, created on first insert).
- **Per-request agentic controls** on `ChatCompletionRequest`:
  `mcp_enabled` (appends tools from every ENABLED server), `confirm_tool_calls`,
  `permission_mode` / `bypass_permissions`, `max_tool_calls_per_message`,
  `tool_call_timeout`, `run_tools_locally` (session-scoped sandboxes), plus
  standard OpenAI `tools`/`tool_choice`.
- **"Agents" settings = coding-agent bridging** (`unsloth start claude|codex
  --model ... [--as-subagent]`), NOT user-defined in-app agents. The in-app
  "agent" unit is a chat project (`instructions` + `root_path`) + per-chat MCP
  server selection + permission mode.
- **MCP management UI lives in the CHAT COMPOSER** (MCP dropdown → per-chat
  server tick-list + "Manage MCP servers"), not in the Settings sidebar.
  Permission mode is the composer's "Approve for me" dropdown; approval
  prompts render per tool call with Allow / Always allow / Deny.
- **Providers**: custom OpenAI-compatible base_url registerable; per-request
  provider override. Registry flags supports_vision/tool_calling per provider.
- **RAG**: knowledge bases + per-project/thread docs + `scan_folders`
  (directory registration - no corpus duplication), retrieval exposed to the
  model as a `search_knowledge_base` tool; research pipeline records
  filename/page/score/snippet per source. NOT used for Knott (our provenance
  discipline stays in our own tools).
- **Security defaults observed**: keyless localhost inference on, but "Allow
  tools" for keyless callers OFF; remote/LAN exposure opt-in behind a password
  (Cloudflare tunnel); risky code/terminal calls "still ask" per their own UI
  copy.

## The PoC (committed as api/knott_mcp.py, 6b83ba0)

Read-only MCP server (official `mcp` SDK, already installed; streamable-HTTP on
`127.0.0.1:8977/mcp`; reads confined to the workspace) exposing three tools
over EXISTING artifacts - zero pipeline changes:

1. `search_extraction_runs` — experiments dir + benchmark-store index
2. `get_extraction_disagreements` — a two-stage JSON's field_agreement=false
   fields with both readers' values
3. `query_ground_truth` — latest label per cell from ground_truth_log.jsonl

Verified: full MCP conformance (initialize/tools-list/call) with real data;
path-confinement rejection; Unsloth's own probe (`ok=true, tool_count=3`);
registered as "Knott Genealogy (read-only PoC)" and surfaced in the chat
composer's MCP dropdown; approval prompts fired per call.

## The decisive result: chat-model capability is the whole ballgame

Same question ("which fields disagreed in the latest run, and what did each
reader say? check GT for row 5 of the 1931 page"), same tools, same flow:

- **Qwen3-VL-4B (local): FABRICATED everything.** Called all three tools
  correctly, then answered with invented names ("John Doe"/"John Smith"),
  invented dates, invented confidence percentages, and an invented GT label -
  none of it present in the tool outputs. Confident formatting, zero grounding.
  (The familiar failure mode, one level up from hint-parroting.)
- **GPT-5.6 (frontier, via ChatGPT connection): VERBATIM FAITHFUL.** Explored
  all three experiment legs (5 tool calls), reproduced all 15 disagreed fields
  exactly (Hawkins fame/Hawkins James, 402/Son, Gadon/Gordon, Huzie
  Bertha/Henzie Bertha, 444/44, ...), attributed readers correctly from the
  model string, and quoted row-5 GT exactly (Hyzie Bertha / 18 / Domestic /
  Manitoba / F). Zero fabrication.

## Production guidance (when this graduates from PoC)

- Chat layer: 4B-class is disqualified by direct evidence; frontier-class is
  validated. Where the fidelity floor sits between them is NOT established —
  this was one A/B, not a sweep. Any intermediate candidate (7B-30B local)
  gets its own faithfulness test on real tool outputs before being trusted.
- Project instructions must include: "Answer only from tool results; quote
  values verbatim; if a tool result doesn't contain the answer, say so."
- Permission split: "Always allow" acceptable for read-only tools once
  trusted; any future mutating tools (e.g. submit review decision) stay on
  per-call approval - or are simply not exposed.
- Do NOT register the WSL vLLM server as an Unsloth provider (dual GPU-owner
  conflict with core/gpu_coordinator.py). Unsloth uses its own models or
  frontier APIs; it touches Knott only through tools.
- Do NOT feed the corpus into Unsloth's RAG; expose our retrieval as MCP
  tools returning citations in our format.

## Known tool refinement (open)

`get_extraction_disagreements` reports an empty `field_agreement` (runs
predating the feature) the same as "all agreed" - distinguish "no agreement
data recorded" from "zero disagreements" in the tool output.

## Reversibility

Delete the "Knott Genealogy (read-only PoC)" entry via the chat composer's
Manage MCP servers, and `git rm api/knott_mcp.py`. Nothing else references
either.
