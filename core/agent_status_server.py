"""
Local-only HTTP + SSE transport for core/agent_status.py's status hub.
Built 2026-08-13 for the LLM Dashboard integration Jon requested - this
file has NO state of its own. Every response is either
`status_hub.snapshot()` verbatim or a pass-through of events pushed via
`status_hub.subscribe()`. It does not read loaders, the residency
manager, or anything else directly - see agent_status.py's own module
docstring for why (one authoritative owner, this is just a window onto
it).

Deliberately stdlib-only (http.server + json + threading) - no new
runtime dependency (Flask/FastAPI/etc) for something this small, and
one less thing to keep compatible with the rest of this project's
lazy-import discipline for heavy packages.

Bound to 127.0.0.1 ONLY, never 0.0.0.0 - this is a local dev/monitoring
tool for a single machine, not a service meant to be reachable over the
network. CORS is permissive (Access-Control-Allow-Origin: *) purely so
a dashboard on a different local port can fetch()/EventSource() this -
that's safe specifically because the bind address already restricts
who can reach it to processes on this same machine.

Endpoints:
    GET /api/status        - one JSON snapshot of AgentStatus
    GET /api/models        - configured models from config/models/*.yaml
    GET /api/current-run   - RunContext info (mostly null today - see
                              agent_status.py's run_context_hash field
                              docstring for why)
    GET /api/events        - Server-Sent Events stream; one `data: {...}`
                              line per status_hub.emit() call, plus a
                              ": keep-alive" comment every 15s so a
                              dashboard can detect a dead connection
                              instead of hanging silently forever.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty
from typing import Any

import yaml

from core.agent_status import status_hub

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

MODELS_DIR = Path(__file__).resolve().parent.parent / "config" / "models"

_KEEPALIVE_SECONDS = 15.0


def _list_configured_models() -> list[dict[str, Any]]:
    """
    Cheap, no GPU/model load - just reads config/models/*.yaml the same
    way model_console/chat_tab.py's list_model_profiles() does, but
    returns more fields (role, loader_class, repo_id) since this is for
    display, not a picker's dropdown values.
    """
    models = []
    for path in sorted(MODELS_DIR.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as e:  # noqa: BLE001 - one bad file shouldn't break the whole list
            models.append({"name": path.stem, "error": f"{type(e).__name__}: {e}"})
            continue
        models.append({
            "name": path.stem,
            "role": data.get("role"),
            "loader_class": data.get("loader_class"),
            "repo_id": data.get("repo_id"),
            "text_only_supported": data.get("text_only_supported", False),
        })
    return models


class _StatusHandler(BaseHTTPRequestHandler):
    # Quiet by default - the base class logs every request to stderr,
    # which would spam the console during normal dashboard polling/SSE
    # keepalives. Uncomment for debugging this server specifically.
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401
        pass

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server's naming convention
        if self.path == "/api/status":
            self._send_json(status_hub.snapshot())
        elif self.path == "/api/models":
            self._send_json({"models": _list_configured_models()})
        elif self.path == "/api/current-run":
            snap = status_hub.snapshot()
            self._send_json({
                "run_context_hash": snap.get("run_context_hash"),
                "note": (
                    "null today - agent-mode chat sessions don't construct a "
                    "core.run_context.RunContext yet (that's the offline batch "
                    "pipeline's concept). Wire this up if/when agent mode gets "
                    "a real RunContext of its own."
                ),
            })
        elif self.path == "/api/events":
            self._serve_sse()
        else:
            self._send_json({"error": f"unknown path {self.path!r}"}, status=404)

    def do_OPTIONS(self) -> None:  # noqa: N802 - CORS preflight
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()

    def _serve_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        # Send the current snapshot immediately as a synthetic first
        # event, so a dashboard that just connected doesn't have to wait
        # for the NEXT real state change to know what's happening right
        # now - matches /api/status's own content exactly.
        try:
            self._write_sse_event({"event": "snapshot", "status": status_hub.snapshot(), "ts": None})
        except (BrokenPipeError, ConnectionResetError):
            return

        q = status_hub.subscribe()
        try:
            while True:
                try:
                    payload = q.get(timeout=_KEEPALIVE_SECONDS)
                    self._write_sse_event(payload)
                except Empty:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # Dashboard closed the connection - normal, not an error.
            pass
        finally:
            status_hub.unsubscribe(q)

    def _write_sse_event(self, payload: dict) -> None:
        data = json.dumps(payload)
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()


def start_status_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """
    Starts the status server on a daemon thread and returns the server
    object (call .shutdown() to stop it, though in practice this runs
    for the lifetime of the model_console process and is never
    explicitly stopped - daemon=True means it doesn't block process
    exit either way).

    ThreadingHTTPServer (not the single-threaded HTTPServer) is
    required here specifically because /api/events holds its connection
    open indefinitely - a non-threading server would block every other
    endpoint (including /api/status) behind that one open SSE
    connection.
    """
    server = ThreadingHTTPServer((host, port), _StatusHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[core.agent_status_server] listening on http://{host}:{port} "
          f"(status/models/current-run/events)")
    return server
