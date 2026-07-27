"""
Application entry point.

Currently a minimal stub (2026-07-25, repo-root reorganization): the
real end-user UI doesn't exist yet, so this just launches debug_tools/
workflow_gui.py - the current dev/debug workspace - as a subprocess,
same "call the real entry point, don't reimplement it" principle
workflow_gui.py itself already uses for every tool IT launches.

Once the real UI is built, this file becomes the actual application
entry point instead of a launcher for workflow_gui.py - `python
main.py` stays the one command users need to know regardless of what's
actually running underneath, so the README's documented startup
command never has to change again.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    workflow_gui = PROJECT_ROOT / "debug_tools" / "workflow_gui.py"
    result = subprocess.run([sys.executable, str(workflow_gui)], cwd=str(PROJECT_ROOT))
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
