"""
PreToolUse hook (Bash + PowerShell): blocks recursive/force delete
commands (rm -rf, Remove-Item -Recurse, git clean -f, rmdir /s, etc.)
that target - directly, via a parent directory, or via a glob - any
path listed in .claude/protected_paths.txt.

Added 2026-07-26 after a real incident: a `rm -rf data/outputs/dewarped`
cleanup command, reused carelessly from earlier test cleanup, destroyed
25 real, manually-dewarped images with no way to recover them. This is
a backstop, not a substitute for reading a cleanup command's actual
target before running it - it only recognizes destructive-looking
syntax paired with a protected-path token, not every possible way to
remove a file.

Reads JSON on stdin (Claude Code's standard hook input), prints JSON on
stdout when blocking. Silence + exit 0 means "not our concern, defer to
normal permission handling" - this hook only ever actively blocks, it
never actively allows (that stays the normal permission system's job).
"""

from __future__ import annotations

import fnmatch
import json
import re
import shlex
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROTECTED_PATHS_FILE = Path(__file__).resolve().parent.parent / "protected_paths.txt"

CLAUSE_SPLIT = re.compile(r"[|;&]+")


def normalize(path_text: str) -> str:
    return path_text.strip().strip('"').strip("'").replace("\\", "/").rstrip("/").lower()


ALWAYS_DANGEROUS_TOKENS = {
    ".", "./",
    normalize(str(PROJECT_ROOT)),
}


def load_protected_paths() -> list[str]:
    if not PROTECTED_PATHS_FILE.exists():
        return []
    paths = []
    for line in PROTECTED_PATHS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        paths.append(normalize(line))
    return paths


def split_clauses(command: str) -> list[str]:
    return [c.strip() for c in CLAUSE_SPLIT.split(command) if c.strip()]


def tokenize(clause: str) -> list[str]:
    try:
        return shlex.split(clause, posix=False)
    except ValueError:
        return clause.split()


def clause_is_destructive(tokens: list[str]) -> bool:
    if not tokens:
        return False
    verb = tokens[0].replace("\\", "/").split("/")[-1].lower().strip('"').strip("'")
    rest = " ".join(tokens[1:]).lower()

    if verb == "rm":
        return bool(re.search(r"-[a-z]*r[a-z]*", rest)) or "--recursive" in rest
    if verb in ("remove-item", "ri"):
        return "-recurse" in rest
    if verb == "git" and len(tokens) > 1 and tokens[1].lower() == "clean":
        return bool(re.search(r"-[a-z]*f", " ".join(tokens[2:]).lower()))
    if verb in ("rmdir", "rd"):
        return "/s" in rest
    return False


def clause_targets_protected_path(tokens: list[str], protected_paths: list[str]) -> str | None:
    for raw_token in tokens:
        if not raw_token or raw_token.startswith("-"):
            continue
        token = normalize(raw_token)
        if not token:
            continue
        if token in ALWAYS_DANGEROUS_TOKENS or token == "*":
            return "(entire project root)"
        for protected in protected_paths:
            if token == protected:
                return protected
            if token.startswith(protected + "/"):
                return protected
            if protected.startswith(token + "/"):
                return protected
            if "*" in token and fnmatch.fnmatch(protected, token):
                return protected
    return None


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
    except json.JSONDecodeError:
        return

    command = (payload.get("tool_input") or {}).get("command", "")
    if not command:
        return

    protected_paths = load_protected_paths()
    if not protected_paths:
        return

    for clause in split_clauses(command):
        tokens = tokenize(clause)
        if not clause_is_destructive(tokens):
            continue
        hit = clause_targets_protected_path(tokens, protected_paths)
        if hit is None:
            continue

        reason = (
            f"Blocked: this looks like a recursive/force delete command targeting "
            f"a protected path ({hit}), listed in .claude/protected_paths.txt because "
            f"it holds real, hard-to-reproduce work product. This backstop exists after "
            f"a real incident where a reused cleanup command destroyed 25 manually-"
            f"dewarped images with no way to recover them. If this deletion is really "
            f"intended, do it manually (don't reuse a scripted command against it), or "
            f"remove/edit the entry in .claude/protected_paths.txt first and explain why."
        )
        print(json.dumps({
            "systemMessage": reason,
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            },
        }))
        return


if __name__ == "__main__":
    main()
