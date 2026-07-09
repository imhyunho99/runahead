"""The one seam between the core and any particular coding agent.

    run(action, worktree) -> Result

Nothing above this line knows what Claude is. The predictor, the Beta counters,
the scheduler, the reversibility boundary, the patch merge -- none of them are
Claude-specific, which is what lets a conductor or codex adapter be a new file
rather than a rewrite.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from .action import Action, Result
from .safety import assert_reversible
from .worktree import capture_patch


class Executor(Protocol):
    def run(self, action: Action, work: Path) -> Result: ...


@dataclass
class ClaudeExecutor:
    """Headless `claude -p`, confined to a worktree."""

    binary: str = "claude"
    timeout: float = 900.0

    def run(self, action: Action, work: Path) -> Result:
        assert_reversible(action.kind, action.prompt)

        proc = subprocess.run(
            [
                self.binary,
                "-p",
                action.prompt,
                "--output-format",
                "json",
                "--permission-mode",
                "acceptEdits",
            ],
            cwd=str(work),
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if proc.returncode != 0:
            return Result(action=action, ok=False, log=proc.stderr.strip()[:2000])

        tokens = _tokens_from(proc.stdout)
        patch = capture_patch(work)
        return Result(action=action, patch=patch, tokens=tokens, ok=True)


def _tokens_from(stdout: str) -> int:
    try:
        payload = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return max(1, len(stdout) // 4)
    usage = payload.get("usage") or {}
    total = int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
    return total or max(1, len(stdout) // 4)


@dataclass
class FakeExecutor:
    """Deterministic stand-in that really edits the worktree.

    The reason the core is testable without an LLM: swap this in and the
    scheduler, budget, boundary, and patch merge all run for real.
    """

    tokens: int = 1_000
    edit: Callable[[Action, Path], None] | None = None

    def run(self, action: Action, work: Path) -> Result:
        assert_reversible(action.kind, action.prompt)
        if self.edit:
            self.edit(action, work)
        else:
            (work / f"{action.kind}.txt").write_text(f"{action.id}\n", encoding="utf-8")
        return Result(action=action, patch=capture_patch(work), tokens=self.tokens, ok=True)
