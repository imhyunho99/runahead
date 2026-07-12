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


TOKENS_PER_TIMEOUT_SECOND = 40
"""Rough charge for a timed-out call, so a hung branch still debits the budget."""


class Executor(Protocol):
    def run(self, action: Action, work: Path) -> Result: ...


@dataclass
class ClaudeExecutor:
    """Headless `claude -p`, confined to a worktree."""

    binary: str = "claude"
    timeout: float = 900.0

    def run(self, action: Action, work: Path) -> Result:
        assert_reversible(action.kind, action.prompt)

        try:
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
        except subprocess.TimeoutExpired:
            # A hung agent is a dead branch, not a dead run. Nobody is watching;
            # the other speculations must keep going. It is charged its full
            # timeout so the budget accounts for the wall clock it burned.
            return Result(
                action=action,
                ok=False,
                tokens=int(self.timeout * TOKENS_PER_TIMEOUT_SECOND),
                log=f"timed out after {self.timeout:.0f}s",
            )
        except (OSError, ValueError) as exc:
            return Result(action=action, ok=False, log=str(exc)[:2000])

        if proc.returncode != 0:
            return Result(action=action, ok=False, log=proc.stderr.strip()[:2000])

        tokens, cost = _usage_from(proc.stdout)
        patch = capture_patch(work)
        return Result(action=action, patch=patch, tokens=tokens, cost_usd=cost, ok=True)


def _usage_from(stdout: str) -> tuple[int, float]:
    """Cumulative (tokens, cost_usd) for one agent invocation.

    The trap this walked into: the top-level `usage` object is the LAST turn
    only, and reading just input+output from it under-reports a multi-turn agent
    by an order of magnitude -- a real task that reads a codebase spends most of
    its tokens on cache reads across many turns. The cumulative truth is in
    `modelUsage`, summed across models and all four token components.
    """
    try:
        payload = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return max(1, len(stdout) // 4), 0.0

    model_usage = payload.get("modelUsage") or {}
    if model_usage:
        tokens = 0
        cost = 0.0
        for stats in model_usage.values():
            tokens += (
                int(stats.get("inputTokens", 0))
                + int(stats.get("outputTokens", 0))
                + int(stats.get("cacheCreationInputTokens", 0))
                + int(stats.get("cacheReadInputTokens", 0))
            )
            cost += float(stats.get("costUSD", 0.0))
        cost = cost or float(payload.get("total_cost_usd", 0.0))
        if tokens:
            return tokens, cost

    # Fallback: older/other shapes without modelUsage. Sum every component of
    # the top-level usage rather than just input+output.
    usage = payload.get("usage") or {}
    tokens = (
        int(usage.get("input_tokens", 0))
        + int(usage.get("output_tokens", 0))
        + int(usage.get("cache_creation_input_tokens", 0))
        + int(usage.get("cache_read_input_tokens", 0))
    )
    cost = float(payload.get("total_cost_usd", 0.0))
    return (tokens or max(1, len(stdout) // 4)), cost


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
