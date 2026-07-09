"""What might the human ask for next?

Cold start: an LLM reads the diff and proposes candidates. From there the Beta
counters take over the ranking, the variant count, and the promotion to
auto-accept. The LLM keeps proposing; the statistics decide what survives.

This is the (2) -> (3) trajectory. The predictor is never replaced, only
increasingly second-guessed by the record of what the human actually chose.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from typing import Protocol

TASK_KINDS = ("feature", "bugfix", "refactor", "chore")

SUCCESSORS: dict[str, tuple[str, ...]] = {
    "write-tests": ("run-tests",),
    "run-tests": ("draft-commit",),
    "lint-typecheck": ("draft-commit",),
    "add-edge-cases": ("write-tests",),
}

CATALOG: dict[str, str] = {
    "write-tests": "Write tests covering the change that was just made. Do not modify the implementation.",
    "run-tests": "Run the project's test suite. If a test fails because it is wrong, fix the test.",
    "lint-typecheck": "Run the project's linter and type checker and fix every issue they report.",
    "draft-commit": "Stage the change and write a commit message in this repo's existing style. Do not push.",
    "add-edge-cases": "Find inputs the change just made mishandles -- null, empty, boundary, concurrent -- and handle them.",
    "error-handling": "Add error handling to the change that was just made, following this repo's existing conventions.",
    "update-docs": "Update the README and any docstrings the change just invalidated.",
    "responsive-ui": "Make the UI touched by this change work on mobile viewports.",
    "perf-pass": "Find the most obvious performance problem introduced by this change and fix it.",
}


@dataclass(frozen=True)
class Proposal:
    kind: str
    prompt: str


class Predictor(Protocol):
    def classify(self, task: str) -> str: ...
    def propose(self, task: str, task_kind: str, diff: str) -> list[Proposal]: ...


def successors(kind: str) -> tuple[str, ...]:
    return SUCCESSORS.get(kind, ())


def prompt_for(kind: str, variant: int = 0) -> str:
    base = CATALOG.get(kind, f"Perform the following on the current change: {kind}.")
    if variant == 0:
        return base
    return f"{base}\n\nProduce approach #{variant}. It must differ substantively from the others."


@dataclass
class ClaudePredictor:
    binary: str = "claude"
    timeout: float = 120.0
    max_proposals: int = 6

    def classify(self, task: str) -> str:
        answer = self._ask(
            f"Classify this coding task as exactly one of {list(TASK_KINDS)}. "
            f"Reply with the single word only.\n\nTask: {task}"
        ).strip().lower()
        for kind in TASK_KINDS:
            if kind in answer:
                return kind
        return "feature"

    def propose(self, task: str, task_kind: str, diff: str) -> list[Proposal]:
        known = ", ".join(CATALOG)
        raw = self._ask(
            "A coding agent just finished a task. Predict what the human will ask for next.\n"
            f"Task ({task_kind}): {task}\n\n"
            f"Diff (truncated):\n{diff[:6000]}\n\n"
            f"Prefer these known action kinds where they fit: {known}\n"
            "You may invent a new kind if the diff calls for something specific.\n"
            "Never propose anything irreversible: no push, no PR, no deploy, no migration.\n"
            f"Reply with a JSON array of at most {self.max_proposals} objects, "
            'each {"kind": "kebab-case", "prompt": "imperative instruction"}. JSON only.'
        )
        return self._parse(raw)

    def _parse(self, raw: str) -> list[Proposal]:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            items = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

        out: list[Proposal] = []
        for item in items[: self.max_proposals]:
            kind = str(item.get("kind", "")).strip()
            if not kind:
                continue
            prompt = str(item.get("prompt") or "").strip() or prompt_for(kind)
            out.append(Proposal(kind=kind, prompt=prompt))
        return out

    def _ask(self, prompt: str) -> str:
        proc = subprocess.run(
            [self.binary, "-p", prompt, "--output-format", "text"],
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        return proc.stdout if proc.returncode == 0 else ""


@dataclass
class FakePredictor:
    kinds: tuple[str, ...] = ("write-tests", "lint-typecheck", "add-edge-cases")
    task_kind: str = "feature"

    def classify(self, task: str) -> str:
        return self.task_kind

    def propose(self, task: str, task_kind: str, diff: str) -> list[Proposal]:
        return [Proposal(kind=k, prompt=prompt_for(k)) for k in self.kinds]
