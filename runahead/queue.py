"""The review queue: the face of the product, and the only place the human pays.

Speculation only pays off when reviewing N results costs less than doing one.
So the human never composes a matrix -- they say yes or no per line, and rebase
does the composing. A chain is accepted whole; competing variants are a radio
group; everything else is independent.

Accepting is also how the system learns. This screen is the labeller.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .scheduler import Chain
from .store import Store
from .worktree import apply_patch, git, state_dir


@dataclass
class Step:
    id: str
    kind: str
    patch: str
    files: list[str] = field(default_factory=list)


@dataclass
class Entry:
    id: str
    group: str | None
    auto_accept: bool
    mean: float
    observations: float
    steps: list[Step]

    @property
    def kinds(self) -> list[str]:
        return [s.kind for s in self.steps]


@dataclass
class Queue:
    task: str
    task_kind: str
    entries: list[Entry] = field(default_factory=list)

    @classmethod
    def build(cls, task: str, task_kind: str, chains: list[Chain]) -> "Queue":
        entries = []
        for chain in chains:
            steps = [
                Step(id=r.action.id, kind=r.action.kind, patch=r.patch or "", files=r.files())
                for r in chain.results
            ]
            if not steps:
                continue
            entries.append(
                Entry(
                    id=chain.root.id,
                    group=chain.group,
                    auto_accept=chain.auto_accept,
                    mean=chain.mean,
                    observations=chain.observations,
                    steps=steps,
                )
            )
        return cls(task=task, task_kind=task_kind, entries=entries)

    # -- persistence -------------------------------------------------------

    @staticmethod
    def path(repo: Path) -> Path:
        return state_dir(repo) / "queue.json"

    def save(self, repo: Path) -> None:
        payload = {
            "task": self.task,
            "task_kind": self.task_kind,
            "entries": [
                {
                    "id": e.id,
                    "group": e.group,
                    "auto_accept": e.auto_accept,
                    "mean": e.mean,
                    "observations": e.observations,
                    "steps": [
                        {"id": s.id, "kind": s.kind, "patch": s.patch, "files": s.files}
                        for s in e.steps
                    ],
                }
                for e in self.entries
            ],
        }
        self.path(repo).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, repo: Path) -> "Queue | None":
        path = cls.path(repo)
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        entries = [
            Entry(
                id=e["id"],
                group=e["group"],
                auto_accept=e["auto_accept"],
                mean=e["mean"],
                observations=e["observations"],
                steps=[Step(**s) for s in e["steps"]],
            )
            for e in raw["entries"]
        ]
        return cls(task=raw["task"], task_kind=raw["task_kind"], entries=entries)

    def clear(self, repo: Path) -> None:
        self.path(repo).unlink(missing_ok=True)

    # -- rendering ---------------------------------------------------------

    def render(self, budget_summary: str = "") -> str:
        auto = [e for e in self.entries if e.auto_accept]
        predicted = [e for e in self.entries if not e.auto_accept]
        lines: list[str] = [f"task ({self.task_kind}): {self.task}", ""]

        if auto:
            lines.append("auto-accepted (graduated)")
            for e in auto:
                lines.append(f"  [x] {_label(e)}")
            lines.append("")

        if predicted:
            lines.append("needs review")
            groups: dict[str, list[Entry]] = {}
            for e in predicted:
                groups.setdefault(e.group or e.id, []).append(e)
            for members in groups.values():
                marker = "( )" if len(members) > 1 else "[ ]"
                for e in members:
                    lines.append(f"  {marker} {_label(e)}")
            lines.append("")

        lines.append("stopped at the reversibility boundary: push, pr, deploy")
        if budget_summary:
            lines.append(f"budget: {budget_summary}")
        return "\n".join(lines)


def _label(entry: Entry) -> str:
    chain = " -> ".join(entry.kinds)
    n = int(entry.observations)
    return f"{entry.id:<24} p={entry.mean:.2f} (n={n})  {chain}"


# -- acceptance ------------------------------------------------------------


@dataclass
class Outcome:
    applied: list[str] = field(default_factory=list)
    conflicted: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)


def accept(repo: Path, queue: Queue, store: Store, selected: set[str]) -> Outcome:
    """Apply the chosen chains in queue order, isolating any that will not compose."""
    if git(repo, "status", "--porcelain").strip():
        raise RuntimeError("working tree must be clean before accepting")

    _reject_conflicting_variants(queue, selected)

    outcome = Outcome()
    applied_files: dict[str, str] = {}

    for entry in queue.entries:
        chosen = entry.auto_accept or entry.id in selected
        if not chosen:
            outcome.rejected.append(entry.id)
            for kind in set(entry.kinds):
                store.record(queue.task_kind, kind, accepted=False)
            continue

        ok = _apply_entry(repo, entry)

        # The human wanted it either way, so the prediction was right. A merge
        # failure is a separate fact about two actions we wrongly believed
        # orthogonal -- and it gets its own label.
        for kind in set(entry.kinds):
            store.record(queue.task_kind, kind, accepted=True)

        if ok:
            outcome.applied.append(entry.id)
            for step in entry.steps:
                for f in step.files:
                    applied_files.setdefault(f, entry.id)
        else:
            outcome.conflicted.append(entry.id)
            for other in _overlapping(entry, applied_files):
                store.record_conflict(entry.id, other)

    store.append_history(
        {
            "event": "queue_resolved",
            "task_kind": queue.task_kind,
            "applied": outcome.applied,
            "conflicted": outcome.conflicted,
            "rejected": outcome.rejected,
        }
    )
    store.flush()
    queue.clear(repo)
    return outcome


def _apply_entry(repo: Path, entry: Entry) -> bool:
    """All steps or none. A partial chain is meaningless: step k+1 assumed step k."""
    head = git(repo, "rev-parse", "HEAD").strip()
    for step in entry.steps:
        ok, _ = apply_patch(repo, step.patch)
        if not ok:
            git(repo, "reset", "--hard", head, check=False)
            return False
    git(repo, "commit", "-q", "-m", f"runahead: {entry.id}", check=False)
    return True


def _overlapping(entry: Entry, applied_files: dict[str, str]) -> set[str]:
    touched = {f for step in entry.steps for f in step.files}
    return {applied_files[f] for f in touched if f in applied_files}


def _reject_conflicting_variants(queue: Queue, selected: set[str]) -> None:
    seen: dict[str, str] = {}
    for entry in queue.entries:
        if entry.group and entry.id in selected:
            if entry.group in seen:
                raise RuntimeError(
                    f"variants '{seen[entry.group]}' and '{entry.id}' are mutually exclusive"
                )
            seen[entry.group] = entry.id
