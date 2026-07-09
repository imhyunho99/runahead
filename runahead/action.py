"""The unit of everything.

An action is one prompt, run in one isolated worktree, producing one patch.
Not a branch. Storage, acceptance, and learning are all keyed on actions, which
is why a rebase conflict can fail one of them without touching the rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Action:
    kind: str
    prompt: str
    variant: int = 0
    depth: int = 0
    parent: str | None = None

    @property
    def id(self) -> str:
        return self.kind if self.variant == 0 else f"{self.kind}#{self.variant}"


@dataclass
class Result:
    action: Action
    patch: str | None = None
    tokens: int = 0
    ok: bool = True
    log: str = ""

    @property
    def empty(self) -> bool:
        return not (self.patch or "").strip()

    def files(self) -> list[str]:
        """Paths touched by the patch, read from its diff headers."""
        if not self.patch:
            return []
        out: list[str] = []
        for line in self.patch.splitlines():
            if line.startswith("+++ b/"):
                out.append(line[6:].strip())
        return out


@dataclass
class Candidate:
    """A proposed action, before it runs."""

    action: Action
    score: float = 0.0
    mean: float = 0.0
    observations: float = 0.0
    auto_accept: bool = False
    variants: int = 1
    may_expand_children: bool = False
    children: list["Candidate"] = field(default_factory=list)
