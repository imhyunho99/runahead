"""The reversibility boundary.

runahead rests on one assumption: discarding a wrong guess is free. Inside a
worktree that is true. `git push` is not. Neither is a deploy, a migration, or
an outbound API call.

So the boundary of speculation is exactly the boundary of reversibility, and it
is drawn by the machine rather than by a human placing checkpoints by hand --
otherwise the whole asynchronous premise collapses.

Confidence never unlocks this. p = 0.99 still does not push.
"""

from __future__ import annotations

import re

IRREVERSIBLE_ACTION_KINDS = frozenset(
    {
        "push",
        "open-pr",
        "merge",
        "deploy",
        "publish",
        "migrate-db",
        "send-notification",
    }
)

IRREVERSIBLE_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bgit\s+push\b",
        r"\bgit\s+(?:merge|rebase)\s+.*\borigin\b",
        r"\bgh\s+(?:pr|release)\s+create\b",
        r"\bgh\s+pr\s+merge\b",
        r"\bnpm\s+publish\b",
        r"\bcargo\s+publish\b",
        r"\bdocker\s+push\b",
        r"\bterraform\s+apply\b",
        r"\bkubectl\s+(?:apply|delete)\b",
        r"\brm\s+-rf\s+/",
        r"\bDROP\s+(?:TABLE|DATABASE)\b",
        r"\b(?:alembic|prisma|rails)\s+.*migrat",
        r"\bcurl\b.*\b-X\s*(?:POST|PUT|DELETE)\b",
    )
)


class IrreversibleAction(Exception):
    """Raised when speculation would leave the sandbox."""


def is_reversible(action_kind: str, prompt: str = "") -> bool:
    if action_kind in IRREVERSIBLE_ACTION_KINDS:
        return False
    return not any(pattern.search(prompt) for pattern in IRREVERSIBLE_PATTERNS)


def assert_reversible(action_kind: str, prompt: str = "") -> None:
    if not is_reversible(action_kind, prompt):
        raise IrreversibleAction(
            f"'{action_kind}' crosses the reversibility boundary; a human must run it"
        )
