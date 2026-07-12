"""Workspace mode: one command, several sibling repos.

runahead's unit is one git repo -- one HEAD, one worktree, one line of
learning. A workspace (onz, a monorepo-of-repos) is not a git repo; it is a
directory of them. This layer sits on top and does exactly one thing: find the
repos you have been working in and run the ordinary per-repo speculation in
each.

Nothing below changes. Each repo keeps its own Store, its own worktrees, its
own queue -- because backend habits are not frontend habits, and the
hierarchical prior was built so each repo grows its own character. The
workspace only decides *which* repos to run and pools the budget across them.

What this does NOT do: a single action spanning two repos (change the backend
contract, regenerate the frontend client). That needs a different sandbox than
a git worktree, and is deliberately out of scope here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .worktree import git


def is_git_repo(path: Path) -> bool:
    return (path / ".git").exists()


def find_repos(root: Path) -> list[Path]:
    """Immediate child directories that are git repos, sorted by name."""
    if not root.is_dir():
        return []
    return sorted(
        (child for child in root.iterdir() if child.is_dir() and is_git_repo(child)),
        key=lambda p: p.name.lower(),
    )


@dataclass
class RepoActivity:
    path: Path
    last_commit_at: int  # epoch seconds; 0 if the repo has no commits
    subject: str

    @property
    def name(self) -> str:
        return self.path.name


def activity(repo: Path) -> RepoActivity:
    at = git(repo, "log", "-1", "--format=%ct", check=False).strip()
    subject = git(repo, "log", "-1", "--format=%s", check=False).strip()
    return RepoActivity(path=repo, last_commit_at=int(at) if at.isdigit() else 0, subject=subject)


def active_repos(
    root: Path,
    now: int,
    within_days: float,
    only: list[str] | None = None,
) -> tuple[list[RepoActivity], list[RepoActivity]]:
    """Split the workspace's repos into (active, skipped).

    Active = committed within `within_days` of `now`, newest first, so the repo
    you touched most recently gets first claim on the shared budget. `only`
    pins an explicit set by name and ignores the time window.
    """
    acts = [activity(r) for r in find_repos(root)]
    acts = [a for a in acts if a.last_commit_at > 0]  # a repo with no commits has no task

    if only:
        wanted = {name.lower() for name in only}
        active = [a for a in acts if a.name.lower() in wanted]
        skipped = [a for a in acts if a.name.lower() not in wanted]
    else:
        window = within_days * 86400
        active = [a for a in acts if now - a.last_commit_at <= window]
        skipped = [a for a in acts if now - a.last_commit_at > window]

    active.sort(key=lambda a: a.last_commit_at, reverse=True)
    skipped.sort(key=lambda a: a.last_commit_at, reverse=True)
    return active, skipped


def age(now: int, then: int) -> str:
    if then <= 0:
        return "no commits"
    secs = max(0, now - then)
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"
