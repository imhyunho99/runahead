"""Git plumbing. The sandbox that makes a wrong guess free."""

from __future__ import annotations

import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class GitError(RuntimeError):
    pass


def git(repo: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
    )
    if check and proc.returncode != 0:
        raise GitError(f"git {' '.join(args)}: {proc.stderr.strip()}")
    return proc.stdout


def repo_root(start: Path) -> Path:
    out = git(start, "rev-parse", "--show-toplevel").strip()
    if not out:
        raise GitError("not inside a git repository")
    return Path(out)


def state_dir(repo: Path) -> Path:
    path = repo / ".git" / "runahead"
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def worktree(repo: Path, name: str) -> Iterator[Path]:
    """A detached worktree at HEAD, removed on exit.

    Speculation happens here and nowhere else.
    """
    path = state_dir(repo) / "worktrees" / name
    if path.exists():
        remove_worktree(repo, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    git(repo, "worktree", "add", "--detach", "--quiet", str(path), "HEAD")
    try:
        yield path
    finally:
        remove_worktree(repo, path)


def remove_worktree(repo: Path, path: Path) -> None:
    git(repo, "worktree", "remove", "--force", str(path), check=False)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    git(repo, "worktree", "prune", check=False)


def capture_patch(work: Path) -> str:
    """Everything the agent changed, as one patch against HEAD."""
    git(work, "add", "-A")
    return git(work, "diff", "--cached", "--binary", "HEAD")


def apply_patch(repo: Path, patch: str) -> tuple[bool, str]:
    """Apply one action's patch onto the working tree.

    A failure here is not a bug. It means two actions we believed orthogonal
    were not, and that pair becomes a training label.
    """
    proc = subprocess.run(
        ["git", "-C", str(repo), "apply", "--3way", "--index"],
        input=patch,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0, proc.stderr.strip()
