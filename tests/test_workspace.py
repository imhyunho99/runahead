"""Workspace mode: discovery and per-repo isolation. No LLM.

The workspace layer only decides which repos to run; the per-repo core is
unchanged. These tests pin the discovery rules and confirm each repo keeps its
own queue and its own learning.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from runahead import workspace
from runahead.worktree import git


def _repo(path: Path, when: int | None = None) -> Path:
    import os

    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    git(path, "config", "user.email", "t@e.com")
    git(path, "config", "user.name", "t")
    (path / "seed.txt").write_text("x", encoding="utf-8")
    git(path, "add", "-A")

    env = dict(os.environ)
    if when is not None:
        stamp = f"@{when} +0000"  # git's epoch date form
        env["GIT_AUTHOR_DATE"] = stamp
        env["GIT_COMMITTER_DATE"] = stamp
    subprocess.run(
        ["git", "-C", str(path), "commit", "-q", "-m", f"work in {path.name}"],
        check=True,
        env=env,
    )
    return path


NOW = 1_800_000_000


class TestDiscovery(unittest.TestCase):
    def test_finds_only_child_git_repos(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            _repo(root / "backend")
            _repo(root / "frontend")
            (root / "notes").mkdir()  # not a repo
            (root / "loose.txt").write_text("x")
            names = [p.name for p in workspace.find_repos(root)]
            self.assertEqual(names, ["backend", "frontend"])

    def test_active_window_excludes_stale_repos(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            _repo(root / "fresh", when=NOW - 2 * 86400)  # 2 days
            _repo(root / "stale", when=NOW - 40 * 86400)  # 40 days
            active, skipped = workspace.active_repos(root, NOW, within_days=14)
            self.assertEqual([a.name for a in active], ["fresh"])
            self.assertEqual([a.name for a in skipped], ["stale"])

    def test_active_repos_sorted_newest_first(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            _repo(root / "older", when=NOW - 5 * 86400)
            _repo(root / "newer", when=NOW - 1 * 86400)
            active, _ = workspace.active_repos(root, NOW, within_days=14)
            self.assertEqual([a.name for a in active], ["newer", "older"])

    def test_repos_pin_ignores_time_window(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            _repo(root / "backend", when=NOW - 100 * 86400)  # very stale
            _repo(root / "frontend", when=NOW - 1 * 86400)
            active, skipped = workspace.active_repos(root, NOW, within_days=14, only=["backend"])
            self.assertEqual([a.name for a in active], ["backend"])
            self.assertEqual([a.name for a in skipped], ["frontend"])

    def test_repo_without_commits_is_ignored(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            empty = root / "empty"
            empty.mkdir()
            subprocess.run(["git", "init", "-q", str(empty)], check=True)
            _repo(root / "real", when=NOW - 1 * 86400)
            active, _ = workspace.active_repos(root, NOW, within_days=14)
            self.assertEqual([a.name for a in active], ["real"])


class TestWorkspaceRun(unittest.TestCase):
    def test_each_repo_gets_its_own_queue(self):
        import os

        from runahead.cli import _run_workspace

        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as home, \
                tempfile.TemporaryDirectory() as wt:
            root = Path(root)
            _repo(root / "backend", when=NOW - 1 * 86400)
            _repo(root / "frontend", when=NOW - 2 * 86400)

            from runahead.executor import FakeExecutor

            os.environ["RUNAHEAD_HOME"] = home
            os.environ["RUNAHEAD_WORKTREES"] = wt
            try:
                args = _Args()
                code = _run_workspace(root, args, executor=FakeExecutor())
            finally:
                for k in ("RUNAHEAD_HOME", "RUNAHEAD_WORKTREES"):
                    os.environ.pop(k, None)

            self.assertEqual(code, 0)
            from runahead.queue import Queue

            self.assertIsNotNone(Queue.load(root / "backend"))
            self.assertIsNotNone(Queue.load(root / "frontend"))


class _Args:
    task = None
    tokens = 10**9
    minutes = 10**6
    actions = 6
    seed = 1
    force = False
    within = 14.0
    repos = None


if __name__ == "__main__":
    unittest.main()
