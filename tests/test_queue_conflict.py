"""Real git. A conflict must take down one action, not the batch.

"Orthogonal" is a hope, not a guarantee. When two accepted actions touch the
same lines, the merge fails -- and that failure is a training label, not a bug.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from runahead.queue import Entry, Queue, Step, accept
from runahead.store import Store
from runahead.worktree import git


def _repo(path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    git(path, "config", "user.email", "t@example.com")
    git(path, "config", "user.name", "t")
    (path / "x.txt").write_text("original\n", encoding="utf-8")
    (path / "y.txt").write_text("other\n", encoding="utf-8")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "seed")
    return path


def _patch_changing(repo: Path, name: str, content: str) -> str:
    """A patch against HEAD that rewrites one file, without disturbing the repo."""
    head = git(repo, "rev-parse", "HEAD").strip()
    (repo / name).write_text(content, encoding="utf-8")
    git(repo, "add", "-A")
    patch = git(repo, "diff", "--cached", "--binary", "HEAD")
    git(repo, "reset", "--hard", head)
    return patch


def _entry(entry_id: str, patch: str, files: list[str]) -> Entry:
    return Entry(
        id=entry_id,
        group=None,
        auto_accept=False,
        mean=0.5,
        observations=4.0,
        steps=[Step(id=entry_id, kind=entry_id, patch=patch, files=files)],
    )


class TestConflictIsolation(unittest.TestCase):
    def test_conflicting_action_fails_alone_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as rd:
            repo = _repo(Path(rd))
            store = Store.open(repo, root=Path(home))

            first = _entry("act-a", _patch_changing(repo, "x.txt", "from-a\n"), ["x.txt"])
            second = _entry("act-b", _patch_changing(repo, "x.txt", "from-b\n"), ["x.txt"])
            queue = Queue(task="t", task_kind="feature", entries=[first, second])

            outcome = accept(repo, queue, store, {"act-a", "act-b"})

            self.assertEqual(outcome.applied, ["act-a"])
            self.assertEqual(outcome.conflicted, ["act-b"])
            self.assertEqual((repo / "x.txt").read_text(), "from-a\n")

            # The tree is left clean: the failed apply was rolled back.
            self.assertEqual(git(repo, "status", "--porcelain").strip(), "")

            events = [
                json.loads(line)
                for line in store.history_path.read_text().splitlines()
                if line.strip()
            ]
            conflicts = [e for e in events if e["event"] == "conflict"]
            self.assertEqual(len(conflicts), 1)
            self.assertEqual(conflicts[0]["actions"], ["act-a", "act-b"])

    def test_orthogonal_actions_both_apply(self):
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as rd:
            repo = _repo(Path(rd))
            store = Store.open(repo, root=Path(home))

            first = _entry("act-a", _patch_changing(repo, "x.txt", "from-a\n"), ["x.txt"])
            second = _entry("act-b", _patch_changing(repo, "y.txt", "from-b\n"), ["y.txt"])
            queue = Queue(task="t", task_kind="feature", entries=[first, second])

            outcome = accept(repo, queue, store, {"act-a", "act-b"})

            self.assertEqual(outcome.applied, ["act-a", "act-b"])
            self.assertEqual(outcome.conflicted, [])
            self.assertEqual((repo / "x.txt").read_text(), "from-a\n")
            self.assertEqual((repo / "y.txt").read_text(), "from-b\n")

    def test_rejection_is_recorded_as_a_negative_label(self):
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as rd:
            repo = _repo(Path(rd))
            store = Store.open(repo, root=Path(home))

            before = store.posterior("feature", "act-b").mean
            first = _entry("act-a", _patch_changing(repo, "x.txt", "from-a\n"), ["x.txt"])
            second = _entry("act-b", _patch_changing(repo, "y.txt", "from-b\n"), ["y.txt"])
            queue = Queue(task="t", task_kind="feature", entries=[first, second])

            accept(repo, queue, store, {"act-a"})

            self.assertLess(store.posterior("feature", "act-b").mean, before)
            self.assertGreater(store.posterior("feature", "act-a").mean, before)

    def test_mutually_exclusive_variants_cannot_both_be_accepted(self):
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as rd:
            repo = _repo(Path(rd))
            store = Store.open(repo, root=Path(home))

            a = _entry("ui#1", _patch_changing(repo, "x.txt", "a\n"), ["x.txt"])
            b = _entry("ui#2", _patch_changing(repo, "x.txt", "b\n"), ["x.txt"])
            a.group = b.group = "ui"
            queue = Queue(task="t", task_kind="feature", entries=[a, b])

            with self.assertRaises(RuntimeError):
                accept(repo, queue, store, {"ui#1", "ui#2"})

    def test_dirty_tree_is_refused(self):
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as rd:
            repo = _repo(Path(rd))
            store = Store.open(repo, root=Path(home))
            entry = _entry("act-a", _patch_changing(repo, "x.txt", "from-a\n"), ["x.txt"])
            queue = Queue(task="t", task_kind="feature", entries=[entry])

            (repo / "x.txt").write_text("dirty\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                accept(repo, queue, store, {"act-a"})


class TestMissRate(unittest.TestCase):
    def test_miss_rate_counts_only_resolutions(self):
        with tempfile.TemporaryDirectory() as home:
            store = Store.open(Path("/repo/x"), root=Path(home))
            store.append_history({"event": "queue_resolved"})
            store.append_history({"event": "queue_resolved"})
            store.record_miss("feature", "write a migration")
            self.assertAlmostEqual(store.miss_rate(), 1 / 3)

    def test_miss_rate_is_none_without_history(self):
        with tempfile.TemporaryDirectory() as home:
            store = Store.open(Path("/repo/x"), root=Path(home))
            self.assertIsNone(store.miss_rate())


if __name__ == "__main__":
    unittest.main()
