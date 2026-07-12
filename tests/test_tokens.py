"""Per-agent token reporting. The spend is captured per action, surfaced in
the queue, and accumulated per action kind across sessions."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from runahead.action import Action, Result
from runahead.queue import Queue, Step
from runahead.scheduler import Chain
from runahead.store import Store


def _chain(kind: str, tokens_per_step: list[int]) -> Chain:
    actions = [Action(kind=kind if i == 0 else f"{kind}-step{i}", prompt="p") for i in range(len(tokens_per_step))]
    results = [
        Result(action=a, patch=f"diff --git a/{a.kind} b/{a.kind}\n", tokens=t)
        for a, t in zip(actions, tokens_per_step)
    ]
    return Chain(
        actions=actions,
        score=0.5,
        mean=0.5,
        observations=4.0,
        auto_accept=False,
        results=results,
    )


class TestQueueTokens(unittest.TestCase):
    def test_step_carries_tokens_from_result(self):
        q = Queue.build("t", "feature", [_chain("write-tests", [1200])])
        self.assertEqual(q.entries[0].steps[0].tokens, 1200)

    def test_entry_tokens_sum_the_chain(self):
        q = Queue.build("t", "feature", [_chain("write-tests", [1000, 500, 300])])
        self.assertEqual(q.entries[0].tokens, 1800)

    def test_tokens_survive_save_and_load(self):
        with tempfile.TemporaryDirectory() as rd:
            import subprocess

            from runahead.worktree import git

            repo = Path(rd)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            git(repo, "config", "user.email", "t@e.com")
            git(repo, "config", "user.name", "t")

            q = Queue.build("t", "feature", [_chain("write-tests", [1000, 500])])
            q.save(repo)
            loaded = Queue.load(repo)
            self.assertEqual(loaded.entries[0].tokens, 1500)

    def test_render_shows_spend(self):
        q = Queue.build("t", "feature", [_chain("write-tests", [2500])])
        text = q.render()
        self.assertIn("2,500 tok", text)
        self.assertIn("spent while you were away", text)


class TestTokenLedger(unittest.TestCase):
    def test_records_calls_and_totals_per_action(self):
        with tempfile.TemporaryDirectory() as home:
            store = Store.open(Path("/repo/x"), root=Path(home))
            store.record_tokens("feature", "write-tests", 1000)
            store.record_tokens("feature", "write-tests", 3000)
            store.record_tokens("feature", "update-docs", 500)

            ledger = store.token_ledger()
            self.assertEqual(ledger["feature|write-tests"]["calls"], 2)
            self.assertEqual(ledger["feature|write-tests"]["tokens"], 4000)
            self.assertEqual(ledger["feature|write-tests"]["avg"], 2000)
            self.assertEqual(ledger["feature|update-docs"]["tokens"], 500)

    def test_ledger_persists(self):
        with tempfile.TemporaryDirectory() as home:
            store = Store.open(Path("/repo/x"), root=Path(home))
            store.record_tokens("feature", "write-tests", 1234)
            store.flush()

            reopened = Store.open(Path("/repo/x"), root=Path(home))
            self.assertEqual(reopened.token_ledger()["feature|write-tests"]["tokens"], 1234)

    def test_ledger_is_separate_from_accept_reject(self):
        """Spend is charged whether or not the action is accepted."""
        with tempfile.TemporaryDirectory() as home:
            store = Store.open(Path("/repo/x"), root=Path(home))
            store.record_tokens("feature", "perf-pass", 5000)  # ran, cost tokens
            store.record("feature", "perf-pass", accepted=False)  # but rejected
            self.assertEqual(store.token_ledger()["feature|perf-pass"]["tokens"], 5000)
            self.assertLess(store.posterior("feature", "perf-pass").mean, 0.5)


class TestLedgerResilience(unittest.TestCase):
    """A corrupt or half-written tokens.json must never crash `stats`/`run`."""

    def _store_over(self, home: str, tokens_contents: str) -> Store:
        (Path(home) / "tokens.json").write_text(tokens_contents, encoding="utf-8")
        return Store.open(Path("/repo/x"), root=Path(home))

    def test_invalid_json_degrades_to_empty_ledger(self):
        with tempfile.TemporaryDirectory() as home:
            store = self._store_over(home, "{ this is not json ")
            self.assertEqual(store.token_ledger(), {})

    def test_truncated_json_degrades_to_empty_ledger(self):
        # The shape an interrupted writer leaves behind: a valid prefix.
        with tempfile.TemporaryDirectory() as home:
            store = self._store_over(home, '{"feature|write-tests": {"calls": 2, "tok')
            self.assertEqual(store.token_ledger(), {})

    def test_whitespace_only_file_degrades_to_empty_ledger(self):
        with tempfile.TemporaryDirectory() as home:
            store = self._store_over(home, "   \n  ")
            self.assertEqual(store.token_ledger(), {})

    def test_non_object_json_degrades_to_empty_ledger(self):
        with tempfile.TemporaryDirectory() as home:
            store = self._store_over(home, "[1, 2, 3]")
            self.assertEqual(store.token_ledger(), {})

    def test_corrupt_ledger_still_records_and_reflushes_cleanly(self):
        with tempfile.TemporaryDirectory() as home:
            store = self._store_over(home, "{ broken")
            store.record_tokens("feature", "write-tests", 900)
            store.flush()

            reopened = Store.open(Path("/repo/x"), root=Path(home))
            self.assertEqual(
                reopened.token_ledger()["feature|write-tests"]["tokens"], 900
            )

    def test_flush_is_atomic_and_leaves_no_temp_files(self):
        with tempfile.TemporaryDirectory() as home:
            store = Store.open(Path("/repo/x"), root=Path(home))
            store.record_tokens("feature", "write-tests", 1234)
            store.flush()

            root = Path(home)
            self.assertEqual(list(root.glob("*.tmp")), [])
            # The persisted file parses on its own -- no torn write survived.
            reloaded = json.loads((root / "tokens.json").read_text(encoding="utf-8"))
            self.assertEqual(reloaded["feature|write-tests"]["tokens"], 1234)


if __name__ == "__main__":
    unittest.main()
