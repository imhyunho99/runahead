"""Core tests. No LLM anywhere: FakeExecutor and FakePredictor stand in.

The one that matters most is test_exposure_bias. Everything else guards a rule;
that one guards the reason the system can learn at all.
"""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from runahead.action import Action
from runahead.beta import Beta
from runahead.budget import Budget
from runahead.executor import FakeExecutor
from runahead.policy import decide
from runahead.predictor import FakePredictor
from runahead.safety import IrreversibleAction, is_reversible
from runahead.scheduler import execute, plan
from runahead.store import Store


class TestBeta(unittest.TestCase):
    def test_mean_and_observations_exclude_prior(self):
        b = Beta(3.0, 1.0)
        self.assertAlmostEqual(b.mean, 0.75)
        self.assertAlmostEqual(b.observations, 2.0)

    def test_variance_shrinks_with_evidence(self):
        few = Beta(2.0, 1.0)
        many = Beta(200.0, 100.0)
        self.assertAlmostEqual(few.mean, many.mean, places=6)
        self.assertLess(many.stdev, few.stdev)

    def test_update(self):
        self.assertEqual(Beta(1.0, 1.0).update(True), Beta(2.0, 1.0))
        self.assertEqual(Beta(1.0, 1.0).update(False), Beta(1.0, 2.0))

    def test_rejects_nonpositive(self):
        with self.assertRaises(ValueError):
            Beta(0.0, 1.0)


class TestPolicy(unittest.TestCase):
    def test_high_mean_with_thin_evidence_is_not_auto_accepted(self):
        """2 of 3 has mean 0.67 and tells us nothing. 47 of 53 does."""
        thin = decide(Beta(3.0, 1.0))
        self.assertFalse(thin.auto_accept)

        thick = decide(Beta(48.0, 6.0))
        self.assertTrue(thick.auto_accept)

    def test_auto_accept_collapses_variants_to_one(self):
        self.assertEqual(decide(Beta(48.0, 6.0)).variants, 1)

    def test_low_confidence_widens_variants(self):
        self.assertEqual(decide(Beta(6.0, 6.0)).variants, 1)
        self.assertEqual(decide(Beta(3.0, 6.0)).variants, 2)
        self.assertEqual(decide(Beta(2.0, 9.0)).variants, 3)

    def test_floor_suppresses_proposal(self):
        self.assertFalse(decide(Beta(1.0, 20.0)).propose)

    def test_children_need_confidence(self):
        self.assertTrue(decide(Beta(8.0, 2.0)).may_expand_children)
        self.assertFalse(decide(Beta(2.0, 8.0)).may_expand_children)


class TestSafety(unittest.TestCase):
    def test_irreversible_kinds_blocked(self):
        self.assertFalse(is_reversible("push"))
        self.assertFalse(is_reversible("deploy"))

    def test_irreversible_commands_blocked_in_prompt(self):
        self.assertFalse(is_reversible("draft-commit", "commit then git push origin main"))
        self.assertFalse(is_reversible("chore", "run terraform apply"))
        self.assertFalse(is_reversible("chore", "gh pr create --fill"))

    def test_ordinary_actions_pass(self):
        self.assertTrue(is_reversible("write-tests", "write tests, do not push"))

    def test_executor_refuses_to_cross_the_boundary(self):
        """Not a test of behaviour. A safety requirement. Confidence never unlocks it."""
        with self.assertRaises(IrreversibleAction):
            FakeExecutor().run(Action(kind="push", prompt="git push"), Path("/tmp"))


class TestBudget(unittest.TestCase):
    def _clock(self):
        self.now = 0.0
        return lambda: self.now

    def test_token_exhaustion(self):
        b = Budget(tokens=1000, actions=100, clock=self._clock()).start()
        b.charge(999)
        self.assertFalse(b.exhausted())
        b.charge(1)
        self.assertTrue(b.exhausted())

    def test_action_exhaustion(self):
        b = Budget(tokens=10**9, actions=2, clock=self._clock()).start()
        b.charge(1)
        b.charge(1)
        self.assertTrue(b.exhausted())

    def test_wall_clock_exhaustion(self):
        b = Budget(tokens=10**9, actions=100, seconds=60, clock=self._clock()).start()
        self.assertFalse(b.exhausted())
        self.now = 61.0
        self.assertTrue(b.exhausted())

    def test_can_afford_looks_ahead(self):
        b = Budget(tokens=1000, clock=self._clock()).start()
        b.charge(900)
        self.assertFalse(b.can_afford(200))
        self.assertTrue(b.can_afford(100))


class TestHierarchicalPrior(unittest.TestCase):
    def test_fresh_repo_inherits_the_global_habit(self):
        """Cold start happens once, in the first repo -- not once per repo."""
        with tempfile.TemporaryDirectory() as home:
            store = Store.open(Path("/repo/a"), root=Path(home))
            for _ in range(30):
                store.record("feature", "write-tests", accepted=True)
            store.flush()

            fresh = Store.open(Path("/repo/b"), root=Path(home))
            self.assertGreater(fresh.posterior("feature", "write-tests").mean, 0.7)
            self.assertAlmostEqual(fresh.posterior("feature", "unseen-action").mean, 0.5)

    def test_repo_observations_override_the_global_prior(self):
        with tempfile.TemporaryDirectory() as home:
            store = Store.open(Path("/repo/a"), root=Path(home))
            for _ in range(30):
                store.record("feature", "responsive-ui", accepted=True)
            store.flush()

            other = Store.open(Path("/repo/b"), root=Path(home))
            for _ in range(20):
                other.record("feature", "responsive-ui", accepted=False)

            self.assertLess(other.posterior("feature", "responsive-ui").mean, 0.4)


class TestExposureBias(unittest.TestCase):
    """The regression that protects the system from its own habits.

    A greedy ranker shows the action with the best posterior mean. If the
    statistics already favour A, B is never shown, so B is never observed, so
    the statistics keep favouring A -- even when the human would have preferred
    B all along. B's superiority is unobservable because it was never on screen.

    Thompson sampling makes B's wide posterior occasionally outdraw A's narrow
    one, which is the only way the truth gets a chance to appear.
    """

    def _store(self, home: str) -> Store:
        store = Store.open(Path("/repo/x"), root=Path(home))
        for _ in range(40):
            store.record("feature", "action-a", accepted=True)
        for _ in range(5):
            store.record("feature", "action-a", accepted=False)
        return store

    def _top_kind_thompson(self, store, rng) -> str:
        chains = plan(
            store,
            FakePredictor(kinds=("action-a", "action-b")),
            task="t",
            task_kind="feature",
            diff="",
            rng=rng,
        )
        return chains[0].root.kind

    def _top_kind_greedy(self, store) -> str:
        candidates = ["action-a", "action-b"]
        return max(candidates, key=lambda k: store.posterior("feature", k).mean)

    def test_greedy_never_shows_the_unexplored_action(self):
        with tempfile.TemporaryDirectory() as home:
            store = self._store(home)
            shown = {self._top_kind_greedy(store) for _ in range(100)}
            self.assertEqual(shown, {"action-a"})

    def test_thompson_sampling_surfaces_the_unexplored_action(self):
        with tempfile.TemporaryDirectory() as home:
            store = self._store(home)
            rng = random.Random(7)
            shown = [self._top_kind_thompson(store, rng) for _ in range(200)]
            self.assertIn("action-b", shown, "exposure bias: B never reached the queue")

    def test_thompson_still_prefers_the_proven_action(self):
        """Exploration must not become indifference."""
        with tempfile.TemporaryDirectory() as home:
            store = self._store(home)
            rng = random.Random(7)
            shown = [self._top_kind_thompson(store, rng) for _ in range(200)]
            self.assertGreater(shown.count("action-a"), shown.count("action-b"))

    def test_a_better_action_is_eventually_learned(self):
        """End to end: B is truly better, starts invisible, and wins."""
        with tempfile.TemporaryDirectory() as home:
            store = self._store(home)
            rng = random.Random(11)
            user = random.Random(3)

            for _ in range(300):
                top = self._top_kind_thompson(store, rng)
                truth = 0.90 if top == "action-a" else 0.98
                store.record("feature", top, accepted=user.random() < truth)

            self.assertGreater(store.posterior("feature", "action-b").observations, 0)
            self.assertGreater(store.posterior("feature", "action-b").mean, 0.8)


class TestSchedulerRespectsBoundaries(unittest.TestCase):
    def test_irreversible_proposals_never_enter_the_plan(self):
        with tempfile.TemporaryDirectory() as home:
            store = Store.open(Path("/repo/x"), root=Path(home))
            chains = plan(
                store,
                FakePredictor(kinds=("write-tests", "push", "deploy")),
                task="t",
                task_kind="feature",
                diff="",
                rng=random.Random(1),
            )
            kinds = {c.root.kind for c in chains}
            self.assertEqual(kinds, {"write-tests"})

    def test_budget_stops_expansion(self):
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as repo_dir:
            repo = _make_repo(Path(repo_dir))
            store = Store.open(repo, root=Path(home))
            chains = plan(
                store,
                FakePredictor(kinds=("write-tests", "lint-typecheck", "add-edge-cases")),
                task="t",
                task_kind="feature",
                diff="",
                rng=random.Random(1),
            )
            budget = Budget(tokens=10**9, actions=1, seconds=10**9).start()
            done = execute(repo, FakeExecutor(), chains, budget)

            self.assertEqual(budget.actions_used, 1)
            self.assertLessEqual(sum(len(c.results) for c in done), 1)


def _make_repo(path: Path) -> Path:
    from runahead.worktree import git

    import subprocess

    subprocess.run(["git", "init", "-q", str(path)], check=True)
    git(path, "config", "user.email", "t@example.com")
    git(path, "config", "user.name", "t")
    (path / "seed.txt").write_text("seed\n", encoding="utf-8")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "seed")
    return path


if __name__ == "__main__":
    unittest.main()
