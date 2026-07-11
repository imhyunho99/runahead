"""The claim the tool lives or dies on: p converges, misses fall.

Not a proof about real humans -- a proof that the learning path (plan()'s
Thompson ranking, the policy, the store) finds a preference that is actually
there. If this failed, real use could not possibly succeed.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from runahead.simulate import simulate

# Two reliably-wanted actions, four medium distractors, a queue that holds two.
PREFS = {
    "write-tests": 0.90,
    "draft-commit": 0.85,
    "add-edge-cases": 0.48,
    "error-handling": 0.45,
    "update-docs": 0.40,
    "perf-pass": 0.35,
}
WANTED = ("write-tests", "draft-commit")
DISTRACTORS = ("add-edge-cases", "error-handling", "update-docs", "perf-pass")


class TestConvergence(unittest.TestCase):
    def _run(self, seed: int):
        with tempfile.TemporaryDirectory() as home:
            return simulate(PREFS, sessions=500, queue_size=2, home=Path(home), seed=seed)

    def test_wanted_confidence_tracks_the_truth(self):
        """Actions that stay in the queue are estimated accurately."""
        r = self._run(seed=5)
        for kind in WANTED:
            self.assertLess(
                abs(r.final_means[kind] - PREFS[kind]),
                0.1,
                f"{kind}: learned {r.final_means[kind]:.2f} vs true {PREFS[kind]:.2f}",
            )

    def test_distractors_converge_to_an_underestimate(self):
        """An honest consequence of exposure: once an action stops being shown
        it stops being learned about, so its estimate freezes below the truth.
        That is fine -- the estimate only has to be low enough to keep it out,
        not accurate. Thompson sampling keeps it from collapsing to zero."""
        r = self._run(seed=5)
        for kind in DISTRACTORS:
            self.assertLessEqual(r.final_means[kind], PREFS[kind] + 0.05)
            self.assertLess(r.final_means[kind], min(r.final_means[k] for k in WANTED))
            self.assertGreater(r.final_means[kind], 0.1)

    def test_wanted_actions_outrank_distractors(self):
        r = self._run(seed=5)
        worst_wanted = min(r.final_means[k] for k in WANTED)
        best_distractor = max(r.final_means[k] for k in DISTRACTORS)
        self.assertGreater(worst_wanted, best_distractor)

    def test_miss_rate_declines(self):
        r = self._run(seed=5)
        early = r.miss_rate(0.0, 0.2)
        late = r.miss_rate(0.8, 1.0)
        self.assertGreater(early, late)
        self.assertLess(late, 0.03)

    def test_wanted_actions_own_the_queue_by_the_end(self):
        r = self._run(seed=5)
        for kind in WANTED:
            self.assertGreater(r.shown_rate(kind, 0.8, 1.0), 0.9)

    def test_convergence_is_not_a_lucky_seed(self):
        for seed in (1, 2, 3, 4):
            r = self._run(seed=seed)
            self.assertLess(r.miss_rate(0.8, 1.0), 0.05, f"seed {seed}")
            for kind in WANTED:
                self.assertGreater(r.final_means[kind], 0.78, f"{kind} seed {seed}")

    def test_a_queue_smaller_than_the_want_set_cannot_reach_zero_misses(self):
        """An honest limit: if the human wants more than fits, learning cannot
        invent slots. Miss rate floors at the structural minimum, not zero."""
        prefs = {"a": 0.9, "b": 0.9, "c": 0.9, "x": 0.2, "y": 0.2}
        with tempfile.TemporaryDirectory() as home:
            r = simulate(prefs, sessions=300, queue_size=2, home=Path(home), seed=1)
        self.assertGreater(r.miss_rate(0.8, 1.0), 0.5)


if __name__ == "__main__":
    unittest.main()
