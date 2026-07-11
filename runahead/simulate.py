"""Does p actually converge? The one claim the whole tool rests on.

Settling it with real agents means dozens of four-minute sessions. Instead we
drive the real learning path -- plan()'s Thompson ranking and the policy's
suppression, the same code the CLI runs -- against a synthetic user whose true
preferences we fixed in advance, and watch the statistics find them.

This validates the mechanism under a stated user model. It does not prove real
humans behave this way; only real use does that. But if convergence failed
here, it could not possibly work there, so this is the cheap falsifier.

The dynamic that matters: the review queue holds fewer actions than the catalog
offers, so something must be left off. Early on every posterior sits near 0.5
and Thompson draws are noisy, so a genuinely-wanted action sometimes ranks
below the cutoff and never gets shown -- a miss. As its posterior sharpens, it
reliably takes a top slot, and misses die out. That is exposure bias being
defeated in slow motion.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from .predictor import FakePredictor
from .scheduler import plan
from .store import Store

WANT_THRESHOLD = 0.7
"""A user 'wants' any action they would accept most of the time."""


@dataclass
class SessionRecord:
    index: int
    shown: list[str]
    accepted: list[str]
    miss: bool
    means: dict[str, float]


@dataclass
class SimResult:
    prefs: dict[str, float]
    sessions: list[SessionRecord] = field(default_factory=list)
    final_means: dict[str, float] = field(default_factory=dict)

    def miss_rate(self, lo: float, hi: float) -> float:
        window = self.sessions[int(len(self.sessions) * lo) : int(len(self.sessions) * hi)]
        if not window:
            return 0.0
        return sum(1 for s in window if s.miss) / len(window)

    def shown_rate(self, kind: str, lo: float, hi: float) -> float:
        window = self.sessions[int(len(self.sessions) * lo) : int(len(self.sessions) * hi)]
        if not window:
            return 0.0
        return sum(1 for s in window if kind in s.shown) / len(window)


def simulate(
    prefs: dict[str, float],
    sessions: int,
    queue_size: int,
    home: Path,
    seed: int = 0,
    task_kind: str = "feature",
) -> SimResult:
    """Run `sessions` review cycles against a user with fixed true `prefs`.

    prefs maps action kind -> probability the user accepts it when shown.
    """
    store = Store.open(Path("/sim/repo"), root=home)
    predictor = FakePredictor(kinds=tuple(prefs), task_kind=task_kind)

    rank_rng = random.Random(seed)
    user_rng = random.Random(seed + 1)
    wanted = {k for k, p in prefs.items() if p >= WANT_THRESHOLD}

    result = SimResult(prefs=dict(prefs))

    for i in range(sessions):
        chains = plan(store, predictor, "task", task_kind, "", rank_rng)

        shown: list[str] = []
        for chain in chains:  # already sorted by Thompson sample, descending
            if chain.root.kind not in shown:
                shown.append(chain.root.kind)
            if len(shown) >= queue_size:
                break

        accepted = [k for k in shown if user_rng.random() < prefs[k]]

        # A miss is a wanted action the queue never offered -- exactly what the
        # human means by "none of these; do this other thing instead".
        missed = wanted - set(shown)
        miss = bool(missed)

        for kind in shown:
            store.record(task_kind, kind, accepted=kind in accepted)
        if miss:
            store.record_miss(task_kind, sorted(missed)[0])
        else:
            store.append_history({"event": "queue_resolved", "task_kind": task_kind})

        result.sessions.append(
            SessionRecord(
                index=i,
                shown=shown,
                accepted=accepted,
                miss=miss,
                means={k: store.posterior(task_kind, k).mean for k in prefs},
            )
        )

    store.flush()
    result.final_means = {k: store.posterior(task_kind, k).mean for k in prefs}
    return result


def render(result: SimResult, buckets: int = 10) -> str:
    """A compact miss-rate curve plus where each action's p landed."""
    n = len(result.sessions)
    lines = [f"sessions: {n}   catalog: {len(result.prefs)}", ""]

    lines.append("miss rate, early to late (each column = equal slice of the run)")
    width = 1.0 / buckets
    ramp = " .:-=+*#%@"
    cols, sparks = [], []
    for b in range(buckets):
        rate = result.miss_rate(b * width, (b + 1) * width)
        cols.append(f"{rate:3.0%}")
        sparks.append(ramp[min(len(ramp) - 1, round(rate * (len(ramp) - 1)))])
    lines.append("  " + " ".join(cols))
    lines.append("  " + "   ".join(sparks))
    lines.append("")

    lines.append(f"{'action':<20} {'true p':>7} {'learned':>8} {'shown early':>12} {'shown late':>11}")
    for kind in sorted(result.prefs, key=result.prefs.get, reverse=True):
        lines.append(
            f"{kind:<20} {result.prefs[kind]:>7.2f} {result.final_means[kind]:>8.2f} "
            f"{result.shown_rate(kind, 0.0, 0.2):>11.0%} {result.shown_rate(kind, 0.8, 1.0):>10.0%}"
        )
    return "\n".join(lines)
