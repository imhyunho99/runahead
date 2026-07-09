"""Beta counters. The whole of runahead's learning is here.

A single Beta(alpha, beta) per (task_kind, action_kind) pair. Posterior mean
gives confidence; posterior variance tells us whether to believe it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Beta:
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError(f"Beta requires positive parameters, got ({self.alpha}, {self.beta})")

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b, n = self.alpha, self.beta, self.alpha + self.beta
        return (a * b) / (n * n * (n + 1.0))

    @property
    def stdev(self) -> float:
        return math.sqrt(self.variance)

    @property
    def observations(self) -> float:
        """Effective sample size, excluding the prior's two pseudo-counts."""
        return max(0.0, self.alpha + self.beta - 2.0)

    def update(self, accepted: bool) -> "Beta":
        if accepted:
            return Beta(self.alpha + 1.0, self.beta)
        return Beta(self.alpha, self.beta + 1.0)

    def sample(self, rng: random.Random) -> float:
        """Thompson sampling. Draw a plausible p rather than trusting the mean.

        This is what keeps a low-observation action from being permanently
        invisible: its wide posterior occasionally produces a high draw.
        """
        return rng.betavariate(self.alpha, self.beta)


UNIFORM = Beta(1.0, 1.0)
