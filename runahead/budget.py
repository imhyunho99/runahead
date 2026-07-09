"""Speculation is reversible, not free.

Nobody is watching for thirty minutes. Without a ceiling the predicted lane will
happily inflate itself and burn the rate limit. This is a v1 part, not a later
refinement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Budget:
    tokens: int = 200_000
    seconds: float = 1800.0
    actions: int = 12

    clock: Callable[[], float] = time.monotonic
    _started: float | None = field(default=None, init=False)
    _tokens_used: int = field(default=0, init=False)
    _actions_used: int = field(default=0, init=False)

    def start(self) -> "Budget":
        self._started = self.clock()
        return self

    @property
    def elapsed(self) -> float:
        return 0.0 if self._started is None else self.clock() - self._started

    @property
    def tokens_used(self) -> int:
        return self._tokens_used

    @property
    def actions_used(self) -> int:
        return self._actions_used

    def charge(self, tokens: int) -> None:
        self._tokens_used += tokens
        self._actions_used += 1

    def exhausted(self) -> bool:
        return (
            self._tokens_used >= self.tokens
            or self._actions_used >= self.actions
            or self.elapsed >= self.seconds
        )

    def can_afford(self, estimated_tokens: int) -> bool:
        if self.exhausted():
            return False
        return self._tokens_used + estimated_tokens <= self.tokens

    def summary(self) -> str:
        return (
            f"tokens {self._tokens_used:,}/{self.tokens:,} · "
            f"{self.elapsed / 60:.0f}m/{self.seconds / 60:.0f}m · "
            f"actions {self._actions_used}/{self.actions}"
        )
