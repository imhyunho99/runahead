"""One dial: confidence p drives variant count, tree expansion, and auto-accept.

Auto-accept requires a high mean AND a tight posterior. 2 of 3 successes has a
mean of 0.67 but tells us almost nothing; 47 of 53 does.
"""

from __future__ import annotations

from dataclasses import dataclass

from .beta import Beta

AUTO_ACCEPT_MEAN = 0.85
AUTO_ACCEPT_MIN_OBSERVATIONS = 20.0
AUTO_ACCEPT_MAX_STDEV = 0.06

EXPAND_CHILDREN_MEAN = 0.60

VARIANTS_ONE_MEAN = 0.50
VARIANTS_TWO_MEAN = 0.25
PROPOSE_FLOOR_MEAN = 0.12


@dataclass(frozen=True)
class Decision:
    propose: bool
    variants: int
    may_expand_children: bool
    auto_accept: bool

    @property
    def lane(self) -> str:
        return "fixed" if self.auto_accept else "predicted"


def decide(posterior: Beta) -> Decision:
    mean = posterior.mean

    auto = (
        mean >= AUTO_ACCEPT_MEAN
        and posterior.observations >= AUTO_ACCEPT_MIN_OBSERVATIONS
        and posterior.stdev <= AUTO_ACCEPT_MAX_STDEV
    )

    if mean < PROPOSE_FLOOR_MEAN:
        return Decision(propose=False, variants=0, may_expand_children=False, auto_accept=False)

    if mean >= VARIANTS_ONE_MEAN:
        variants = 1
    elif mean >= VARIANTS_TWO_MEAN:
        variants = 2
    else:
        variants = 3

    return Decision(
        propose=True,
        variants=1 if auto else variants,
        may_expand_children=mean >= EXPAND_CHILDREN_MEAN,
        auto_accept=auto,
    )
