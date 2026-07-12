"""Best-first expansion under a budget.

Depth is not a global parameter. A node earns children by being confident.
At p = 0.6 a depth-2 guess is worth 0.36, and it sits on top of a depth-1 guess
that was probably wrong -- cost grows linearly, expected value decays
geometrically. Uniform depth is the wrong shape.

Confident paths deepen on their own. Unconfident ones stay wide and shallow.
Either way the reversibility boundary caps the whole thing long before p does.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from .action import Action, Result
from .budget import Budget
from .executor import Executor
from .policy import decide
from .predictor import Predictor, Proposal, prompt_for, successors
from .safety import is_reversible
from .store import Store
from .worktree import git, worktree

MAX_DEPTH_PREDICTED = 1
MAX_DEPTH_FIXED = 3
ESTIMATED_TOKENS_PER_ACTION = 250_000
"""A real agent reading a real repo spends most of its tokens on cache reads
across many turns -- hundreds of thousands per action, not the fake executor's
fictional thousand. can_afford() must estimate against reality or it overshoots."""


@dataclass
class Chain:
    """An ordered run of actions sharing one worktree.

    Step k+1 sees step k's edits, so a chain is accepted as a prefix or not
    at all. Competing variants of the same root form a radio group.
    """

    actions: list[Action]
    score: float
    mean: float
    observations: float
    auto_accept: bool
    group: str | None = None
    results: list[Result] = field(default_factory=list)

    @property
    def root(self) -> Action:
        return self.actions[0]

    @property
    def lane(self) -> str:
        return "fixed" if self.auto_accept else "predicted"


def plan(
    store: Store,
    predictor: Predictor,
    task: str,
    task_kind: str,
    diff: str,
    rng: random.Random,
) -> list[Chain]:
    proposals = [
        p for p in predictor.propose(task, task_kind, diff) if is_reversible(p.kind, p.prompt)
    ]

    chains: list[Chain] = []
    for proposal in proposals:
        posterior = store.posterior(task_kind, proposal.kind)
        verdict = decide(posterior)
        if not verdict.propose:
            continue

        # Thompson sampling, not the posterior mean. An action with a wide
        # posterior occasionally draws high and gets onto the queue -- which is
        # the only way the human's real preference can ever be observed if the
        # statistics have already narrowed onto something else.
        score = posterior.sample(rng)

        group = proposal.kind if verdict.variants > 1 else None
        for v in range(verdict.variants):
            variant_n = 0 if verdict.variants == 1 else v + 1
            actions = _chain_from(
                proposal, variant_n, verdict.auto_accept, verdict.may_expand_children
            )
            chains.append(
                Chain(
                    actions=actions,
                    score=score,
                    mean=posterior.mean,
                    observations=posterior.observations,
                    auto_accept=verdict.auto_accept,
                    group=group,
                )
            )

    chains.sort(key=lambda c: c.score, reverse=True)
    return chains


def _chain_from(proposal: Proposal, variant_n: int, auto: bool, may_expand: bool) -> list[Action]:
    prompt = proposal.prompt
    if variant_n:
        prompt = (
            f"{prompt}\n\nProduce approach #{variant_n}. "
            "It must differ substantively from the alternatives."
        )
    root = Action(kind=proposal.kind, prompt=prompt, variant=variant_n)
    actions = [root]

    if not may_expand:
        return actions

    limit = MAX_DEPTH_FIXED if auto else MAX_DEPTH_PREDICTED
    kind = proposal.kind
    while len(actions) - 1 < limit:
        nexts = successors(kind)
        if not nexts:
            break
        kind = nexts[0]
        prompt = prompt_for(kind)
        if not is_reversible(kind, prompt):
            break
        actions.append(
            Action(kind=kind, prompt=prompt, depth=len(actions), parent=actions[-1].id)
        )
    return actions


def execute(
    repo: Path,
    executor: Executor,
    chains: list[Chain],
    budget: Budget,
) -> list[Chain]:
    """Run chains in score order until the budget says stop."""
    done: list[Chain] = []
    for index, chain in enumerate(chains):
        if not budget.can_afford(ESTIMATED_TOKENS_PER_ACTION):
            break
        chain.results = _run_chain(repo, executor, chain, budget, index)
        if chain.results:
            done.append(chain)
    return done


def _run_chain(
    repo: Path,
    executor: Executor,
    chain: Chain,
    budget: Budget,
    index: int,
) -> list[Result]:
    name = f"{index:02d}-{chain.root.id.replace('#', '-')}"
    results: list[Result] = []

    with worktree(repo, name) as work:
        for action in chain.actions:
            if not budget.can_afford(ESTIMATED_TOKENS_PER_ACTION):
                break

            result = executor.run(action, work)
            budget.charge(result.tokens)

            if not result.ok or result.empty:
                break

            # Commit each step so the next one starts from it and every action
            # still carries its own isolated patch.
            git(work, "add", "-A", check=False)
            git(work, "commit", "-q", "-m", f"runahead: {action.id}", check=False)
            results.append(result)

    return results
