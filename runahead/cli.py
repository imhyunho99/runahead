from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from .budget import Budget
from .executor import ClaudeExecutor
from .predictor import ClaudePredictor
from .queue import Queue, accept
from .scheduler import execute, plan
from .store import Store
from .worktree import git, repo_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="runahead", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="speculate on what comes after the task you just finished")
    run.add_argument("task", help="what was just implemented")
    run.add_argument("--tokens", type=int, default=200_000)
    run.add_argument("--minutes", type=float, default=30.0)
    run.add_argument("--actions", type=int, default=12)
    run.add_argument("--seed", type=int, default=None)

    sub.add_parser("queue", help="show what is waiting for review")

    acc = sub.add_parser("accept", help="apply chosen results and record the labels")
    acc.add_argument("ids", nargs="*")
    acc.add_argument("--all", action="store_true")

    miss = sub.add_parser("miss", help="record that the queue offered nothing you wanted")
    miss.add_argument("requested", help="what you actually wanted next")

    sub.add_parser("stats", help="miss rate and learned confidences")

    args = parser.parse_args(argv)
    repo = repo_root(Path.cwd())
    store = Store.open(repo)

    if args.command == "run":
        return _run(repo, store, args)
    if args.command == "queue":
        return _queue(repo)
    if args.command == "accept":
        return _accept(repo, store, args)
    if args.command == "miss":
        return _miss(repo, store, args)
    if args.command == "stats":
        return _stats(store)
    return 1


def _run(repo: Path, store: Store, args) -> int:
    if git(repo, "status", "--porcelain").strip():
        print("working tree is dirty; commit the finished task first", file=sys.stderr)
        return 1

    predictor = ClaudePredictor()
    task_kind = predictor.classify(args.task)
    diff = git(repo, "show", "--patch", "--stat", "HEAD")

    rng = random.Random(args.seed)
    budget = Budget(tokens=args.tokens, seconds=args.minutes * 60, actions=args.actions).start()

    chains = plan(store, predictor, args.task, task_kind, diff, rng)
    if not chains:
        print("no candidates above the proposal floor")
        return 0

    print(f"speculating on {len(chains)} chains ({task_kind})", file=sys.stderr)
    done = execute(repo, ClaudeExecutor(), chains, budget)

    barren = budget.actions_used - sum(len(c.results) for c in done)
    if barren:
        # An agent that edits nothing still costs tokens and still reports
        # success. Never let that pass silently.
        print(f"{barren} action(s) produced no patch", file=sys.stderr)

    queue = Queue.build(args.task, task_kind, done)
    queue.save(repo)
    print(queue.render(budget.summary()))
    return 0


def _queue(repo: Path) -> int:
    queue = Queue.load(repo)
    if not queue:
        print("queue is empty")
        return 0
    print(queue.render())
    return 0


def _accept(repo: Path, store: Store, args) -> int:
    queue = Queue.load(repo)
    if not queue:
        print("queue is empty", file=sys.stderr)
        return 1

    selected = {e.id for e in queue.entries} if args.all else set(args.ids)
    outcome = accept(repo, queue, store, selected)

    for entry_id in outcome.applied:
        print(f"applied     {entry_id}")
    for entry_id in outcome.conflicted:
        print(f"conflicted  {entry_id}  (not orthogonal; recorded)")
    for entry_id in outcome.rejected:
        print(f"rejected    {entry_id}")
    return 0


def _miss(repo: Path, store: Store, args) -> int:
    queue = Queue.load(repo)
    task_kind = queue.task_kind if queue else "unknown"

    if queue:
        for entry in queue.entries:
            for kind in set(entry.kinds):
                store.record(task_kind, kind, accepted=False)
        queue.clear(repo)

    store.record_miss(task_kind, args.requested)
    store.flush()
    print("recorded. this is the signal that actually moves the predictor.")
    return 0


def _stats(store: Store) -> int:
    rate = store.miss_rate()
    print(f"miss rate: {'n/a' if rate is None else f'{rate:.0%}'}")
    print()
    rows = []
    for task_kind, action_kind in store.known_pairs():
        posterior = store.posterior(task_kind, action_kind)
        rows.append((posterior.mean, f"{task_kind}|{action_kind}", posterior))
    for mean, key, posterior in sorted(rows, key=lambda r: r[0], reverse=True):
        print(f"  {key:<36} p={mean:.2f}  sd={posterior.stdev:.2f}  n={int(posterior.observations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
