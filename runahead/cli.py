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
    run.add_argument(
        "task",
        nargs="?",
        default=None,
        help="what was just implemented (default: the last commit's message)",
    )
    run.add_argument("--tokens", type=int, default=200_000)
    run.add_argument("--minutes", type=float, default=30.0)
    run.add_argument("--actions", type=int, default=12)
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--force", action="store_true", help="discard an unresolved queue and start over")

    sub.add_parser("queue", help="show what is waiting for review")

    acc = sub.add_parser("accept", help="apply chosen results and record the labels")
    acc.add_argument("ids", nargs="*")
    acc.add_argument("--all", action="store_true")

    miss = sub.add_parser("miss", help="record that the queue offered nothing you wanted")
    miss.add_argument("requested", help="what you actually wanted next")

    sub.add_parser("stats", help="miss rate and learned confidences")

    sim = sub.add_parser(
        "simulate",
        help="drive the learning loop against a synthetic user to check convergence",
    )
    sim.add_argument("--sessions", type=int, default=500)
    sim.add_argument("--queue-size", type=int, default=2)
    sim.add_argument("--seed", type=int, default=5)

    args = parser.parse_args(argv)

    # simulate needs no repository -- it drives the learning loop in memory.
    if args.command == "simulate":
        return _simulate(args)

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

    existing = Queue.load(repo)
    if existing and existing.entries and not args.force:
        # Overwriting an unresolved queue would drop patches the human never
        # ruled on -- and, worse, never record the accept/reject labels, so the
        # session's learning silently vanishes. Make them resolve or discard it.
        print(
            f"a queue from '{existing.task}' is still unresolved "
            f"({len(existing.entries)} entries).\n"
            "resolve it with `runahead accept`/`miss`, or re-run with --force to discard it.",
            file=sys.stderr,
        )
        return 1

    # "The task you just finished" is, by default, your last commit -- so
    # `runahead run` with no argument just works in a clean repo.
    task = args.task or git(repo, "log", "-1", "--pretty=%s").strip()
    if not task:
        print("no task given and no commit to infer one from", file=sys.stderr)
        return 1
    args.task = task
    print(f"speculating after: {task}", file=sys.stderr)

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

    # Per-agent token accounting. Each action is one agent invocation; charge
    # its spend to its kind so `stats` can show what each speculation costs
    # over time. Recorded at run time, independent of accept/reject.
    for chain in done:
        for result in chain.results:
            store.record_tokens(task_kind, result.action.kind, result.tokens)
    store.flush()

    queue = Queue.build(args.task, task_kind, done)
    queue.save(repo)
    print(queue.render(budget.summary()))
    print()
    print(_token_report(done))
    return 0


def _token_report(done) -> str:
    rows = []
    for chain in done:
        for result in chain.results:
            rows.append((result.tokens, result.action.id))
    rows.sort(reverse=True)
    total = sum(t for t, _ in rows)
    lines = ["tokens per agent:"]
    for tokens, agent_id in rows:
        share = (tokens / total * 100) if total else 0.0
        lines.append(f"  {agent_id:<28} {tokens:>8,}  {share:4.0f}%")
    lines.append(f"  {'total':<28} {total:>8,}")
    return "\n".join(lines)


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


def _simulate(args) -> int:
    import tempfile

    from .simulate import render, simulate

    # Two reliably-wanted actions and four medium distractors, into a queue that
    # holds two. Early exploration keeps bumping a wanted action for a
    # distractor -- a miss -- until the posteriors sharpen and the two wanted
    # actions own both slots. This is the convergence claim, made falsifiable.
    prefs = {
        "write-tests": 0.90,
        "draft-commit": 0.85,
        "add-edge-cases": 0.48,
        "error-handling": 0.45,
        "update-docs": 0.40,
        "perf-pass": 0.35,
    }
    with tempfile.TemporaryDirectory() as home:
        result = simulate(
            prefs,
            sessions=args.sessions,
            queue_size=args.queue_size,
            home=Path(home),
            seed=args.seed,
        )
    print(render(result))
    print()
    print(
        "this validates the mechanism under a stated user model, not real humans. "
        "only real use settles that."
    )
    return 0


def _stats(store: Store) -> int:
    rate = store.miss_rate()
    print(f"miss rate: {'n/a' if rate is None else f'{rate:.0%}'}")
    print()

    ledger = store.token_ledger()
    rows = []
    for task_kind, action_kind in store.known_pairs():
        posterior = store.posterior(task_kind, action_kind)
        cost = ledger.get(f"{task_kind}|{action_kind}", {})
        rows.append((posterior.mean, f"{task_kind}|{action_kind}", posterior, cost))

    print(f"  {'action':<36} {'p':>4}  {'sd':>4}  {'n':>3}  {'avg tok':>8}  {'total tok':>10}")
    for mean, key, posterior, cost in sorted(rows, key=lambda r: r[0], reverse=True):
        avg = f"{cost['avg']:,.0f}" if cost.get("calls") else "-"
        total = f"{cost['tokens']:,}" if cost.get("calls") else "-"
        print(
            f"  {key:<36} {mean:>4.2f}  {posterior.stdev:>4.2f}  "
            f"{int(posterior.observations):>3}  {avg:>8}  {total:>10}"
        )

    grand = sum(c["tokens"] for c in ledger.values())
    if grand:
        print(f"\n  total spent across all speculation: {grand:,} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
