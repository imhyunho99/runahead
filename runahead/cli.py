from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

from . import workspace
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
    run.add_argument("--tokens", type=int, default=3_000_000)
    run.add_argument("--minutes", type=float, default=30.0)
    run.add_argument("--actions", type=int, default=12)
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--force", action="store_true", help="discard an unresolved queue and start over")
    run.add_argument(
        "--within",
        type=float,
        default=14.0,
        help="workspace mode: only speculate in repos committed within N days (default 14)",
    )
    run.add_argument(
        "--repos",
        default=None,
        help="workspace mode: comma-separated repo names to run, ignoring the time window",
    )

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

    cwd = Path.cwd()

    # run/queue work at a workspace root too: a directory that is not itself a
    # git repo but holds sibling repos. Everything else stays single-repo.
    if args.command in ("run", "queue") and not workspace.is_git_repo(cwd):
        repos = workspace.find_repos(cwd)
        if repos:
            return _run_workspace(cwd, args) if args.command == "run" else _queue_workspace(cwd)
        # fall through to repo_root, which gives the normal "not a repo" error

    repo = repo_root(cwd)
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


def _run(repo: Path, store: Store, args, executor=None) -> int:
    budget = Budget(tokens=args.tokens, seconds=args.minutes * 60, actions=args.actions).start()
    queue, code = _speculate(repo, store, args, budget, executor)
    if code or queue is None:
        return code
    print(queue.render(budget.summary()))
    print()
    print(_token_report_from_queue(queue))
    return 0


def _speculate(repo: Path, store: Store, args, budget: Budget, executor=None):
    """Run one repo's speculation into `budget`. Returns (Queue|None, exit_code).

    A non-zero code means stop (dirty tree, unresolved queue). A None queue with
    code 0 means "ran, nothing to show". Shared by single-repo and workspace
    runs -- the workspace passes one budget through every repo in turn.
    """
    if git(repo, "status", "--porcelain").strip():
        print(f"{repo.name}: working tree is dirty; commit the finished task first", file=sys.stderr)
        return None, 1

    existing = Queue.load(repo)
    if existing and existing.entries and not args.force:
        # Overwriting an unresolved queue would drop patches the human never
        # ruled on -- and, worse, never record the accept/reject labels, so the
        # session's learning silently vanishes. Make them resolve or discard it.
        print(
            f"{repo.name}: a queue from '{existing.task}' is still unresolved "
            f"({len(existing.entries)} entries).\n"
            "resolve it with `runahead accept`/`miss`, or re-run with --force to discard it.",
            file=sys.stderr,
        )
        return None, 1

    # "The task you just finished" is, by default, your last commit -- so
    # `runahead run` with no argument just works in a clean repo.
    task = args.task or git(repo, "log", "-1", "--pretty=%s").strip()
    if not task:
        print(f"{repo.name}: no task given and no commit to infer one from", file=sys.stderr)
        return None, 1
    print(f"speculating after: {task}", file=sys.stderr)

    predictor = ClaudePredictor()
    task_kind = predictor.classify(task)
    diff = git(repo, "show", "--patch", "--stat", "HEAD")

    rng = random.Random(args.seed)
    chains = plan(store, predictor, task, task_kind, diff, rng)
    if not chains:
        print(f"{repo.name}: no candidates above the proposal floor")
        return None, 0

    print(f"{repo.name}: speculating on {len(chains)} chains ({task_kind})", file=sys.stderr)
    actions_before = budget.actions_used
    done = execute(repo, ClaudeExecutor(), chains, budget)

    # Per repo, since the budget's action counter is shared across the workspace.
    barren = (budget.actions_used - actions_before) - sum(len(c.results) for c in done)
    if barren > 0:
        # An agent that edits nothing still costs tokens and still reports
        # success. Never let that pass silently.
        print(f"{repo.name}: {barren} action(s) produced no patch", file=sys.stderr)

    # Per-agent token accounting. Each action is one agent invocation; charge
    # its spend to its kind so `stats` can show what each speculation costs
    # over time. Recorded at run time, independent of accept/reject.
    for chain in done:
        for result in chain.results:
            store.record_tokens(task_kind, result.action.kind, result.tokens, result.cost_usd)
    store.flush()

    queue = Queue.build(task, task_kind, done)
    queue.save(repo)
    return queue, 0


def _run_workspace(root: Path, args, executor=None) -> int:
    only = [name.strip() for name in args.repos.split(",")] if args.repos else None
    now = int(time.time())
    active, skipped = workspace.active_repos(root, now, args.within, only)

    if not active:
        print(f"no repos to speculate in under {root}", file=sys.stderr)
        for a in skipped:
            print(f"  skipped {a.name:<24} last commit {workspace.age(now, a.last_commit_at)}", file=sys.stderr)
        print("pin repos with --repos a,b or widen the window with --within DAYS", file=sys.stderr)
        return 1

    print("detected recent activity:", file=sys.stderr)
    for a in active:
        print(f"  {a.name:<24} {workspace.age(now, a.last_commit_at):>9}  {a.subject}", file=sys.stderr)
    for a in skipped:
        print(f"  {a.name:<24} {workspace.age(now, a.last_commit_at):>9}  (skipped)", file=sys.stderr)
    print(file=sys.stderr)

    # One budget for the whole workspace, spent newest-repo-first. tokens,
    # minutes and actions are workspace totals -- the repo you touched most
    # recently gets first claim on what you were willing to spend while away.
    budget = Budget(tokens=args.tokens, seconds=args.minutes * 60, actions=args.actions).start()

    sections: list[str] = []
    accept_lines: list[str] = []
    for a in active:
        if budget.exhausted():
            print(f"{a.name}: budget spent before reaching this repo; skipped", file=sys.stderr)
            continue
        store = Store.open(a.path)
        # In a workspace the task must come from each repo's own last commit,
        # never a single --task shared across all of them.
        per_repo_args = _clone_args(args, task=None)
        queue, code = _speculate(a.path, store, per_repo_args, budget, executor)
        if queue is None:
            continue
        sections.append(f"[{a.name}]\n" + _indent(queue.render()))
        ids = " ".join(e.id for e in queue.entries if not e.auto_accept)
        if ids:
            accept_lines.append(f"  cd {a.name} && runahead accept {ids}")

    if not sections:
        print("nothing speculated.")
        return 0

    print("\n\n".join(sections))
    print(f"\nworkspace spend: {budget.summary()}")
    if accept_lines:
        print("\nto apply, review each repo and accept there:")
        print("\n".join(accept_lines))
    return 0


def _queue_workspace(root: Path) -> int:
    shown = False
    for repo in workspace.find_repos(root):
        queue = Queue.load(repo)
        if queue and queue.entries:
            print(f"[{repo.name}]")
            print(_indent(queue.render()))
            print()
            shown = True
    if not shown:
        print("no unresolved queues across the workspace")
    return 0


def _clone_args(args, **overrides):
    import copy

    clone = copy.copy(args)
    for key, value in overrides.items():
        setattr(clone, key, value)
    return clone


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line if line else line for line in text.splitlines())


def _token_report_from_queue(queue: Queue) -> str:
    rows = []
    for entry in queue.entries:
        for step in entry.steps:
            rows.append((step.tokens, step.cost_usd, step.id))
    rows.sort(reverse=True)
    total = sum(t for t, _, _ in rows)
    total_cost = sum(c for _, c, _ in rows)
    lines = ["tokens per agent:"]
    for tokens, cost, agent_id in rows:
        share = (tokens / total * 100) if total else 0.0
        cost_str = f"  ${cost:.2f}" if cost else ""
        lines.append(f"  {agent_id:<28} {tokens:>9,}  {share:4.0f}%{cost_str}")
    tail = f"  ${total_cost:.2f}" if total_cost else ""
    lines.append(f"  {'total':<28} {total:>9,}{tail}")
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

    print(f"  {'action':<34} {'p':>4}  {'sd':>4}  {'n':>3}  {'avg tok':>9}  {'total tok':>11}  {'$':>7}")
    for mean, key, posterior, cost in sorted(rows, key=lambda r: r[0], reverse=True):
        avg = f"{cost['avg']:,.0f}" if cost.get("calls") else "-"
        total = f"{cost['tokens']:,}" if cost.get("calls") else "-"
        usd = f"${cost['cost_usd']:.2f}" if cost.get("cost_usd") else "-"
        print(
            f"  {key:<34} {mean:>4.2f}  {posterior.stdev:>4.2f}  "
            f"{int(posterior.observations):>3}  {avg:>9}  {total:>11}  {usd:>7}"
        )

    grand = sum(c["tokens"] for c in ledger.values())
    grand_cost = sum(c["cost_usd"] for c in ledger.values())
    if grand:
        tail = f" (${grand_cost:.2f})" if grand_cost else ""
        print(f"\n  total spent across all speculation: {grand:,} tokens{tail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
