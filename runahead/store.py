"""Two-layer local storage.

    ~/.runahead/       user-scoped, permanent, never committed, never uploaded
    .git/runahead/     repo-scoped, disposable

Learning data must not live in the repo. Committed statistics get averaged
across teammates, and an averaged habit predicts nobody's.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from .beta import Beta

PRIOR_STRENGTH = 2.0
"""Weight of the global habit when seeding a fresh repo.

kappa=2 means a new repo starts at Beta(1 + 2*g, 1 + 2*(1-g)). With no global
data at all that is Beta(2, 2): mean 0.5, stdev 0.22 — uncertain enough that
Thompson sampling still explores.
"""


def _key(task_kind: str, action_kind: str) -> str:
    return f"{task_kind}|{action_kind}"


def _counts(raw: dict, key: str) -> tuple[float, float]:
    entry = raw.get(key)
    if not entry:
        return 0.0, 0.0
    return float(entry.get("accepted", 0)), float(entry.get("rejected", 0))


@dataclass
class Store:
    """Hierarchical Beta counters: global prior, per-repo posterior."""

    root: Path
    repo_id: str
    _global: dict = field(default_factory=dict)
    _repo: dict = field(default_factory=dict)
    _tokens: dict = field(default_factory=dict)

    @classmethod
    def open(cls, repo_path: Path, root: Path | None = None) -> "Store":
        root = root or Path(os.environ.get("RUNAHEAD_HOME", Path.home() / ".runahead"))
        repo_id = hashlib.sha256(str(repo_path.resolve()).encode()).hexdigest()[:16]
        store = cls(root=root, repo_id=repo_id)
        store._global = _read_json(store.global_path)
        store._repo = _read_json(store.repo_path)
        store._tokens = _read_json(store.tokens_path)
        return store

    @property
    def global_path(self) -> Path:
        return self.root / "priors.json"

    @property
    def repo_path(self) -> Path:
        return self.root / "repos" / f"{self.repo_id}.json"

    @property
    def tokens_path(self) -> Path:
        return self.root / "tokens.json"

    @property
    def history_path(self) -> Path:
        return self.root / "history.jsonl"

    def known_pairs(self) -> list[tuple[str, str]]:
        """Every (task_kind, action_kind) either layer has ever seen."""
        keys = set(self._global) | set(self._repo)
        return sorted(tuple(k.split("|", 1)) for k in keys if "|" in k)

    def global_mean(self, task_kind: str, action_kind: str) -> float:
        accepted, rejected = _counts(self._global, _key(task_kind, action_kind))
        return (accepted + 1.0) / (accepted + rejected + 2.0)

    def posterior(self, task_kind: str, action_kind: str) -> Beta:
        """Repo observations layered on a prior derived from global habit."""
        g = self.global_mean(task_kind, action_kind)
        prior_alpha = 1.0 + PRIOR_STRENGTH * g
        prior_beta = 1.0 + PRIOR_STRENGTH * (1.0 - g)

        accepted, rejected = _counts(self._repo, _key(task_kind, action_kind))
        return Beta(prior_alpha + accepted, prior_beta + rejected)

    def record(self, task_kind: str, action_kind: str, accepted: bool) -> None:
        for raw in (self._global, self._repo):
            entry = raw.setdefault(_key(task_kind, action_kind), {"accepted": 0, "rejected": 0})
            entry["accepted" if accepted else "rejected"] += 1

    def record_tokens(self, task_kind: str, action_kind: str, tokens: int, cost_usd: float = 0.0) -> None:
        """Charge an agent's spend to its action kind.

        Spend is per-agent and happens at run time, accepted or not -- so this
        is recorded when the action runs, independent of the accept/reject
        label. It answers 'which speculations are worth what they cost me'.

        Tokens and cost are tracked separately because they do not track each
        other: cache reads dominate the token count but cost almost nothing, so
        a report that shows only one hides half the picture.
        """
        entry = self._tokens.setdefault(
            _key(task_kind, action_kind), {"calls": 0, "tokens": 0, "cost_usd": 0.0}
        )
        entry["calls"] += 1
        entry["tokens"] += int(tokens)
        entry["cost_usd"] = entry.get("cost_usd", 0.0) + float(cost_usd)

    def token_ledger(self) -> dict[str, dict]:
        """{ 'task|action': {calls, tokens, avg, cost_usd} } over the store's life."""
        out: dict[str, dict] = {}
        for key, entry in self._tokens.items():
            calls = entry.get("calls", 0) or 0
            tokens = entry.get("tokens", 0) or 0
            out[key] = {
                "calls": calls,
                "tokens": tokens,
                "avg": (tokens / calls) if calls else 0.0,
                "cost_usd": entry.get("cost_usd", 0.0) or 0.0,
            }
        return out

    def record_miss(self, task_kind: str, requested: str) -> None:
        """The human ignored the queue entirely and asked for something else.

        The single most informative signal we get: it names an action the
        predictor failed to imagine. Never derivable from accept/reject alone.
        """
        self.append_history({"event": "miss", "task_kind": task_kind, "requested": requested})

    def record_conflict(self, first: str, second: str) -> None:
        """Two actions accepted together whose patches would not compose.

        Not a bug. A label: do not propose this pair together again.
        """
        self.append_history({"event": "conflict", "actions": sorted([first, second])})

    def append_history(self, event: dict) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(event)
        payload.setdefault("at", int(time.time()))
        with self.history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def miss_rate(self, window: int = 50) -> float | None:
        """The metric that gates depth. Not the accept rate.

        Accept rate is measured only over what we chose to show; it climbs as
        the system narrows onto its own habits. Miss rate cannot be gamed that
        way -- it counts the times the human wanted something we never offered.
        """
        events = [e for e in self._history() if e.get("event") in ("miss", "queue_resolved")]
        if not events:
            return None
        recent = events[-window:]
        misses = sum(1 for e in recent if e["event"] == "miss")
        return misses / len(recent)

    def _history(self) -> list[dict]:
        if not self.history_path.exists():
            return []
        out = []
        for line in self.history_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def flush(self) -> None:
        _write_json(self.global_path, self._global)
        _write_json(self.repo_path, self._repo)
        _write_json(self.tokens_path, self._tokens)


def _read_json(path: Path) -> dict:
    """Load a JSON object, degrading to {} on anything unreadable.

    A missing, empty, whitespace-only, truncated, or hand-mangled file must
    never crash the caller -- losing corrupt history is acceptable, aborting
    `stats` or `run` is not. A non-object payload (list, scalar, null) is
    treated as corrupt for the same reason: the counters that consume it
    assume a dict. The exists() check is skipped deliberately so a file
    deleted between check and read just falls through to {}.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = json.loads(text or "{}")
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict) -> None:
    """Persist JSON atomically: fully materialise a temp file, fsync it, then
    rename it over the target. rename(2) is atomic, so a crash mid-write leaves
    a reader with either the old complete file or the new one -- never a torn
    file that fails to parse on reload. A per-pid temp name keeps concurrent
    writers from clobbering each other's scratch file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
