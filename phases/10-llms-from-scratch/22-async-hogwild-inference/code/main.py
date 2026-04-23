import random
import threading
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class CacheEntry:
    owner: str
    position: int
    token: str
    round_committed: int


class SharedCache:
    def __init__(self, prompt_tokens, worker_ids):
        self._lock = threading.Lock()
        self._entries = []
        for i, t in enumerate(prompt_tokens):
            self._entries.append(CacheEntry(owner="prompt", position=i, token=t, round_committed=-1))
        self._prompt_len = len(prompt_tokens)
        self._next_pos = {wid: self._prompt_len + i * 10_000 for i, wid in enumerate(worker_ids)}
        self._worker_tokens = {wid: [] for wid in worker_ids}

    def snapshot(self):
        with self._lock:
            return list(self._entries)

    def worker_text(self, worker_id):
        with self._lock:
            return " ".join(e.token for e in self._entries if e.owner == worker_id)

    def append(self, worker_id, token, current_round):
        with self._lock:
            pos = self._next_pos[worker_id]
            self._next_pos[worker_id] += 1
            entry = CacheEntry(owner=worker_id, position=pos, token=token, round_committed=current_round)
            self._entries.append(entry)
            self._worker_tokens[worker_id].append(token)
            return entry

    def other_workers_text(self, worker_id):
        with self._lock:
            parts = []
            for wid, toks in self._worker_tokens.items():
                if wid == worker_id or wid == "prompt":
                    continue
                if toks:
                    parts.append(f"{wid}: {' '.join(toks)}")
            return " | ".join(parts)


@dataclass
class Plan:
    subtask_id: str
    keyword: str
    tokens: list
    answered: bool = False
    cursor: int = 0
    rerouted: bool = False
    reroute_log: list = field(default_factory=list)


class Worker:
    def __init__(self, worker_id, primary_plan, backup_plans, cache):
        self.worker_id = worker_id
        self.active_plan = primary_plan
        self.backup_plans = backup_plans
        self.cache = cache
        self.done = False

    def _another_worker_covered(self, subtask_keyword):
        seen = self.cache.other_workers_text(self.worker_id)
        return subtask_keyword.lower() in seen.lower()

    def _choose_next_plan(self):
        for p in self.backup_plans:
            if p.answered:
                continue
            if self._another_worker_covered(p.keyword):
                p.answered = True
                continue
            return p
        return None

    def step(self, current_round):
        if self.done:
            return None
        plan = self.active_plan
        if plan.cursor >= len(plan.tokens):
            plan.answered = True
            next_plan = self._choose_next_plan()
            if next_plan is None:
                self.done = True
                return None
            self.active_plan = next_plan
            plan = next_plan

        if plan.cursor == 0 and self._another_worker_covered(plan.keyword):
            plan.answered = True
            plan.rerouted = True
            plan.reroute_log.append(f"round {current_round}: {plan.subtask_id} already covered, rerouting")
            next_plan = self._choose_next_plan()
            if next_plan is None:
                self.done = True
                return None
            self.active_plan = next_plan
            plan = next_plan

        token = plan.tokens[plan.cursor]
        plan.cursor += 1
        entry = self.cache.append(self.worker_id, token, current_round)
        if plan.cursor >= len(plan.tokens):
            plan.answered = True
        return entry


class HogwildScheduler:
    def __init__(self, workers, cache, seed=0):
        self.workers = workers
        self.cache = cache
        self.rng = random.Random(seed)
        self.history = []

    def run(self, max_rounds=40, on_round: Callable = None):
        for r in range(max_rounds):
            order = list(self.workers)
            self.rng.shuffle(order)
            round_commits = []
            for w in order:
                entry = w.step(r)
                if entry is not None:
                    round_commits.append(entry)
            self.history.append(round_commits)
            if on_round is not None:
                on_round(r, round_commits)
            if all(w.done for w in self.workers):
                break
        return self.history


SUBTASKS = {
    "arithmetic": "391",
    "spell": "suonorhcnysa",
    "geography": "ulaanbaatar",
}

SCRIPTS = {
    ("alice", "arithmetic"): "[A] 17 times 23 equals 391 .",
    ("alice", "spell"): "[A] reversed ' asynchronous ' is suonorhcnysa .",
    ("alice", "geography"): "[A] capital of mongolia is ulaanbaatar .",
    ("bob", "spell"): "[B] asynchronous reversed is suonorhcnysa .",
    ("bob", "geography"): "[B] mongolia capital ulaanbaatar .",
    ("bob", "arithmetic"): "[B] 17 * 23 = 391 .",
    ("carol", "geography"): "[C] mongolia -> ulaanbaatar .",
    ("carol", "arithmetic"): "[C] 17x23 is 391 .",
    ("carol", "spell"): "[C] backward asynchronous is suonorhcnysa .",
}


def _make_plan(worker_id, subtask):
    return Plan(subtask_id=subtask, keyword=SUBTASKS[subtask], tokens=SCRIPTS[(worker_id, subtask)].split())


def build_task():
    prompt = "solve three problems in parallel : arithmetic spell geography".split()
    worker_ids = ["alice", "bob", "carol"]
    cache = SharedCache(prompt_tokens=prompt, worker_ids=worker_ids)
    primaries = {"alice": "arithmetic", "bob": "spell", "carol": "geography"}
    order = ["arithmetic", "spell", "geography"]
    workers = []
    for wid in worker_ids:
        primary = _make_plan(wid, primaries[wid])
        backups = [_make_plan(wid, s) for s in order if s != primaries[wid]]
        workers.append(Worker(wid, primary, backups, cache))
    return cache, workers


def print_round(round_idx, commits):
    if not commits:
        print(f"round {round_idx:02d}: (no commits)")
        return
    parts = [f"{e.owner}:{e.token}" for e in commits]
    print(f"round {round_idx:02d}: " + "  |  ".join(parts))


def main():
    cache, workers = build_task()
    scheduler = HogwildScheduler(workers, cache, seed=42)
    print("=== hogwild simulation: 3 workers, shared KV cache ===")
    print("prompt:", " ".join(e.token for e in cache.snapshot() if e.owner == "prompt"))
    print()
    scheduler.run(max_rounds=25, on_round=print_round)
    print("\n=== per-worker final transcripts ===")
    for w in workers:
        print(f"{w.worker_id}: {cache.worker_text(w.worker_id)}")
    print("\n=== reroutes (self-check fired) ===")
    logs = [(w.worker_id, e) for w in workers for p in [w.active_plan] + w.backup_plans for e in p.reroute_log]
    if logs:
        for wid, e in logs:
            print(f"  {wid} :: {e}")
    else:
        print("  (none this seed; try another)")


if __name__ == "__main__":
    main()
