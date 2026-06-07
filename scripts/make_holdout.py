"""Carve a leak-free eval holdout from the cleaned dataset (Stage 1 / §5 Step 5).

Splits on `query_text` (not row index): every holdout query is removed from the
train file, so no phrase straddles both. Stratifies across agents and spreads
across each agent's tools so the holdout is not dominated by one tool.

Inputs/outputs:
  --input       cleaned labeled JSONL (default data/registry_clean.jsonl)
  --holdout     -> data/eval_holdout.jsonl  ({query_text, agent_id, tool_id})
  --train-out   -> data/registry_train.jsonl (clean rows minus holdout queries)

The train file (not the raw clean file) is what activation extraction should run
on, guaranteeing the probe never sees a holdout query.

    python scripts/make_holdout.py
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_SEED = 1337
DEFAULT_PER_AGENT = 10  # 5 agents -> 50-row holdout (DATA_CLEANING gate: >= 50)
KEEP = ("query_text", "agent_id", "tool_id")


def _load_rows(path: Path) -> list[dict]:
    text = path.read_text()
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _drop_ambiguous(rows: list[dict]) -> list[dict]:
    """Defensive: remove any query_text that still maps to >1 tool_id."""
    tools_per_query: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        tools_per_query[row["query_text"]].add(row["tool_id"])
    bad = {q for q, tools in tools_per_query.items() if len(tools) > 1}
    return [r for r in rows if r["query_text"] not in bad]


def _pick_for_agent(rows: list[dict], quota: int, rng: random.Random) -> list[dict]:
    """Round-robin distinct query_text across the agent's tools for diversity."""
    by_tool: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_tool[row["tool_id"]].append(row)

    tool_ids = list(by_tool.values())
    for bucket in tool_ids:
        rng.shuffle(bucket)
    rng.shuffle(tool_ids)

    picked: list[dict] = []
    seen_queries: set[str] = set()
    exhausted = False
    while len(picked) < quota and not exhausted:
        exhausted = True
        for bucket in tool_ids:
            if not bucket:
                continue
            exhausted = False
            row = bucket.pop()
            if row["query_text"] in seen_queries:
                continue
            seen_queries.add(row["query_text"])
            picked.append(row)
            if len(picked) >= quota:
                break
    return picked


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/registry_clean.jsonl")
    parser.add_argument("--holdout", default="data/eval_holdout.jsonl")
    parser.add_argument("--train-out", default="data/registry_train.jsonl")
    parser.add_argument("--per-agent", type=int, default=DEFAULT_PER_AGENT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found", file=sys.stderr)
        return 1

    rows = _drop_ambiguous(_load_rows(input_path))
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_agent[row["agent_id"]].append(row)

    rng = random.Random(args.seed)
    holdout: list[dict] = []
    for agent_id in sorted(by_agent):
        holdout.extend(_pick_for_agent(by_agent[agent_id], args.per_agent, rng))

    holdout_queries = {r["query_text"] for r in holdout}
    train = [r for r in rows if r["query_text"] not in holdout_queries]

    # Verify leak-free + unambiguous before writing.
    train_queries = {r["query_text"] for r in train}
    leak = holdout_queries & train_queries
    if leak:
        print(f"ERROR: {len(leak)} holdout queries leaked into train", file=sys.stderr)
        return 1

    _write_jsonl(Path(args.holdout), [{k: r[k] for k in KEEP} for r in holdout])
    _write_jsonl(Path(args.train_out), train)

    per_agent_counts = defaultdict(int)
    for r in holdout:
        per_agent_counts[r["agent_id"]] += 1
    print(f"[holdout] {len(holdout)} rows ({len(holdout_queries)} unique queries), "
          f"0 leak, per-agent={dict(sorted(per_agent_counts.items()))}")
    print(f"[holdout] wrote {args.holdout}")
    print(f"[holdout] wrote {args.train_out} ({len(train)} train rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
