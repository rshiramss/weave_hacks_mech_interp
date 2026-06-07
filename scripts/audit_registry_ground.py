"""Audit + clean a labeled routing dataset for linear-probe training (Stage 1).

Implements DATA_CLEANING.md §5 Steps 1-2. The probe fails when the same
`query_text` carries different `tool_id` labels: identical text -> identical
hidden state `h` -> contradictory targets, which no linear classifier can fit.

This script:
  1. Reports total/unique rows, conflicting queries, and per-source/style impact.
  2. (optional) Writes a clean JSONL with every conflicting query removed.
  3. (optional) Quarantines the removed rows for inspection.
  4. (optional) Checks holdout overlap (leakage) against the training rows.

Accepts a JSON array (registry_ground.json) or a JSONL file.

    python scripts/audit_registry_ground.py registry_ground.json
    python scripts/audit_registry_ground.py registry_ground.json \
        --output data/registry_clean.jsonl \
        --quarantine data/quarantine/conflicting_labels.jsonl
    python scripts/audit_registry_ground.py registry_ground.json \
        --holdout data/eval_holdout.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "data" / "registry.json"

# How many example conflicts to print so the report stays readable on a big file.
MAX_CONFLICT_EXAMPLES = 8
MAX_TOOLS_PER_EXAMPLE = 6


def _load_rows(path: Path) -> list[dict]:
    """Load a JSON array or JSONL file into a list of row dicts."""
    text = path.read_text()
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _registry_ids() -> tuple[set[str], set[str]]:
    """(agent_ids, tool_ids) from the catalog — labels must never be invented."""
    reg = json.loads(REGISTRY_PATH.read_text())
    return (
        {a["agent_id"] for a in reg["agents"]},
        {t["tool_id"] for t in reg["tools"]},
    )


def _tools_per_query(rows: list[dict]) -> dict[str, set[str]]:
    by_query: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_query[row["query_text"]].add(row["tool_id"])
    return by_query


def _agents_per_query(rows: list[dict]) -> dict[str, set[str]]:
    by_query: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_query[row["query_text"]].add(row["agent_id"])
    return by_query


def _print_report(rows: list[dict], bad_queries: set[str]) -> None:
    tools_per_query = _tools_per_query(rows)
    agents_per_query = _agents_per_query(rows)
    bad_rows = [r for r in rows if r["query_text"] in bad_queries]
    agent_conflicts = {q for q, agents in agents_per_query.items() if len(agents) > 1}

    print(f"[audit] total rows           : {len(rows)}")
    print(f"[audit] unique query_text    : {len(tools_per_query)}")
    print(f"[audit] conflicting queries  : {len(bad_queries)} (>1 tool_id)")
    print(f"[audit] rows in conflict     : {len(bad_rows)} "
          f"({len(bad_rows) / len(rows):.1%})")
    print(f"[audit] agent-level conflicts: {len(agent_conflicts)} (>1 agent_id)")

    # Impact per source / style (where the noise concentrates — see §4).
    for field in ("source", "style"):
        if not any(field in r for r in rows):
            continue
        per_value: dict[str, int] = defaultdict(int)
        for row in bad_rows:
            per_value[row.get(field, "?")] += 1
        breakdown = ", ".join(f"{v}={n}" for v, n in sorted(per_value.items()))
        print(f"[audit] bad rows by {field:<7}: {breakdown or 'none'}")

    print("[audit] example conflicts:")
    for query in list(bad_queries)[:MAX_CONFLICT_EXAMPLES]:
        tools = sorted(tools_per_query[query])
        shown = ", ".join(tools[:MAX_TOOLS_PER_EXAMPLE])
        more = f" (+{len(tools) - MAX_TOOLS_PER_EXAMPLE} more)" \
            if len(tools) > MAX_TOOLS_PER_EXAMPLE else ""
        print(f'  - "{query[:70]}" -> {shown}{more}')


def _check_registry(rows: list[dict]) -> int:
    """Count rows whose agent_id / tool_id are not in the catalog."""
    agents, tools = _registry_ids()
    bad = sum(
        1 for r in rows
        if r.get("agent_id") not in agents or r.get("tool_id") not in tools
    )
    if bad:
        print(f"[audit] WARNING: {bad} rows have ids not in registry.json")
    else:
        print("[audit] registry check     : all ids valid")
    return bad


def _report_holdout_overlap(rows: list[dict], bad_queries: set[str],
                            holdout_path: Path) -> None:
    train_queries = {r["query_text"] for r in rows if r["query_text"] not in bad_queries}
    holdout_rows = _load_rows(holdout_path)
    holdout_queries = {r["query_text"] for r in holdout_rows}
    leaked = holdout_queries & train_queries
    ambiguous = holdout_queries & bad_queries
    print(f"[audit] holdout rows         : {len(holdout_rows)}")
    print(f"[audit] holdout leaking train: {len(leaked)} (must be 0)")
    print(f"[audit] holdout ambiguous    : {len(ambiguous)} (conflicting query_text)")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="registry_ground.json or any labeled JSONL")
    parser.add_argument("--output", help="write conflict-free rows to this JSONL")
    parser.add_argument("--quarantine", help="write removed conflicting rows here")
    parser.add_argument("--holdout", help="report holdout overlap / leakage")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: {dataset_path} not found", file=sys.stderr)
        return 1

    rows = _load_rows(dataset_path)
    if not rows:
        print(f"ERROR: {dataset_path} has no rows", file=sys.stderr)
        return 1

    tools_per_query = _tools_per_query(rows)
    bad_queries = {q for q, tools in tools_per_query.items() if len(tools) > 1}

    _print_report(rows, bad_queries)
    _check_registry(rows)
    if args.holdout:
        _report_holdout_overlap(rows, bad_queries, Path(args.holdout))

    if args.quarantine:
        quarantined = [r for r in rows if r["query_text"] in bad_queries]
        _write_jsonl(Path(args.quarantine), quarantined)
        print(f"[audit] quarantined {len(quarantined)} rows -> {args.quarantine}")

    if args.output:
        clean = [r for r in rows if r["query_text"] not in bad_queries]
        _write_jsonl(Path(args.output), clean)
        print(f"[audit] wrote {len(clean)} clean rows -> {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
