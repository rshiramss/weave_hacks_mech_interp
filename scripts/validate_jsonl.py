"""Validate a routing JSONL against the query contract + registry (Stage 1).

Checks every row: query_text passes validate_query(), and agent_id / tool_id
exist in registry.json (never invent ids). Exits non-zero on any failure.

    python scripts/validate_jsonl.py data/eval_holdout.jsonl
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.probe.query_contract import normalize_query, validate_query  # noqa: E402
from src.tools.factory import _registry  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/validate_jsonl.py <path.jsonl>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        return 1

    reg = _registry()
    agents = {a["agent_id"] for a in reg["agents"]}
    tools = {t["tool_id"] for t in reg["tools"]}

    errors = []
    n = 0
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        n += 1
        row = json.loads(line)
        ok, reason = validate_query(normalize_query(row.get("query_text", "")))
        if not ok:
            errors.append(f"row {i}: bad query_text ({reason})")
        if row.get("agent_id") not in agents:
            errors.append(f"row {i}: unknown agent_id {row.get('agent_id')!r}")
        if row.get("tool_id") not in tools:
            errors.append(f"row {i}: unknown tool_id {row.get('tool_id')!r}")

    if errors:
        print(f"[validate] {len(errors)} problem(s) in {n} rows:")
        for e in errors[:25]:
            print("  -", e)
        return 1
    print(f"[validate] OK — {n} rows, all queries valid, all ids in registry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
