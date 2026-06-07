"""Label-uniqueness gate for probe training data (Stage 1).

Implements the DATA_CLEANING.md §4 gate: every `query_text` must map to exactly
one `(agent_id, tool_id)` pair. A query that carries two different labels gives
the probe identical hidden states with contradictory targets — unlearnable.

Exit 0 only when the bijection holds; exit 1 (with examples) otherwise. Wire this
into the pipeline before extracting activations.

    python scripts/check_label_uniqueness.py data/raw_dataset.jsonl
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

MAX_VIOLATIONS_SHOWN = 15


def _load_rows(path: Path) -> list[dict]:
    """Load a JSON array or JSONL file into a list of row dicts."""
    text = path.read_text()
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/check_label_uniqueness.py <path>",
              file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        return 1

    rows = _load_rows(path)
    labels_per_query: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        labels_per_query[row["query_text"]].add(
            (row.get("agent_id"), row.get("tool_id"))
        )

    violations = {q: labels for q, labels in labels_per_query.items()
                  if len(labels) > 1}

    if violations:
        print(f"[uniqueness] FAIL — {len(violations)} of {len(labels_per_query)} "
              f"unique queries map to multiple labels:")
        for query, labels in list(violations.items())[:MAX_VIOLATIONS_SHOWN]:
            pretty = ", ".join(f"{a}/{t}" for a, t in sorted(labels))
            print(f'  - "{query[:70]}" -> {pretty}')
        return 1

    print(f"[uniqueness] OK — {len(labels_per_query)} unique queries, "
          "each maps to exactly one (agent_id, tool_id)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
