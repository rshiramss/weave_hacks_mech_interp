"""Publish the routing holdout as a Weave Dataset (Stage 6, §7.2).

Loads data/eval_holdout.jsonl ({query_text, agent_id, tool_id} per line) and
publishes it as weave.Dataset(name="soc-routing-holdout").

Usage:
    python scripts/publish_holdout.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from src.weave_setup import init_weave  # noqa: E402

DATASET_NAME = "soc-routing-holdout"
HOLDOUT_PATH = Path(__file__).resolve().parents[1] / "data" / "eval_holdout.jsonl"
REQUIRED = ("query_text", "agent_id", "tool_id")


def load_rows() -> list[dict]:
    if not HOLDOUT_PATH.exists():
        raise FileNotFoundError(
            f"{HOLDOUT_PATH} not found. Create the holdout (Stage 1) first."
        )
    rows = []
    for line in HOLDOUT_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        missing = [k for k in REQUIRED if k not in row]
        if missing:
            raise ValueError(f"Holdout row missing {missing}: {row}")
        rows.append({k: row[k] for k in REQUIRED})
    return rows


def main() -> int:
    load_dotenv()
    init_weave()

    import weave

    rows = load_rows()
    dataset = weave.Dataset(name=DATASET_NAME, rows=rows)
    ref = weave.publish(dataset)
    print(f"[publish] published '{DATASET_NAME}' with {len(rows)} rows")
    print(f"[publish] ref: {ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
