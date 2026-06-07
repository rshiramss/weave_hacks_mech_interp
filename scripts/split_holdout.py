"""Derive data/eval_holdout.jsonl from the generated dataset (Stage 1).

Reads data/raw_dataset.jsonl (written by modal_app/generate_dataset.py, with a
`split` tag) and writes the test rows as {query_text, agent_id, tool_id}.

    python scripts/split_holdout.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw_dataset.jsonl"
OUT_PATH = ROOT / "data" / "eval_holdout.jsonl"
KEEP = ("query_text", "agent_id", "tool_id")


def main() -> int:
    if not RAW_PATH.exists():
        print(f"ERROR: {RAW_PATH} not found. Run "
              "`modal run modal_app/generate_dataset.py` first.", file=sys.stderr)
        return 1

    rows = [json.loads(line) for line in RAW_PATH.read_text().splitlines() if line.strip()]
    test = [r for r in rows if r.get("split") == "test"]
    if not test:  # generator without split tags -> fall back to a 10% tail slice
        n = max(20, len(rows) // 10)
        test = rows[-n:]
        print(f"[split] no 'test' split tags; using last {len(test)} rows as holdout")

    with OUT_PATH.open("w") as f:
        for r in test:
            f.write(json.dumps({k: r[k] for k in KEEP}) + "\n")
    print(f"[split] wrote {len(test)} holdout rows -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
