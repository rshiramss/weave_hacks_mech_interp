"""Concatenate the three query datasets into one, tagging each row's source.

Combines data/queries_{nl,mixed,descriptive}.json into data/queries_all.json,
adding a "source" field to every row so the origin of each query is traceable.

Run from the project root:
    python scripts/concat_queries.py
"""

import json
import os

SOURCES = {
    "nl": "data/queries_nl.json",
    "mixed": "data/queries_mixed.json",
    "descriptive": "data/queries_descriptive.json",
}
OUTPUT_PATH = "data/queries_all.json"


def main():
    combined = []
    counts = {}

    for source, path in SOURCES.items():
        if not os.path.exists(path):
            print(f"WARNING: {path} not found; skipping '{source}'.")
            counts[source] = 0
            continue
        with open(path) as f:
            rows = json.load(f)
        for row in rows:
            row["source"] = source
            combined.append(row)
        counts[source] = len(rows)
        print(f"{source:<12} {len(rows)} rows ({path})")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nGrand total: {len(combined)} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
