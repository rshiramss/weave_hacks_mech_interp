"""Clean the query datasets and emit cache index mappings.

The hidden-state cache .npy files are indexed by row position, so whenever we
drop rows from a query file we also write a list of the ORIGINAL row indices
that survived. Those index files let us slice the .npy caches back into
alignment with the cleaned query files.

Filters:
  - queries_nl.json:          dedupe query_text (keep first) + drop <5 words
  - queries_mixed.json:       dedupe query_text (keep first)
  - queries_descriptive.json: untouched (already clean)

No dependencies beyond json + os. Run from the project root:
    python scripts/clean_queries.py
"""

import json
import os

DATA_DIR = "data"

NL_PATH = os.path.join(DATA_DIR, "queries_nl.json")
MIXED_PATH = os.path.join(DATA_DIR, "queries_mixed.json")
DESCRIPTIVE_PATH = os.path.join(DATA_DIR, "queries_descriptive.json")

NL_INDICES_PATH = os.path.join(DATA_DIR, "clean_indices_nl.json")
MIXED_INDICES_PATH = os.path.join(DATA_DIR, "clean_indices_mixed.json")

MIN_WORDS = 5


def load(path):
    with open(path) as f:
        return json.load(f)


def save(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def clean(rows, min_words=None):
    """Filter rows, returning (kept_rows, kept_original_indices).

    Always dedupes on query_text (keeping the first occurrence). If min_words
    is given, also drops queries with fewer than min_words whitespace tokens.
    """
    kept_rows = []
    kept_indices = []
    seen = set()
    for i, row in enumerate(rows):
        text = row["query_text"]
        if text in seen:
            continue
        if min_words is not None and len(text.split()) < min_words:
            continue
        seen.add(text)
        kept_rows.append(row)
        kept_indices.append(i)
    return kept_rows, kept_indices


def report(name, before, after):
    removed = before - after
    print(f"{name}: {before} -> {after}  (removed {removed})")


def main():
    print("Cleaning query datasets...\n")

    # queries_nl.json: dedupe + min word count
    nl_rows = load(NL_PATH)
    nl_clean, nl_indices = clean(nl_rows, min_words=MIN_WORDS)
    report("queries_nl.json", len(nl_rows), len(nl_clean))
    save(NL_PATH, nl_clean)
    save(NL_INDICES_PATH, nl_indices)
    print(f"  wrote {NL_PATH}")
    print(f"  wrote {NL_INDICES_PATH} ({len(nl_indices)} kept indices)\n")

    # queries_mixed.json: dedupe only
    mixed_rows = load(MIXED_PATH)
    mixed_clean, mixed_indices = clean(mixed_rows)
    report("queries_mixed.json", len(mixed_rows), len(mixed_clean))
    save(MIXED_PATH, mixed_clean)
    save(MIXED_INDICES_PATH, mixed_indices)
    print(f"  wrote {MIXED_PATH}")
    print(f"  wrote {MIXED_INDICES_PATH} ({len(mixed_indices)} kept indices)\n")

    # queries_descriptive.json: untouched
    desc_rows = load(DESCRIPTIVE_PATH)
    report("queries_descriptive.json", len(desc_rows), len(desc_rows))
    print("  (untouched)\n")

    print("Done.")


if __name__ == "__main__":
    main()
