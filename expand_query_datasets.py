#!/usr/bin/env python3
"""Append more queries to existing chunk files (does NOT regenerate from scratch).

- Adds +20 queries/tool to each agent's NL and mixed chunks (10 -> 30).
- NL additions: all style "nl" (fresh phrasings).
- Mixed additions: 70% descriptive / 30% nl (14 desc + 6 nl per tool).
- Re-merges nl + mixed final datasets.
- Generates a fresh 100% descriptive dataset (30/tool) in *_descriptive.json chunks.

Cache invalidation in data/probe_cache/ is handled elsewhere; this script never
touches it.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import generate_query_datasets as g

ADD_PER_TOOL = 20
TARGET_PER_TOOL = 30
MIXED_ADD_DESC = 14   # 70% of 20
MIXED_ADD_NL = 6      # 30% of 20
DESC_FILE_PER_TOOL = 30

ROOT = g.ROOT
CHUNK_DIR = g.CHUNK_DIR
AGENT_ORDER = g.AGENT_ORDER


# ── fresh phrasing templates (distinct from generate_query_datasets.py) ──────
def new_nl_patterns(tool: dict, topic: str) -> list:
    P, U, H, I, W = g.pick, g.USERS, g.HOSTS, g.IPS, g.WHEN
    agent = tool["agent_id"]
    pats = [
        lambda i: f"can someone look into {topic} on {P(H, i)}",
        lambda i: f"wondering if {P(U, i)} had any {topic} {P(W, i)}",
        lambda i: f"mind pulling {topic} for {P(H, i)}",
        lambda i: f"anything weird with {topic} on {P(H, i)} {P(W, i)}",
        lambda i: f"{P(U, i)} flagged {topic} on {P(H, i)}, take a look",
        lambda i: f"do we see {topic} for {P(I, i)}",
        lambda i: f"give me {topic} on {P(H, i)} {P(W, i)}",
        lambda i: f"checking {topic} for {P(U, i)}, what do you see",
        lambda i: f"real quick {topic} on {P(H, i)}",
        lambda i: f"follow up on {topic} for {P(U, i)} {P(W, i)}",
        lambda i: f"loop back on {topic} from {P(I, i)}",
        lambda i: f"need eyes on {topic} for {P(H, i)}",
        lambda i: f"whats the {topic} look like for {P(H, i)} {P(W, i)}",
        lambda i: f"can you dig into {topic} for {P(U, i)}",
        lambda i: f"flag me any {topic} on {P(H, i)} {P(W, i)}",
    ]
    if agent == "threat_intel":
        pats += [
            lambda i: f"is {g.indicator_for(tool, i)} sketchy",
            lambda i: f"run intel on {g.indicator_for(tool, i)}",
            lambda i: f"whats the deal with {g.indicator_for(tool, i)}",
        ]
    elif agent == "malware_analysis":
        pats += [
            lambda i: f"can you look at {P(g.FILES, i)} from {P(H, i)}",
            lambda i: f"whats {P(g.FILES, i)} doing on {P(H, i)}",
            lambda i: f"analyze {P(g.FILES, i)} we pulled off {P(H, i)}",
        ]
    elif agent == "network_analysis":
        pats += [
            lambda i: f"see anything off in {topic} for {P(H, i)}",
            lambda i: f"is {P(H, i)} talking to {P(I, i)}",
            lambda i: f"traffic check on {P(H, i)} {P(W, i)}",
        ]
    elif agent == "email_security":
        pats += [
            lambda i: f"take a look at {P(g.MSG_IDS, i)}",
            lambda i: f"does {P(g.MSG_IDS, i)} look malicious",
            lambda i: f"review {P(g.MSG_IDS, i)} for me",
        ]
    elif agent == "log_search":
        pats += [
            lambda i: f"pull up {topic} for {P(U, i)} {P(W, i)}",
            lambda i: f"anything on {P(U, i)} for {topic} {P(W, i)}",
            lambda i: f"check {P(H, i)} logs for {topic}",
        ]
    return pats


def new_desc_patterns(tool: dict, topic: str, req: str) -> list:
    P, U, H, I, W, S, T, N = (
        g.pick, g.USERS, g.HOSTS, g.IPS, g.WHEN, g.SEVERITIES, g.TIMES, g.INC_IDS
    )
    return [
        lambda i: f"Priority {P(S, i)} — observed {topic} on {P(H, i)} ({P(I, i)}) at {P(T, i)} UTC, {req}",
        lambda i: f"SOC queue INC-{P(N, i)}: {P(U, i)} associated with {topic} on {P(H, i)}, {req}",
        lambda i: f"Detection fired Severity {P(S, i)} on {P(H, i)} for {topic} originating {P(I, i)}, {req}",
        lambda i: f"Analyst note — {P(U, i)} reported {topic} affecting {P(H, i)} around {P(T, i)} UTC, {req}",
        lambda i: f"Correlation hit — {topic} spanning {P(H, i)} and {P(I, i)} {P(W, i)}, {req}",
        lambda i: f"Ticket INC-{P(N, i)} Severity {P(S, i)}: {topic} flagged on {P(H, i)}, {req}",
        lambda i: f"Threat hunt lead — {topic} pattern on {P(H, i)} tied to {g.indicator_for(tool, i)}, {req}",
        lambda i: f"Sensor alert {P(S, i)} — {P(H, i)} exhibiting {topic} toward {P(I, i)} at {P(T, i)} UTC, {req}",
        lambda i: f"Handoff INC-{P(N, i)} — {P(U, i)} on {P(H, i)}, {topic} under review, {req}",
        lambda i: f"Watchlist trigger — {g.indicator_for(tool, i)} linked to {topic} on {P(H, i)}, {req}",
        lambda i: f"EDR flagged Severity {P(S, i)}: {topic} on {P(H, i)} by {P(U, i)} {P(W, i)}, {req}",
        lambda i: f"Containment review — {topic} on {P(H, i)} ({P(I, i)}) since {P(T, i)} UTC, {req}",
        lambda i: f"Queue item INC-{P(N, i)} — {topic} on {P(H, i)} from {P(I, i)} at {P(T, i)} UTC, {req}",
        lambda i: f"Hunt finding Severity {P(S, i)} — {P(U, i)} {topic} on {P(H, i)}, {req}",
    ]


def build_unique(builders, count, seen, start=0, nl=False):
    out = []
    i = start
    attempt = 0
    while len(out) < count:
        builder = builders[(start + len(out)) % len(builders)]
        text = builder(i)
        i += 1
        attempt += 1
        if nl:
            text = g.trim_nl(text)
        text = re.sub(r"\s+", " ", text).strip()
        if text not in seen:
            seen.add(text)
            out.append(text)
        if attempt > count * 100:
            raise RuntimeError(f"stuck generating {count} for {tool_id_hint(builders)}")
    return out


def tool_id_hint(_builders):
    return "<tool>"


def tool_start(tool: dict) -> int:
    return sum(ord(c) for c in tool["tool_id"]) % 997


def load_chunk(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_chunk(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
        f.write("\n")


def group_existing(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        grouped[r["tool_id"]].append(r)
    return grouped


# ── Step 1 ───────────────────────────────────────────────────────────────────
def step1_confirm(grouped_tools: dict[str, list[dict]]) -> None:
    print("=== Step 1: Confirm current count per tool (expect 10 each) ===")
    for agent in AGENT_ORDER:
        nl = load_chunk(CHUNK_DIR / f"{agent}_nl.json")
        mixed = load_chunk(CHUNK_DIR / f"{agent}_mixed.json")
        nl_counts = Counter(r["tool_id"] for r in nl)
        mx_counts = Counter(r["tool_id"] for r in mixed)
        nl_bad = {t: c for t, c in nl_counts.items() if c != 10}
        mx_bad = {t: c for t, c in mx_counts.items() if c != 10}
        print(f"  {agent}: tools={len(nl_counts)} "
              f"nl[min={min(nl_counts.values())},max={max(nl_counts.values())}] "
              f"mixed[min={min(mx_counts.values())},max={max(mx_counts.values())}] "
              f"-> all==10: {not nl_bad and not mx_bad}")
        if nl_bad or mx_bad:
            print(f"    WARN nl_bad={nl_bad} mixed_bad={mx_bad}")


# ── Step 2 ───────────────────────────────────────────────────────────────────
def step2_expand(grouped_tools: dict[str, list[dict]]) -> None:
    for idx, agent in enumerate(AGENT_ORDER):
        tools = grouped_tools[agent]
        nl_path = CHUNK_DIR / f"{agent}_nl.json"
        mixed_path = CHUNK_DIR / f"{agent}_mixed.json"

        existing_nl = load_chunk(nl_path)
        existing_mixed = load_chunk(mixed_path)
        nl_by_tool = group_existing(existing_nl)
        mixed_by_tool = group_existing(existing_mixed)

        new_nl_records: list[dict] = []
        new_mixed_records: list[dict] = []

        for tool in tools:
            tid = tool["tool_id"]
            topic = g.tool_topic(tool)
            req = g.tool_request(tool)
            start = tool_start(tool)

            nl_builders = new_nl_patterns(tool, topic)
            desc_builders = new_desc_patterns(tool, topic, req)

            # NL chunk: +20 nl, distinct from existing nl
            seen_nl = {r["query_text"] for r in nl_by_tool[tid]}
            add_nl = build_unique(nl_builders, ADD_PER_TOOL, seen_nl, start, nl=True)
            new_nl_records.extend(g.make_records(tool, add_nl, "nl"))

            # Mixed chunk: +14 desc, +6 nl, distinct from existing mixed entries
            seen_mx_desc = {r["query_text"] for r in mixed_by_tool[tid]
                            if r["style"] == "descriptive"}
            seen_mx_nl = {r["query_text"] for r in mixed_by_tool[tid]
                          if r["style"] == "nl"}
            add_desc = build_unique(desc_builders, MIXED_ADD_DESC, seen_mx_desc, start)
            add_mx_nl = build_unique(nl_builders, MIXED_ADD_NL, seen_mx_nl, start + 7, nl=True)
            new_mixed_records.extend(g.make_records(tool, add_desc, "descriptive"))
            new_mixed_records.extend(g.make_records(tool, add_mx_nl, "nl"))

        save_chunk(nl_path, existing_nl + new_nl_records)
        save_chunk(mixed_path, existing_mixed + new_mixed_records)

        step = idx + 2  # Step 2..6 conceptually, agent-by-agent
        print(f"\n=== Step 2 ({agent}) — appended {len(new_nl_records)} NL, "
              f"{len(new_mixed_records)} mixed ({MIXED_ADD_DESC} desc + {MIXED_ADD_NL} nl per tool) ===")
        print(f"  {nl_path.name}: {len(existing_nl)} -> {len(existing_nl)+len(new_nl_records)}")
        print(f"  {mixed_path.name}: {len(existing_mixed)} -> {len(existing_mixed)+len(new_mixed_records)}")
        print("  Sample 3 NEW queries:")
        samples = new_nl_records[:2] + new_mixed_records[:1]
        for rec in samples:
            print(f"    [{rec['style']}] ({rec['tool_id']}) {rec['query_text']}")


# ── Step 4 (fresh descriptive) ───────────────────────────────────────────────
def step4_descriptive(grouped_tools: dict[str, list[dict]]) -> None:
    print("\n=== Step 4: Generate fresh 100% descriptive chunks (30/tool) ===")
    for agent in AGENT_ORDER:
        tools = grouped_tools[agent]
        # avoid reusing descriptive queries already present in the mixed chunks
        existing_mixed = load_chunk(CHUNK_DIR / f"{agent}_mixed.json")
        used_desc_by_tool: dict[str, set] = defaultdict(set)
        for r in existing_mixed:
            if r["style"] == "descriptive":
                used_desc_by_tool[r["tool_id"]].add(r["query_text"])

        records: list[dict] = []
        for tool in tools:
            tid = tool["tool_id"]
            topic = g.tool_topic(tool)
            req = g.tool_request(tool)
            start = tool_start(tool)
            builders = g.desc_patterns(tool, topic, req) + new_desc_patterns(tool, topic, req)
            seen = set(used_desc_by_tool[tid])
            queries = build_unique(builders, DESC_FILE_PER_TOOL, seen, start)
            records.extend(g.make_records(tool, queries, "descriptive"))

        path = CHUNK_DIR / f"{agent}_descriptive.json"
        save_chunk(path, records)
        print(f"  {path.name}: {len(records)} queries")
        print("  Sample 3:")
        seen_t = set()
        shown = 0
        for rec in records:
            if rec["tool_id"] in seen_t:
                continue
            seen_t.add(rec["tool_id"])
            print(f"    [descriptive] ({rec['tool_id']}) {rec['query_text']}")
            shown += 1
            if shown >= 3:
                break


# ── merge + verify ───────────────────────────────────────────────────────────
def merge_kind(suffix: str) -> list[dict]:
    merged: list[dict] = []
    for agent in AGENT_ORDER:
        merged.extend(load_chunk(CHUNK_DIR / f"{agent}_{suffix}.json"))
    return merged


def verify(records: list[dict], label: str, expect: str) -> None:
    tool_counts = Counter(r["tool_id"] for r in records)
    agent_counts = Counter(r["agent_id"] for r in records)
    style_counts = Counter(r["style"] for r in records)
    fields_ok = all(set(r.keys()) == {"query_text", "agent_id", "tool_id", "style"}
                    for r in records)
    total = len(records)
    mn, mx = min(tool_counts.values()), max(tool_counts.values())

    print(f"\n=== Verify {label} ===")
    print(f"  total={total} tools={len(tool_counts)} per_tool[min={mn},max={mx}] fields_ok={fields_ok}")
    print(f"  by_agent={dict(sorted(agent_counts.items()))}")
    pct = {s: round(100 * c / total, 1) for s, c in style_counts.items()}
    print(f"  by_style={dict(sorted(style_counts.items()))} pct={pct}")

    assert total == 6000, f"{label}: expected 6000 got {total}"
    assert len(tool_counts) == 200, f"{label}: expected 200 tools"
    assert mn == mx == TARGET_PER_TOOL, f"{label}: per-tool not all {TARGET_PER_TOOL}"
    assert fields_ok, f"{label}: bad fields"
    if expect == "nl":
        assert style_counts.get("nl") == total, f"{label}: not 100% nl"
    elif expect == "descriptive":
        assert style_counts.get("descriptive") == total, f"{label}: not 100% descriptive"
    elif expect == "mixed":
        desc_pct = 100 * style_counts["descriptive"] / total
        print(f"  descriptive share: {desc_pct:.1f}% (additions were 70/30)")


def step3_merge() -> None:
    print("\n=== Step 3: Re-merge nl + mixed final datasets ===")
    nl_final = merge_kind("nl")
    mixed_final = merge_kind("mixed")

    for name, recs in [("queries_nl.json", nl_final), ("queries_mixed.json", mixed_final)]:
        path = ROOT / "data" / name
        with path.open("w", encoding="utf-8") as f:
            json.dump(recs, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"  wrote {len(recs)} -> {path}")

    verify(nl_final, "queries_nl.json", "nl")
    verify(mixed_final, "queries_mixed.json", "mixed")


def step4_merge() -> None:
    desc_final = merge_kind("descriptive")
    path = ROOT / "data" / "queries_descriptive.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(desc_final, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\n  wrote {len(desc_final)} -> {path}")
    verify(desc_final, "queries_descriptive.json", "descriptive")


def main() -> None:
    registry = g.load_registry(g.REGISTRY_PATH)
    grouped_tools = g.group_tools_by_agent(registry)

    step1_confirm(grouped_tools)
    step2_expand(grouped_tools)
    step3_merge()
    step4_descriptive(grouped_tools)
    step4_merge()
    print("\nDone.")


if __name__ == "__main__":
    main()
