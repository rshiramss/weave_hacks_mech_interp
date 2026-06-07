#!/usr/bin/env python3
"""Generate query datasets in per-agent chunks, then merge."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

QUERIES_PER_TOOL = 10
NL_IN_MIXED = 5
DESC_IN_MIXED = 5
SEED = 42

ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "data" / "registry.json"
CHUNK_DIR = ROOT / "data" / "chunks"
AGENT_ORDER = [
    "log_search",
    "threat_intel",
    "malware_analysis",
    "network_analysis",
    "email_security",
]

USERS = [
    "john.doe", "j.davis", "m.chen", "sarah.kim", "t.nguyen", "p.walsh", "c.lopez",
    "admin.root", "svc_backup", "svc_deploy", "svc_ci", "ext_jlee", "a.patel", "r.gomez",
]
HOSTS = [
    "WKSTN-042", "WKSTN-087", "WKSTN-019", "LAPTOP-019", "DEV-LT-09", "prod-dc-01",
    "SRV-APP-03", "SRV-DC-01", "SRV-FS-01", "SRV-WEB-02", "SRV-LNX-04", "SRV-SQL-02",
    "SRV-PAY-01", "SRV-VPN-01", "FW-EDGE-01", "PROXY-01",
]
IPS = [
    "45.33.32.156", "185.220.101.45", "203.0.113.44", "103.21.244.0", "198.51.100.77",
    "192.0.2.58", "172.16.44.12", "10.10.5.12", "52.86.141.33", "91.134.204.11",
]
DOMAINS = [
    "evil-login.com", "cdn-update.net", "secure-corp-login.com", "c0rp-invoices.net",
    "login-microsoftonline.net", "corp-sso-auth.net", "c0rp-executive.com", "mail-relay.xyz",
]
HASHES = [
    "7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd7abf06b4f2c",
    "a3f5c8d2e1b9047f6a8c3d2e1b9047f6a8c3d2e1b9047f6a8c3d2e1b9047f",
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
]
URLS = [
    "hxxps://cdn-update.net/dl/payload",
    "hxxps://secure-corp-login.com/auth",
    "hxxps://bit.ly/3xK9mPq",
]
WHEN = [
    "this morning", "overnight", "in the last hour", "today", "since midnight",
    "over the past 24 hours", "this week", "after hours", "during the night shift", "just now",
]
SEVERITIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
INC_IDS = ["4421", "4478", "4510", "4399", "4555", "4602", "4688", "4712", "4750", "4801"]
TIMES = [
    "01:22", "02:14", "03:00", "03:30", "04:17", "05:01", "06:33", "07:12", "14:33", "23:58",
]
FILES = ["update.exe", "invoice_Q2.xlsm", "statement.pdf", "invoice.doc", "payload.ps1"]
MSG_IDS = ["msg-88421", "msg-99012", "msg-77102", "msg-90144", "msg-91555"]

NAME_PREFIXES = (
    "search ", "lookup ", "check ", "detect ", "analyze ", "correlate by ", "correlate ",
    "build ", "export ", "compute ", "count ", "query ", "tail ", "summarize ", "submit ",
    "get ", "extract ", "scan with ", "scan ", "unpack ", "disassemble ", "decompile ",
    "identify ", "classify ", "trace ", "capture ", "map sample to ", "map to ", "map ",
    "generate ", "assess ", "quarantine ", "retrieve ", "detonate ", "parse ", "block ",
    "release ", "recall ", "enrich ", "score ", "simulate ", "inspect ", "list ", "compare ",
    "find ", "pull ", "run ", "follow ", "group ", "flag ", "verify ", "evaluate ", "match ",
    "measure ", "test ", "add ", "resolve ", "join ", "link ", "aggregate ", "construct ",
    "stream ", "audit ", "profile ", "reassemble ", "carve ", "estimate ", "reconstruct ",
    "establish ", "normalize ", "break down ", "show ", "determine ", "assign ", "retrieve ",
)


def load_registry(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def group_tools_by_agent(registry: dict) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for tool in registry["tools"]:
        grouped[tool["agent_id"]].append(tool)
    for agent in grouped:
        grouped[agent].sort(key=lambda t: t["tool_id"])
    return grouped


def step1_print_counts(grouped: dict[str, list[dict]]) -> int:
    print("=== Step 1: Tool count per agent ===")
    total = 0
    for agent in AGENT_ORDER:
        count = len(grouped[agent])
        total += count
        print(f"  {agent}: {count} tools")
    print(f"  TOTAL: {total} tools")
    print(f"  Target per file: {total * QUERIES_PER_TOOL} queries ({QUERIES_PER_TOOL} per tool)")
    return total


def tool_topic(tool: dict) -> str:
    name = tool["name"].lower()
    for prefix in NAME_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :].strip()
    return name


def tool_request(tool: dict) -> str:
    desc = tool["description"].rstrip(".")
    req = desc[0].lower() + desc[1:]
    replacements = {
        "Retrieve ": "pull ",
        "Query ": "query ",
        "Search ": "search ",
        "Check ": "check ",
        "Detect ": "detect ",
        "Analyze ": "analyze ",
        "Get ": "get ",
        "Find ": "find ",
        "List ": "list ",
        "Compute ": "compute ",
        "Construct ": "build ",
        "Export ": "export ",
        "Determine ": "determine ",
        "Flag ": "flag ",
        "Match ": "match ",
        "Map ": "map ",
        "Generate ": "generate ",
        "Assess ": "assess ",
        "Summarize ": "summarize ",
        "Add ": "add ",
        "Verify ": "verify ",
        "Evaluate ": "evaluate ",
        "Follow ": "follow ",
        "Group ": "group ",
        "Count ": "count ",
        "Test ": "test ",
        "Join ": "join ",
        "Link ": "link ",
        "Aggregate ": "aggregate ",
        "Stream ": "stream ",
        "Inspect ": "inspect ",
        "Profile ": "profile ",
        "Reassemble ": "reassemble ",
        "Carve ": "carve ",
        "Measure ": "measure ",
        "Trace ": "trace ",
        "Simulate ": "simulate ",
        "Resolve ": "resolve ",
        "Score ": "score ",
        "Assign ": "assign ",
        "Run ": "run ",
        "Scan ": "scan ",
        "Extract ": "extract ",
        "Submit ": "submit ",
        "Detonate ": "detonate ",
        "Parse ": "parse ",
        "Quarantine ": "quarantine ",
        "Recall ": "recall ",
        "Release ": "release ",
        "Block ": "block ",
        "Enrich ": "enrich ",
        "Produce ": "produce ",
        "Decompile ": "decompile ",
        "Disassemble ": "disassemble ",
        "Unpack ": "unpack ",
        "Identify ": "identify ",
        "Classify ": "classify ",
        "Capture ": "capture ",
        "Reconstruct ": "reconstruct ",
        "Establish ": "establish ",
        "Compare ": "compare ",
        "Pull ": "pull ",
        "Look up ": "look up ",
    }
    for old, new in replacements.items():
        if req.startswith(old):
            req = new + req[len(old) :]
            break
    return req


def pick(pool: list, idx: int):
    return pool[idx % len(pool)]


def word_count(text: str) -> int:
    return len(text.split())


def trim_nl(text: str, max_words: int = 20) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def indicator_for(tool: dict, idx: int) -> str:
    tid = tool["tool_id"]
    if any(k in tid for k in ("hash", "sample", "file", "attachment", "macro", "binary", "pe", "dll", "yara", "antivirus", "packer", "shellcode", "fuzzy", "imphash", "nsrl", "certificate_signing")):
        return pick(HASHES, idx) if idx % 3 else pick(FILES, idx)
    if "domain" in tid or "whois" in tid or "dns" in tid or "typosquat" in tid or "brand" in tid or "dmarc" in tid or "spf" in tid or "dkim" in tid or "sender" in tid or "homoglyph" in tid or "url" in tid or "phish" in tid or "email" in tid or "campaign" in tid:
        return pick(DOMAINS, idx) if idx % 2 else pick(URLS, idx)
    if "ip" in tid or "netflow" in tid or "firewall" in tid or "vpn" in tid or "tor" in tid or "proxy" in tid or "asn" in tid or "geo" in tid or "beacon" in tid or "pcap" in tid or "tls" in tid or "rdp" in tid or "smb" in tid or "dhcp" in tid or "arp" in tid or "bandwidth" in tid or "connection" in tid or "traffic" in tid or "port" in tid or "protocol" in tid or "lateral" in tid or "exfil" in tid or "icmp" in tid or "5tuple" in tid:
        return pick(IPS, idx)
    if "cve" in tid or "exploit" in tid:
        return "CVE-2024-38112" if idx % 2 else "CVE-2023-23397"
    if "session" in tid:
        return "session 7f3e91b2-4420-4ac1-9d8e-001122334455"
    if "message" in tid or "email" in tid or "header" in tid or "quarantine" in tid or "recall" in tid or "thread" in tid:
        return pick(MSG_IDS, idx)
    if "sandbox" in tid and "report" in tid:
        return "sandbox job sb-88421"
    if "timeline" in tid or "case" in tid:
        return f"INC-{pick(INC_IDS, idx)}"
    return pick(IPS, idx)


def nl_patterns(tool: dict, topic: str) -> list:
    tid = tool["tool_id"]
    agent = tool["agent_id"]
    patterns = [
        lambda i: trim_nl(f"can you check {topic} for {pick(USERS, i)} {pick(WHEN, i)}"),
        lambda i: trim_nl(f"any {topic} on {pick(HOSTS, i)} {pick(WHEN, i)}"),
        lambda i: trim_nl(f"pull {topic} for {pick(HOSTS, i)} {pick(WHEN, i)}"),
        lambda i: trim_nl(f"did {pick(USERS, i)} hit {topic} on {pick(HOSTS, i)} {pick(WHEN, i)}"),
        lambda i: trim_nl(f"show me {topic} tied to {indicator_for(tool, i)}"),
        lambda i: trim_nl(f"need {topic} on {pick(HOSTS, i)} asap"),
        lambda i: trim_nl(f"what {topic} do we have for {pick(USERS, i)}"),
        lambda i: trim_nl(f"run {topic} on {pick(HOSTS, i)} {pick(WHEN, i)}"),
        lambda i: trim_nl(f"is there {topic} from {pick(IPS, i)} {pick(WHEN, i)}"),
        lambda i: trim_nl(f"quick {topic} lookup for {pick(USERS, i)}"),
        lambda i: trim_nl(f"hey can you grab {topic} for {pick(USERS, i)} on {pick(HOSTS, i)}"),
        lambda i: trim_nl(f"got a ticket — check {topic} for {pick(HOSTS, i)} {pick(WHEN, i)}"),
    ]
    if agent == "threat_intel":
        patterns.extend([
            lambda i: trim_nl(f"is {indicator_for(tool, i)} on any blocklists"),
            lambda i: trim_nl(f"what's the reputation on {indicator_for(tool, i)}"),
            lambda i: trim_nl(f"enrich {indicator_for(tool, i)} for me real quick"),
        ])
    elif agent == "malware_analysis":
        patterns.extend([
            lambda i: trim_nl(f"pull sandbox report for {indicator_for(tool, i)}"),
            lambda i: trim_nl(f"hash this sample {pick(FILES, i)} from {pick(HOSTS, i)}"),
            lambda i: trim_nl(f"detonate {pick(FILES, i)} from {pick(HOSTS, i)} in the sandbox"),
        ])
    elif agent == "network_analysis":
        patterns.extend([
            lambda i: trim_nl(f"is {pick(HOSTS, i)} beaconing to {pick(IPS, i)}"),
            lambda i: trim_nl(f"check flows from {pick(HOSTS, i)} to {pick(IPS, i)} {pick(WHEN, i)}"),
            lambda i: trim_nl(f"any weird traffic on {pick(HOSTS, i)} {pick(WHEN, i)}"),
        ])
    elif agent == "email_security":
        patterns.extend([
            lambda i: trim_nl(f"parse headers on {pick(MSG_IDS, i)} for me"),
            lambda i: trim_nl(f"is {pick(MSG_IDS, i)} a phish"),
            lambda i: trim_nl(f"check the attachment on {pick(MSG_IDS, i)}"),
        ])
    elif agent == "log_search":
        patterns.extend([
            lambda i: trim_nl(f"did {pick(USERS, i)} log in from a new device {pick(WHEN, i)}"),
            lambda i: trim_nl(f"show {pick(USERS, i)} activity on {pick(HOSTS, i)} {pick(WHEN, i)}"),
            lambda i: trim_nl(f"any lockouts for {pick(USERS, i)} {pick(WHEN, i)}"),
        ])
    return patterns


def desc_patterns(tool: dict, topic: str, req: str) -> list:
    return [
        lambda i: f"ALERT Severity {pick(SEVERITIES, i)} — {pick(HOSTS, i)} {topic} activity from {pick(IPS, i)}, {req}",
        lambda i: f"INC-{pick(INC_IDS, i)}: {pick(USERS, i)} on {pick(HOSTS, i)} at {pick(TIMES, i)} UTC, {req}",
        lambda i: f"Triage note — {topic} on {pick(HOSTS, i)} linked to {indicator_for(tool, i)}, {req}",
        lambda i: f"Suspicious {topic} on {pick(HOSTS, i)} by {pick(USERS, i)} at {pick(TIMES, i)} UTC, {req}",
        lambda i: f"Case review INC-{pick(INC_IDS, i)} — {pick(HOSTS, i)} and {pick(IPS, i)}, {req}",
        lambda i: f"Escalation Severity {pick(SEVERITIES, i)} — {pick(USERS, i)} {topic} event on {pick(HOSTS, i)}, {req}",
        lambda i: f"Monitoring alert — {topic} spike on {pick(HOSTS, i)} since {pick(TIMES, i)} UTC, {req}",
        lambda i: f"Contractor {pick(USERS, i)} on {pick(HOSTS, i)} with {topic} from {pick(IPS, i)}, {req}",
        lambda i: f"After-hours {topic} detected on {pick(HOSTS, i)} at {pick(TIMES, i)} UTC, {req}",
        lambda i: f"Investigation INC-{pick(INC_IDS, i)} — {topic} involving {indicator_for(tool, i)}, {req}",
    ]


def generate_unique_queries(tool: dict, patterns, count: int, style: str) -> list[str]:
    topic = tool_topic(tool)
    req = tool_request(tool) if style == "descriptive" else topic
    builders = nl_patterns(tool, topic) if style == "nl" else desc_patterns(tool, topic, req)

    seen: set[str] = set()
    results: list[str] = []
    base = sum(ord(c) for c in tool["tool_id"]) % 997
    attempt = 0

    while len(results) < count:
        builder = builders[(base + len(results)) % len(builders)]
        text = builder(base + attempt)
        attempt += 1
        text = re.sub(r"\s+", " ", text).strip()
        if text not in seen:
            seen.add(text)
            results.append(text)
        if attempt > count * 30:
            raise RuntimeError(f"Could not generate {count} unique {style} queries for {tool['tool_id']}")
    return results


def make_records(tool: dict, queries: list[str], style: str) -> list[dict]:
    return [
        {
            "query_text": q,
            "agent_id": tool["agent_id"],
            "tool_id": tool["tool_id"],
            "style": style,
        }
        for q in queries
    ]


def generate_agent_chunks(agent_id: str, tools: list[dict]) -> tuple[list[dict], list[dict]]:
    nl_all: list[dict] = []
    mixed_all: list[dict] = []

    for tool in tools:
        nl_queries = generate_unique_queries(tool, None, QUERIES_PER_TOOL, "nl")
        desc_queries = generate_unique_queries(tool, None, QUERIES_PER_TOOL, "descriptive")

        nl_all.extend(make_records(tool, nl_queries, "nl"))
        mixed_all.extend(make_records(tool, nl_queries[:NL_IN_MIXED], "nl"))
        mixed_all.extend(make_records(tool, desc_queries[:DESC_IN_MIXED], "descriptive"))

    return nl_all, mixed_all


def save_chunk(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
        f.write("\n")


def print_samples(label: str, records: list[dict], n: int = 3) -> None:
    print(f"\n  Sample {label} queries:")
    # Show diverse samples across different tools
    seen_tools: set[str] = set()
    samples: list[dict] = []
    for rec in records:
        if rec["tool_id"] not in seen_tools:
            samples.append(rec)
            seen_tools.add(rec["tool_id"])
        if len(samples) >= n:
            break
    for rec in samples:
        print(f"    [{rec['style']}] ({rec['tool_id']}) {rec['query_text']}")


def generate_all_chunks(grouped: dict[str, list[dict]]) -> None:
    step_map = {
        "log_search": 2,
        "threat_intel": 3,
        "malware_analysis": 4,
        "network_analysis": 4,
        "email_security": 4,
    }
    for agent_id in AGENT_ORDER:
        tools = grouped[agent_id]
        step = step_map[agent_id]
        print(f"\n=== Step {step}: Generating {agent_id} ({len(tools)} tools x {QUERIES_PER_TOOL}) ===")

        nl_records, mixed_records = generate_agent_chunks(agent_id, tools)

        nl_path = CHUNK_DIR / f"{agent_id}_nl.json"
        mixed_path = CHUNK_DIR / f"{agent_id}_mixed.json"
        save_chunk(nl_path, nl_records)
        save_chunk(mixed_path, mixed_records)

        print(f"  Saved {len(nl_records)} NL -> {nl_path}")
        print(f"  Saved {len(mixed_records)} mixed -> {mixed_path}")
        print_samples("NL", nl_records)
        print_samples("mixed", mixed_records)


def merge_chunks() -> tuple[list[dict], list[dict]]:
    nl_merged: list[dict] = []
    mixed_merged: list[dict] = []

    for agent_id in AGENT_ORDER:
        nl_path = CHUNK_DIR / f"{agent_id}_nl.json"
        mixed_path = CHUNK_DIR / f"{agent_id}_mixed.json"
        with nl_path.open(encoding="utf-8") as f:
            nl_merged.extend(json.load(f))
        with mixed_path.open(encoding="utf-8") as f:
            mixed_merged.extend(json.load(f))

    return nl_merged, mixed_merged


def verify_dataset(records: list[dict], label: str, expect_style: str | None = None) -> None:
    tool_counts: dict[str, int] = defaultdict(int)
    agent_counts: dict[str, int] = defaultdict(int)
    style_counts: dict[str, int] = defaultdict(int)

    for rec in records:
        tool_counts[rec["tool_id"]] += 1
        agent_counts[rec["agent_id"]] += 1
        style_counts[rec["style"]] += 1
        required = {"query_text", "agent_id", "tool_id", "style"}
        if set(rec.keys()) != required:
            raise ValueError(f"{label}: bad fields on {rec['tool_id']}: {set(rec.keys())}")

    min_count = min(tool_counts.values()) if tool_counts else 0
    max_count = max(tool_counts.values()) if tool_counts else 0
    missing_tools = 200 - len(tool_counts)

    print(f"\n=== Verification: {label} ===")
    print(f"  Total queries: {len(records)}")
    print(f"  Unique tools: {len(tool_counts)} (missing: {missing_tools})")
    print(f"  Queries per tool: min={min_count}, max={max_count}")
    print(f"  By agent: {dict(sorted(agent_counts.items()))}")
    print(f"  By style: {dict(sorted(style_counts.items()))}")

    if min_count < QUERIES_PER_TOOL:
        low = [t for t, c in tool_counts.items() if c < QUERIES_PER_TOOL]
        raise ValueError(f"{label}: tools with <{QUERIES_PER_TOOL} queries: {low[:10]}")
    if expect_style == "nl" and style_counts.get("nl", 0) != len(records):
        raise ValueError(f"{label}: not all queries are style nl")
    if expect_style == "mixed":
        nl_n = style_counts.get("nl", 0)
        desc_n = style_counts.get("descriptive", 0)
        if nl_n != desc_n:
            raise ValueError(f"{label}: mixed split uneven ({nl_n} nl vs {desc_n} descriptive)")


def step5_merge_and_verify() -> None:
    print("\n=== Step 5: Merge chunks into final datasets ===")
    nl_final, mixed_final = merge_chunks()

    nl_path = ROOT / "data" / "queries_nl.json"
    mixed_path = ROOT / "data" / "queries_mixed.json"

    with nl_path.open("w", encoding="utf-8") as f:
        json.dump(nl_final, f, indent=2, ensure_ascii=False)
        f.write("\n")
    with mixed_path.open("w", encoding="utf-8") as f:
        json.dump(mixed_final, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"  Wrote {len(nl_final)} queries -> {nl_path}")
    print(f"  Wrote {len(mixed_final)} queries -> {mixed_path}")

    verify_dataset(nl_final, "queries_nl.json", expect_style="nl")
    verify_dataset(mixed_final, "queries_mixed.json", expect_style="mixed")


def main() -> None:
    registry = load_registry(REGISTRY_PATH)
    grouped = group_tools_by_agent(registry)
    step1_print_counts(grouped)
    generate_all_chunks(grouped)
    step5_merge_and_verify()
    print("\nDone.")


if __name__ == "__main__":
    main()
