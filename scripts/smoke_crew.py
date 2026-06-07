"""Smoke-test one specialist crew (Stage 3 verification, §6.5).

Builds a single crew with a manually supplied tool shortlist (no probes) and
kicks it off on a sample analyst message, then checks the output for the five
required triage keys.

Usage:
    python scripts/smoke_crew.py --agent log_search \
        --tools search_auth_logs,correlate_by_source_ip,build_event_timeline,search_impossible_travel,query_login_history
"""

import argparse
import re
import sys
from pathlib import Path

# Allow `python scripts/smoke_crew.py` (adds repo root so `src` imports resolve).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from src.crews.specialist import MAX_TOOLS, build_specialist_crew  # noqa: E402
from src.probe.query_contract import normalize_query  # noqa: E402

SAMPLE_QUERY = (
    "We're seeing 847 failed SSH logins to prod-bastion-01 from 203.0.113.44 "
    "in the last 15 minutes, mostly root — can someone pull auth logs?"
)

REQUIRED_KEYS = [
    "alert_summary",
    "tools_used",
    "findings",
    "severity_assessment",
    "recommended_next_step",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test one specialist crew.")
    parser.add_argument("--agent", required=True, help="agent_id, e.g. log_search")
    parser.add_argument(
        "--tools",
        required=True,
        help="comma-separated tool_ids (≤5), e.g. search_auth_logs,...",
    )
    parser.add_argument(
        "--query",
        default=SAMPLE_QUERY,
        help="natural-language analyst message (default: sample SSH brute-force)",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    tool_ids = [t.strip() for t in args.tools.split(",") if t.strip()]
    if len(tool_ids) > MAX_TOOLS:
        print(f"ERROR: {len(tool_ids)} tools given; cap is {MAX_TOOLS}.", file=sys.stderr)
        return 2

    query = normalize_query(args.query)

    print(f"[smoke] agent={args.agent} tools={tool_ids}")
    print(f"[smoke] query={query!r}\n")

    crew = build_specialist_crew(args.agent, tool_ids)
    print(f"[smoke] crew built: {len(crew.tasks)} task, "
          f"{len(crew.agents[0].tools)} tools attached\n")

    result = crew.kickoff(inputs={"context": query})
    raw = getattr(result, "raw", None) or str(result)

    print("\n[smoke] --- crew output ---")
    print(raw)

    found = [k for k in REQUIRED_KEYS if re.search(rf'["\']?{k}["\']?\s*:', raw)]
    missing = [k for k in REQUIRED_KEYS if k not in found]

    print(f"\n[smoke] triage keys found: {found}")
    if missing:
        print(f"[smoke] FAIL — missing keys: {missing}", file=sys.stderr)
        return 1

    print("[smoke] PASS — all required triage keys present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
