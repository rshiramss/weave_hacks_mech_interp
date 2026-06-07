"""Run the probe router (Arm 3) on one query (Stage 4 verification, §6).

    python scripts/run_probe_router.py --query "847 failed SSH logins from 203.0.113.44 on prod-bastion-01"

Confirm the Weave trace tree: ingest_event -> forward_pass -> agent_probe ->
agent_picker -> tool_probe -> crewai.Crew.kickoff.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.setrecursionlimit(10_000)

from dotenv import load_dotenv  # noqa: E402

from src.flow import run_probe_flow  # noqa: E402
from src.weave_setup import init_weave  # noqa: E402

SAMPLE_QUERY = "847 failed SSH logins from 203.0.113.44 on prod-bastion-01"


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the probe router (Arm 3).")
    parser.add_argument("--query", default=SAMPLE_QUERY)
    args = parser.parse_args()

    client = init_weave()
    out = run_probe_flow(args.query)

    route = {k: out[k] for k in ("arm", "agent_id", "tool_id", "tool_shortlist",
                                 "agent_candidates")}
    print("[probe-router] --- route ---")
    print(json.dumps(route, indent=2))
    print("\n[probe-router] --- triage ---")
    print(out["result"])

    if client is not None:
        client.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
