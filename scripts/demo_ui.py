"""CLI demo (Stage 8): sim alert -> NL -> route -> triage + Weave trace link.

Usage:
    python scripts/demo_ui.py                       # preset SSH alert, stub backend
    python scripts/demo_ui.py --backend rag
    python scripts/demo_ui.py --event '{"summary":"...","source_ip":"203.0.113.44"}'
    python scripts/demo_ui.py --message "..." --no-exec
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.setrecursionlimit(10_000)

from dotenv import load_dotenv  # noqa: E402

from src.demo_pipeline import route_alert  # noqa: E402
from src.weave_setup import WEAVE_PROJECT, init_weave  # noqa: E402

# Matches the SSH holdout-style alert used in the Trace Comparison demo (§7.4).
SAMPLE_EVENT = {
    "summary": "847 failed SSH logins, mostly root",
    "source_ip": "203.0.113.44",
    "host": "prod-bastion-01",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SOC probe-router demo (CLI).")
    parser.add_argument("--event", help="structured alert as a JSON string")
    parser.add_argument("--message", help="plain natural-language alert")
    parser.add_argument("--backend", help="stub | rag | incontext | probe")
    parser.add_argument("--no-exec", action="store_true",
                        help="route only, skip the (slow) crew kickoff")
    return parser.parse_args()


def resolve_event(args: argparse.Namespace):
    if args.event:
        return json.loads(args.event)
    if args.message:
        return args.message
    return SAMPLE_EVENT


def main() -> int:
    load_dotenv()
    args = parse_args()
    client = init_weave()
    event = resolve_event(args)

    out, call = route_alert.call(event, backend=args.backend, execute=not args.no_exec)

    print(f"[demo-ui] backend     : {out['backend']}")
    print(f"[demo-ui] NL query    : {out['query_text']!r}")
    print(f"[demo-ui] routed agent: {out['agent_id']}")
    print(f"[demo-ui] routed tool : {out['tool_id']}")
    print(f"[demo-ui] shortlist   : {out['tool_shortlist']}")
    if out["result"]:
        print("\n[demo-ui] --- triage ---")
        print(out["result"])

    url = getattr(call, "ui_url", None)
    print(f"\n[demo-ui] Weave trace : {url or '(project) ' + WEAVE_PROJECT}")

    if client is not None:
        client.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
