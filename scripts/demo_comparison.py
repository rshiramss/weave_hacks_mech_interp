"""Trace Comparison demo: same query through two arms (Stage 7, §7.4).

Runs the FULL routing+crew path for two arms on one query so Weave records a
rich trace tree for each, then prints the trace URLs/ids. Open both in the Weave
UI → select → Compare → Calls view to show the depth difference.

Arm 3 (probe) is not built yet; passing it uses a clearly-labeled STUB (RAG
retrieval standing in) so the demo flow can be rehearsed. Swap in the real probe
router before the final demo.

Usage:
    python scripts/demo_comparison.py --query "847 failed SSH logins from 203.0.113.44 on prod-bastion-01"
    python scripts/demo_comparison.py --arms incontext,probe --query "..."
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.setrecursionlimit(10_000)

import weave  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from src.probe.query_contract import normalize_query  # noqa: E402
from src.weave_setup import init_weave  # noqa: E402

SAMPLE_QUERY = "847 failed SSH logins from 203.0.113.44 on prod-bastion-01"


@weave.op(name="probe_router_STUB")
def _stub_probe_router(query_text: str) -> dict:
    """STUB — NOT a real probe. RAG retrieval stands in to rehearse the Arm-3 trace."""
    from src.baselines.rag_router import rag_router

    out = rag_router(query_text)
    return {**out, "arm": "probe-STUB"}


def _load_arm(arm: str):
    """Return (display_name, op, is_stub) for an arm."""
    if arm == "incontext":
        from src.baselines.frontier_incontext import incontext_router
        return "Arm 1 — Full-catalog in-context", incontext_router, False
    if arm == "rag":
        from src.baselines.rag_router import rag_router
        return "Arm 2 — RAG", rag_router, False
    if arm == "probe":
        return "Arm 3 — Probe router (STUB)", _stub_probe_router, True
    raise ValueError(f"Unknown arm: {arm}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace Comparison demo.")
    parser.add_argument("--query", default=SAMPLE_QUERY)
    parser.add_argument("--arms", default="incontext,rag",
                        help="two arms to compare, csv of: incontext,rag,probe")
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    query = normalize_query(args.query)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    init_weave()
    print(f"[demo] query={query!r}")
    print(f"[demo] comparing arms: {arms}\n")

    results = []
    for arm in arms:
        display, op, is_stub = _load_arm(arm)
        out, call = op.call(query)
        url = getattr(call, "ui_url", None) or getattr(call, "id", "?")
        stub_tag = "  ⚠️ STUB (not a real probe)" if is_stub else ""
        print(f"[demo] {display}{stub_tag}")
        print(f"       route: agent={out.get('agent_id')} tool={out.get('tool_id')}")
        print(f"       trace: {url}\n")
        results.append((display, url, is_stub))

    print("[demo] --- Trace Comparison ---")
    print("Open Weave UI → Traces → select the two traces above → Compare → Calls view.")
    if any(is_stub for _d, _u, is_stub in results):
        print("NOTE: a STUB stands in for the probe router — swap in the real Arm 3 "
              "before the final demo so judges are not misled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
