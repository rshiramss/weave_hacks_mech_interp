"""Run a baseline router arm on one query (Stage 5 verification, §8).

Local control plane: init Weave, route (Arm 1 in-context LLM / Arm 2 RAG), run
the specialist crew locally, flush Weave.

Usage:
    python scripts/run_baseline.py --arm incontext --query "..."   # Arm 1 (local Ollama)
    python scripts/run_baseline.py --arm rag --query "..."         # Arm 2 (Modal embeddings)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Weave auto-traces CrewAI's deeply-nested Crew/Agent/Task object graph; its
# serializer can exceed the default 1000-frame recursion limit. Raise it so the
# crew subtree traces cleanly (the graph is deep but finite, not cyclic).
sys.setrecursionlimit(10_000)

from dotenv import load_dotenv  # noqa: E402

from src.probe.query_contract import normalize_query  # noqa: E402
from src.weave_setup import init_weave  # noqa: E402

SAMPLE_QUERY = (
    "VPN login from 203.0.113.44 for svc-backup — can you check auth logs?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a baseline router arm.")
    parser.add_argument("--arm", required=True, choices=["incontext", "rag"])
    parser.add_argument("--query", default=SAMPLE_QUERY)
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    query = normalize_query(args.query)

    client = init_weave()
    print(f"[baseline] arm={args.arm} query={query!r}\n")

    if args.arm == "incontext":
        from src.baselines.frontier_incontext import incontext_router
        out = incontext_router(query)
    else:
        from src.baselines.rag_router import rag_router
        out = rag_router(query)

    route = {k: out[k] for k in ("arm", "agent_id", "tool_id", "tool_shortlist")}
    print("[baseline] --- route ---")
    print(json.dumps(route, indent=2))
    print("\n[baseline] --- crew result ---")
    print(out["result"])

    if client is not None:
        client.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
