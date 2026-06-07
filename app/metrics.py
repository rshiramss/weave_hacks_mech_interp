"""Aggregate eval metrics for the front-end sidebar (Phase 2).

Reads the per-arm summary produced by scripts/run_eval.py — the flat
`eval/{arm}/{metric}` scalars on the latest finished eval run in the
`weavehacks-soc-probes` W&B project (these read back reliably via wandb.Api,
unlike Weave Evaluation scorer aggregates). Falls back to the local
data/eval_summary.json the same script writes, then to empty.

Cached in-process (evals change rarely); `?refresh=1` bypasses the cache.
"""

import json
import time
from pathlib import Path

WANDB_PROJECT = "weavehacks-soc-probes"
SUMMARY_PATH = Path(__file__).resolve().parents[1] / "data" / "eval_summary.json"
CACHE_TTL_SECONDS = 300

_cache: dict = {"at": 0.0, "data": None}


def _from_wandb() -> dict | None:
    """Per-arm metrics from the newest finished eval run's summary scalars."""
    import wandb

    api = wandb.Api()
    runs = api.runs(
        f"{api.default_entity}/{WANDB_PROJECT}",
        filters={"jobType": "eval"},
        order="-created_at",
    )
    for run in runs:
        keys = [k for k in run.summary.keys() if k.startswith("eval/")]
        if not keys:
            continue
        arms: dict[str, dict] = {}
        for key in keys:
            _, arm, metric = key.split("/", 2)
            arms.setdefault(arm, {})[metric] = run.summary[key]
        return {"source": "wandb", "run": run.name, "arms": arms}
    return None


def _from_local() -> dict | None:
    if SUMMARY_PATH.exists():
        return {"source": "local", "arms": json.loads(SUMMARY_PATH.read_text())}
    return None


def eval_metrics(refresh: bool = False) -> dict:
    """Per-arm aggregate metrics, cached. Shape: {source, arms: {arm: {metric: v}}}."""
    if (not refresh and _cache["data"]
            and time.time() - _cache["at"] < CACHE_TTL_SECONDS):
        return _cache["data"]

    data: dict | None = None
    try:
        data = _from_wandb()
    except Exception as exc:  # noqa: BLE001 — never 500 the sidebar over a read
        data = {"source": "error", "error": str(exc), "arms": {}}

    if not data or not data.get("arms"):
        data = _from_local() or data or {"source": "empty", "arms": {}}

    _cache.update(at=time.time(), data=data)
    return data
