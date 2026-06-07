"""Validate the probe artifacts against the registry before any demo/eval (§5.5).

Cheap guard against the silent-accuracy-collapse failure mode (PROBE_PLAN.md §G.6).
For the agent probe and every per-agent tool probe in probe_config.json:

  - load the pkl
  - assert its classes_ match the registry exactly
        agent probe  -> the 5 agent_ids
        tool  probe  -> that agent's 40 tool_ids (no foreign / missing tools)
  - assert n_features_in_ == d_model (3584)
  - run a dummy predict_proba(zeros(d_model)) to prove the pickled sklearn version
    is compatible (old minors load but crash here)

Exit non-zero on any mismatch.

    python scripts/validate_probes.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROBES_DIR = ROOT / "data" / "probes"
REGISTRY_PATH = ROOT / "data" / "registry.json"
CONFIG_FILE = "probe_config.json"


def _registry_vocab() -> tuple[set[str], dict[str, set[str]]]:
    """Return (all agent_ids, {agent_id: set(tool_ids)}) from the registry."""
    reg = json.loads(REGISTRY_PATH.read_text())
    agents = {a["agent_id"] for a in reg["agents"]}
    tools_by_agent: dict[str, set[str]] = {a: set() for a in agents}
    for tool in reg["tools"]:
        tools_by_agent.setdefault(tool["agent_id"], set()).add(tool["tool_id"])
    return agents, tools_by_agent


def _check(clf, expected: set[str], d_model: int, label: str,
           errors: list[str]) -> None:
    classes = {str(c) for c in clf.classes_}
    if classes != expected:
        extra = sorted(classes - expected)[:5]
        missing = sorted(expected - classes)[:5]
        errors.append(f"{label}: classes_ != registry "
                      f"(extra={extra}, missing={missing})")
    n_feat = getattr(clf, "n_features_in_", None)
    if n_feat != d_model:
        errors.append(f"{label}: n_features_in_={n_feat}, expected {d_model}")
    try:
        proba = clf.predict_proba(np.zeros((1, n_feat or d_model), dtype=np.float32))
        if proba.shape != (1, len(clf.classes_)):
            errors.append(f"{label}: predict_proba shape {proba.shape} unexpected")
    except Exception as exc:  # noqa: BLE001 — surface the sklearn-version crash clearly
        errors.append(f"{label}: predict_proba crashed ({exc}) — check sklearn>=1.9")


def main() -> int:
    config = json.loads((PROBES_DIR / CONFIG_FILE).read_text())
    d_model = int(config.get("d_model", 3584))
    agent_ids, tools_by_agent = _registry_vocab()
    errors: list[str] = []

    agent_clf = joblib.load(PROBES_DIR / config["agent_probe_file"])
    _check(agent_clf, agent_ids, d_model, "agent_probe", errors)

    tool_files = config["tool_probe_files"]
    if set(tool_files) != agent_ids:
        errors.append(f"tool_probe_files keys {sorted(tool_files)} "
                      f"!= registry agents {sorted(agent_ids)}")

    for agent_id, filename in tool_files.items():
        path = PROBES_DIR / filename
        if not path.exists():
            errors.append(f"{agent_id}: missing probe file {filename}")
            continue
        clf = joblib.load(path)
        _check(clf, tools_by_agent.get(agent_id, set()), d_model,
               f"tool_probe[{agent_id}]", errors)

    if errors:
        print("[validate] FAIL:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"[validate] OK — agent probe (5 classes) + {len(tool_files)} per-agent "
          f"tool probes (40 classes each), d_model={d_model}, predict_proba ok.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
