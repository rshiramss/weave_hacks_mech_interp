"""Probe inference + masking over the trained probes (Stage 2/4, §6.3/§6.4).

Loads the saved LogisticRegression probes (data/probes/*.pkl) and exposes:
  AgentProbe.top_k(h, k=2)            -> [(agent_id, score), ...]
  ToolProbe.top_k_masked(h, agent_id, k=5) -> [(tool_id, score), ...]

Masking (zeroing tools outside the chosen agent before top-k) is added here — the
training scripts on abe/probe-training do not mask. The probe `classes_` already
match registry.json exactly (5 agents, 200 tools).

Requires scikit-learn >= 1.9 (the probes were pickled with 1.9.0).
"""

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from src.tools.factory import _registry

_PROBES_DIR = Path(__file__).resolve().parents[2] / "data" / "probes"
AGENT_PROBE_FILE = "orchestrator_probe.pkl"
TOOL_PROBE_FILE = "tool_probe.pkl"


@lru_cache(maxsize=1)
def _tool_to_agent() -> dict:
    return {t["tool_id"]: t["agent_id"] for t in _registry()["tools"]}


class AgentProbe:
    def __init__(self, clf):
        self.clf = clf

    def top_k(self, h: np.ndarray, k: int = 2) -> list[tuple[str, float]]:
        proba = self.clf.predict_proba(np.asarray(h).reshape(1, -1))[0]
        order = np.argsort(-proba)[:k]
        return [(str(self.clf.classes_[i]), float(proba[i])) for i in order]


class ToolProbe:
    def __init__(self, clf, tool_to_agent: dict):
        self.clf = clf
        self.tool_to_agent = tool_to_agent
        # boolean masks per agent, aligned to clf.classes_ order (built once)
        self._owner = np.array([tool_to_agent.get(str(c)) for c in clf.classes_])

    def top_k_masked(self, h: np.ndarray, agent_id: str,
                     k: int = 5) -> list[tuple[str, float]]:
        proba = self.clf.predict_proba(np.asarray(h).reshape(1, -1))[0]
        masked = np.where(self._owner == agent_id, proba, 0.0)
        order = np.argsort(-masked)[:k]
        return [
            (str(self.clf.classes_[i]), float(masked[i]))
            for i in order
            if masked[i] > 0.0
        ]


@lru_cache(maxsize=1)
def load_probes(probes_dir: str | None = None) -> tuple[AgentProbe, ToolProbe]:
    base = Path(probes_dir) if probes_dir else _PROBES_DIR
    agent_clf = joblib.load(base / AGENT_PROBE_FILE)
    tool_clf = joblib.load(base / TOOL_PROBE_FILE)
    return AgentProbe(agent_clf), ToolProbe(tool_clf, _tool_to_agent())
