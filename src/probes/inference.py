"""Probe inference over the trained probes (Stage 2/4).

Loads the saved probes (data/probes/*.pkl) and exposes:
  AgentProbe.top_k(h, k=2)                  -> [(agent_id, score), ...]
  PerAgentToolProbe.top_k(h, agent_id, k=5) -> [(tool_id, score), ...]

Tool probes are PER-AGENT (5 × 40-class). The agent_id selects which probe runs;
each probe's classes_ are already scoped to that agent's tools, so there is no
masking step (unlike the retired single 200-class probe). Each tool probe is a
sklearn Pipeline(StandardScaler -> LogisticRegression): feed RAW h to predict_proba
and the pipeline scales internally. The agent probe is a bare LogisticRegression
on raw h. Filenames come from probe_config.json, not hardcoded here.

Requires scikit-learn >= 1.9 (the probes were pickled with 1.9.0).
"""

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

_PROBES_DIR = Path(__file__).resolve().parents[2] / "data" / "probes"
CONFIG_FILE = "probe_config.json"


class AgentProbe:
    def __init__(self, clf):
        self.clf = clf

    def top_k(self, h: np.ndarray, k: int = 2) -> list[tuple[str, float]]:
        proba = self.clf.predict_proba(np.asarray(h).reshape(1, -1))[0]
        order = np.argsort(-proba)[:k]
        return [(str(self.clf.classes_[i]), float(proba[i])) for i in order]


class PerAgentToolProbe:
    """Five probes keyed by agent_id; each scores only its agent's 40 tools."""

    def __init__(self, clfs: dict):
        self.clfs = clfs  # agent_id -> fitted Pipeline(StandardScaler, LogReg)

    def top_k(self, h: np.ndarray, agent_id: str,
              k: int = 5) -> list[tuple[str, float]]:
        clf = self.clfs.get(agent_id)
        if clf is None:  # unknown / empty agent -> no tools
            return []
        proba = clf.predict_proba(np.asarray(h).reshape(1, -1))[0]
        order = np.argsort(-proba)[:k]
        return [(str(clf.classes_[i]), float(proba[i])) for i in order]

    def pool_size(self, agent_id: str) -> int:
        """How many tool classes this agent's probe scores over (0 if unknown)."""
        clf = self.clfs.get(agent_id)
        return int(len(clf.classes_)) if clf is not None else 0


@lru_cache(maxsize=1)
def load_probes(probes_dir: str | None = None) -> tuple[AgentProbe, PerAgentToolProbe]:
    base = Path(probes_dir) if probes_dir else _PROBES_DIR
    config = json.loads((base / CONFIG_FILE).read_text())
    agent_clf = joblib.load(base / config["agent_probe_file"])
    clfs = {
        agent_id: joblib.load(base / filename)
        for agent_id, filename in config["tool_probe_files"].items()
    }
    return AgentProbe(agent_clf), PerAgentToolProbe(clfs)
