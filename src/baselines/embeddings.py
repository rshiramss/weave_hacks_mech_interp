"""Local retrieval over Modal-precomputed tool embeddings (Stage 5 hybrid, Arm 2).

Tool vectors are computed once on Modal GPU (modal_app/precompute_embeddings.py,
BAAI/bge-small-en-v1.5) and stored in the probe-router-data Volume as
`tool_embeddings.npz` + `tool_ids.json`. This module syncs them to data/cache/
once, embeds the query via the same Modal model, and does cosine top-k locally —
so sentence-transformers / PyTorch never run on the laptop.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

VOLUME_NAME = "probe-router-data"
EMB_APP = "baseline-embeddings"
EMB_FILE = "tool_embeddings.npz"
IDS_FILE = "tool_ids.json"

_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


def _ensure_local() -> tuple[Path, Path]:
    """Sync embeddings + ids from the Modal Volume into data/cache/ (once)."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    emb_path, ids_path = _CACHE_DIR / EMB_FILE, _CACHE_DIR / IDS_FILE
    if emb_path.exists() and ids_path.exists():
        return emb_path, ids_path

    import modal

    vol = modal.Volume.from_name(VOLUME_NAME)
    try:
        if not emb_path.exists():
            emb_path.write_bytes(b"".join(vol.read_file(EMB_FILE)))
        if not ids_path.exists():
            ids_path.write_text(b"".join(vol.read_file(IDS_FILE)).decode())
    except Exception as exc:  # noqa: BLE001 — surface the real cause
        raise RuntimeError(
            f"No embeddings in volume '{VOLUME_NAME}'. Run "
            "`modal run modal_app/precompute_embeddings.py` first."
        ) from exc
    return emb_path, ids_path


@lru_cache(maxsize=1)
def _load() -> tuple[list[str], list[str], np.ndarray]:
    emb_path, ids_path = _ensure_local()
    with np.load(emb_path) as data:
        matrix = data["embeddings"]
    meta = json.loads(ids_path.read_text())
    return meta["tool_ids"], meta["agent_ids"], matrix


def agent_of_tool() -> dict:
    tool_ids, agent_ids, _matrix = _load()
    return dict(zip(tool_ids, agent_ids))


def embed_query(text: str) -> np.ndarray:
    """Embed one query via the Modal model (keeps ST off the laptop)."""
    import modal

    embedder = modal.Cls.from_name(EMB_APP, "ToolEmbedder")()
    vector = embedder.encode.remote([text])[0]
    return np.asarray(vector, dtype=np.float32)


def top_k_tools(query_text: str, k: int = 5) -> list[tuple[str, float]]:
    """Cosine top-k tool_ids for the query over the cached matrix."""
    tool_ids, _agent_ids, matrix = _load()
    query_vec = embed_query(query_text)
    scores = matrix @ query_vec
    order = np.argsort(-scores)[:k]
    return [(tool_ids[i], float(scores[i])) for i in order]
