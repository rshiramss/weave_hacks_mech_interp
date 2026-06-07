"""Local retrieval over tool-description embeddings (Stage 5, Arm 2 RAG).

Self-contained MiniLM baseline: encode all 200 tool descriptions from
`data/registry.json` with `all-MiniLM-L6-v2` (sentence-transformers, normalized),
then cosine top-k a query against them with in-memory NumPy. No Modal, no Volume,
no vector DB — just MiniLM embeddings + normalized cosine. The model and the tool
matrix are built once per process and cached in memory.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "data" / "registry.json"


@lru_cache(maxsize=1)
def _model():
    """Load the MiniLM encoder once per process."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _catalog() -> tuple[list[str], list[str], np.ndarray]:
    """Tool ids, their agent ids, and the normalized tool-description matrix."""
    registry = json.loads(_REGISTRY_PATH.read_text())
    tools = registry["tools"]
    tool_ids = [t["tool_id"] for t in tools]
    agent_ids = [t["agent_id"] for t in tools]
    descriptions = [t["description"] for t in tools]

    matrix = _model().encode(
        descriptions, normalize_embeddings=True, show_progress_bar=False
    )
    return tool_ids, agent_ids, np.asarray(matrix, dtype=np.float32)


def agent_of_tool() -> dict:
    """Map each tool_id to its owning agent_id."""
    tool_ids, agent_ids, _matrix = _catalog()
    return dict(zip(tool_ids, agent_ids))


def embed_query(text: str) -> np.ndarray:
    """Embed one query with the same normalized MiniLM model."""
    vector = _model().encode([text], normalize_embeddings=True)[0]
    return np.asarray(vector, dtype=np.float32)


def top_k_tools(query_text: str, k: int = TOP_K) -> list[tuple[str, float]]:
    """Cosine top-k (tool_id, score) for the query over the tool matrix.

    Both sides are L2-normalized, so the dot product is cosine similarity.
    """
    tool_ids, _agent_ids, matrix = _catalog()
    query_vec = embed_query(query_text)
    scores = matrix @ query_vec
    order = np.argsort(-scores)[:k]
    return [(tool_ids[i], float(scores[i])) for i in order]
