"""Arm 2 tool-description embeddings on Modal GPU (Stage 5 hybrid, one-time).

A GPU container (sentence-transformers, BAAI/bge-small-en-v1.5) encodes all 200
tool descriptions and writes `tool_embeddings.npz` + `tool_ids.json` to the
probe-router-data Volume. Retrieval (cosine top-k) runs LOCALLY over the cached
matrix; query embedding reuses `ToolEmbedder.encode` so sentence-transformers /
PyTorch never load on the laptop.

Deploy (so the local adapter can call encode) + precompute:
    modal deploy modal_app/precompute_embeddings.py
    modal run    modal_app/precompute_embeddings.py     # writes vectors to Volume
"""

import json
import sys
from pathlib import Path

import modal

APP_NAME = "baseline-embeddings"
VOLUME_NAME = "probe-router-data"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMB_FILE = "tool_embeddings.npz"
IDS_FILE = "tool_ids.json"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "sentence-transformers>=3.0", "numpy"
)


@app.cls(image=image, gpu="T4", volumes={"/data": vol}, timeout=600)
class ToolEmbedder:
    @modal.enter()
    def load(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(EMBED_MODEL)

    @modal.method()
    def encode(self, texts: list) -> list:
        """Encode arbitrary texts (e.g. the query). Returns normalized vectors."""
        import numpy as np

        vectors = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32).tolist()

    @modal.method()
    def build(self, tool_texts: list, tool_ids: list, agent_ids: list) -> dict:
        """Encode the tool corpus and persist vectors + ids to the Volume."""
        import numpy as np

        vectors = self.model.encode(
            tool_texts, normalize_embeddings=True, batch_size=64
        )
        vectors = np.asarray(vectors, dtype=np.float32)

        np.savez(f"/data/{EMB_FILE}", embeddings=vectors)
        Path(f"/data/{IDS_FILE}").write_text(
            json.dumps(
                {
                    "tool_ids": tool_ids,
                    "agent_ids": agent_ids,
                    "model": EMBED_MODEL,
                    "dim": int(vectors.shape[1]),
                }
            )
        )
        vol.commit()
        return {"count": int(vectors.shape[0]), "dim": int(vectors.shape[1])}


@app.local_entrypoint()
def main():
    """Build the tool corpus locally from the registry, encode on Modal GPU."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.tools.factory import _registry

    tools = _registry()["tools"]
    texts = [
        f"{t['tool_id'].replace('_', ' ')} {t.get('name', '')} "
        f"{t.get('description', '')}"
        for t in tools
    ]
    tool_ids = [t["tool_id"] for t in tools]
    agent_ids = [t["agent_id"] for t in tools]

    info = ToolEmbedder().build.remote(texts, tool_ids, agent_ids)
    print(
        f"[precompute] wrote {info['count']} vectors dim={info['dim']} "
        f"to volume '{VOLUME_NAME}' ({EMB_FILE}, {IDS_FILE})"
    )
