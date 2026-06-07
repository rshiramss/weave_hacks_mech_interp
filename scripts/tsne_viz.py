"""Standalone t-SNE visualization of cached probe hidden states (no GPU).

Compares layer-8 vs layer-24 hidden-state geometry for tool intent, colored by
tool_id, and annotates each panel with a silhouette score.
"""

import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

QUERIES_PATH = "data/queries_mixed.json"
CACHE_TMPL = "data/probe_cache/mixed_Qwen2.5-7B-Instruct_layer{}.npy"
OUT_PATH = "data/tsne_comparison.png"
SEED = 42

with open(QUERIES_PATH) as f:
    labels = np.array([r["tool_id"] for r in json.load(f)])

X24 = np.load(CACHE_TMPL.format(24))
if X24.shape[0] != len(labels):
    print(f"ERROR: cache rows ({X24.shape[0]}) != query rows ({len(labels)}).")
    sys.exit(1)

layers = []
try:
    X8 = np.load(CACHE_TMPL.format(8))
    if X8.shape[0] == len(labels):
        layers.append((8, X8))
    else:
        print(f"Warning: layer 8 cache row mismatch; skipping layer 8.")
except FileNotFoundError:
    print("Layer 8 cache not found; plotting layer 24 only.")
layers.append((24, X24))

# Stable integer color index per tool_id for a qualitative colormap.
classes = sorted(set(labels))
class_to_int = {c: i for i, c in enumerate(classes)}
color_idx = np.array([class_to_int[l] for l in labels])
cmap = plt.get_cmap("gist_ncar", len(classes))

rng = np.random.default_rng(SEED)
sub = rng.choice(len(labels), size=min(2000, len(labels)), replace=False)

fig, axes = plt.subplots(1, len(layers), figsize=(7 * len(layers), 7), squeeze=False)
scores = {}
for ax, (layer, X) in zip(axes[0], layers):
    emb = TSNE(n_components=2, perplexity=30, random_state=SEED).fit_transform(X)
    ax.scatter(emb[:, 0], emb[:, 1], c=color_idx, cmap=cmap, s=6, alpha=0.7)
    sil = silhouette_score(emb[sub], labels[sub])
    scores[layer] = sil
    ax.set_title(f"Layer {layer} \u2014 Hidden State Space")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.02, 0.98, f"silhouette = {sil:.3f}",
        transform=ax.transAxes, va="top", ha="left",
        bbox=dict(boxstyle="round", fc="white", alpha=0.7),
    )

fig.suptitle("Tool Intent Geometry in Qwen2.5-7B", fontsize=15)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150)
print(f"Saved {OUT_PATH}")
for layer in sorted(scores):
    print(f"Layer {layer} silhouette score: {scores[layer]:.4f}")
