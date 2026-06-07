"""
One-time offline attention-head importance analysis for tool routing.

For Qwen2.5-7B-Instruct we measure how much each attention head contributes to a
trained layer-24 LogReg tool-routing probe. For every (layer, head) we zero out
that single head's contribution to the attention output (via a forward-pre hook
on the head's slice of o_proj's input), re-extract layer-24 hidden states, and
measure the mean drop in the probe's confidence for the correct tool_id versus a
no-hook baseline. Results are saved as a numpy array + heatmap.

This is a standalone analysis script. It is NOT part of the inference pipeline.

Run from the project root:
    python scripts/attention_head_importance.py
"""

import os
import gc
import json
import time
import pickle
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
PROBE_LAYER = 24

N_SAMPLES = 500
SEED = 42
BATCH_SIZE = 16
MAX_LENGTH = 128

# Paths are relative to the project root (same convention as probe_ab_test.py).
QUERIES_PATH = "data/queries_mixed.json"
REGISTRY_PATH = "data/registry.json"
CACHE_DIR = "data/probe_cache"
PROBE_PATH = os.path.join(CACHE_DIR, "attn_probe_Qwen2.5-7B-Instruct_layer24.pkl")
IMPORTANCE_PATH = "data/attention_importance.npy"
HEATMAP_PATH = "data/attention_heatmap.png"

PROBE_KW = dict(C=0.01, solver="lbfgs", max_iter=2000, class_weight="balanced")

HEATMAP_TITLE = (
    "Attention Head Importance for Tool Routing (Qwen2.5-7B Layer 24 Probe)"
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_tokenizer = None
_model = None


def load_data():
    """Return (queries, tool_labels) for a random 500-query sample."""
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    tool_to_agent = {t["tool_id"]: t["agent_id"] for t in registry["tools"]}

    with open(QUERIES_PATH) as f:
        raw = json.load(f)

    rng = random.Random(SEED)
    if len(raw) > N_SAMPLES:
        raw = rng.sample(raw, N_SAMPLES)

    queries = [r["query_text"] for r in raw]
    labels = [r["tool_id"] for r in raw]
    # Touch tool_to_agent so the registry mapping is built/validated as specified.
    _ = [tool_to_agent.get(t) for t in labels]
    return queries, np.array(labels)


def get_model():
    """Load Qwen2.5-7B-Instruct on cuda in fp16 (probe_ab_test.py pattern)."""
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        output_hidden_states=True,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
    if device == "cpu":
        model = model.to(device)

    _tokenizer, _model = tokenizer, model
    return tokenizer, model


# ---------------------------------------------------------------------------
# Hidden state extraction (layer 24, last token)
# ---------------------------------------------------------------------------

def extract_hidden_states(queries: list[str]) -> np.ndarray:
    import torch

    tokenizer, model = get_model()
    device = next(model.parameters()).device

    vectors = []
    for start in range(0, len(queries), BATCH_SIZE):
        batch = queries[start : start + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # output of transformer block PROBE_LAYER (0-indexed); index +1 because
        # hidden_states[0] is the embedding output (same as probe_ab_test.py).
        hs = outputs.hidden_states[PROBE_LAYER + 1]
        last_tok = hs[:, -1, :]  # left padding => final position is real token
        vectors.extend(last_tok.float().cpu().numpy())

    return np.stack(vectors)


# ---------------------------------------------------------------------------
# Attention-head ablation hook
# ---------------------------------------------------------------------------

def _make_head_zeroing_hook(head_idx: int, head_dim: int):
    """Forward-pre hook on o_proj that zeros one head's slice of its input.

    The input to o_proj is the concatenated per-head attention output of shape
    (batch, seq, num_heads * head_dim). Zeroing the head's contiguous slice
    removes that single head's contribution to the attention output.
    """
    lo = head_idx * head_dim
    hi = lo + head_dim

    def hook(module, args):
        hidden = args[0].clone()
        hidden[..., lo:hi] = 0
        return (hidden,) + tuple(args[1:])

    return hook


def get_attention_layers(model):
    """Return the list of decoder layers (model.model.layers)."""
    return model.model.layers


# ---------------------------------------------------------------------------
# Probe load / train
# ---------------------------------------------------------------------------

def correct_class_confidence(clf, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Probe probability assigned to each sample's true tool_id (NaN if unseen)."""
    proba = clf.predict_proba(X)
    class_to_idx = {c: i for i, c in enumerate(clf.classes_)}
    conf = np.full(len(labels), np.nan)
    for i, lab in enumerate(labels):
        j = class_to_idx.get(lab)
        if j is not None:
            conf[i] = proba[i, j]
    return conf


def load_or_train_probe(queries: list[str], labels: np.ndarray):
    from sklearn.linear_model import LogisticRegression

    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(PROBE_PATH):
        print(f"Loading cached probe from {PROBE_PATH}")
        with open(PROBE_PATH, "rb") as f:
            return pickle.load(f)

    print("No cached probe found; training a fresh layer-24 probe ...")
    from sklearn.model_selection import train_test_split

    X = extract_hidden_states(queries)

    idx = np.arange(len(labels))
    try:
        train_idx, _ = train_test_split(
            idx, test_size=0.2, random_state=SEED, stratify=labels
        )
    except ValueError:
        # Stratification fails when some tool_id has a single sample.
        train_idx, _ = train_test_split(idx, test_size=0.2, random_state=SEED)

    clf = LogisticRegression(**PROBE_KW)
    clf.fit(X[train_idx], labels[train_idx])

    with open(PROBE_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"Trained probe on {len(train_idx)} samples; cached to {PROBE_PATH}")
    return clf


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def save_heatmap(importance: np.ndarray, top_heads: list[tuple]):
    num_layers, num_heads = importance.shape
    fig, ax = plt.subplots(
        figsize=(max(8, num_heads * 0.45), max(6, num_layers * 0.4))
    )
    sns.heatmap(
        importance,
        cmap="coolwarm",
        center=0.0,
        ax=ax,
        cbar_kws={"label": "Mean drop in correct-class confidence"},
    )
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(HEATMAP_TITLE)
    ax.invert_yaxis()  # layer 0 at the bottom

    for layer, head, _ in top_heads[:5]:
        ax.add_patch(
            Rectangle((head, layer), 1, 1, fill=False, edgecolor="red", lw=2.5)
        )

    fig.tight_layout()
    fig.savefig(HEATMAP_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved heatmap to {HEATMAP_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import torch

    t0 = time.time()
    random.seed(SEED)
    np.random.seed(SEED)

    queries, labels = load_data()
    print(f"Loaded {len(queries)} queries (sampled, seed={SEED}).")

    clf = load_or_train_probe(queries, labels)

    _, model = get_model()
    layers = get_attention_layers(model)
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    print(
        f"Model: {num_layers} layers, {num_heads} heads/layer, head_dim={head_dim}"
    )

    print("Computing baseline (no ablation) confidences ...")
    baseline_X = extract_hidden_states(queries)
    baseline_conf = correct_class_confidence(clf, baseline_X, labels)

    # Ablating layers above PROBE_LAYER cannot affect layer-24 activations, so
    # only sweep layers 0..PROBE_LAYER.
    num_sweep_layers = PROBE_LAYER + 1

    importance = np.zeros((num_sweep_layers, num_heads), dtype=np.float64)

    for layer in range(num_sweep_layers):
        o_proj = layers[layer].self_attn.o_proj
        for head in range(num_heads):
            print(f"Layer {layer}/{num_sweep_layers - 1}, Head {head}/{num_heads - 1}")
            handle = o_proj.register_forward_pre_hook(
                _make_head_zeroing_hook(head, head_dim)
            )
            try:
                hooked_X = extract_hidden_states(queries)
                hooked_conf = correct_class_confidence(clf, hooked_X, labels)
                importance[layer, head] = float(
                    np.nanmean(baseline_conf - hooked_conf)
                )
            finally:
                handle.remove()

    np.save(IMPORTANCE_PATH, importance)
    print(f"Saved importance array {importance.shape} to {IMPORTANCE_PATH}")

    flat_order = np.argsort(-importance.flatten())
    ranked = [
        (int(i // num_heads), int(i % num_heads), float(importance.flatten()[i]))
        for i in flat_order
    ]

    save_heatmap(importance, ranked)

    print("\nTop 10 most important attention heads:")
    for layer, head, score in ranked[:10]:
        print(f"Layer {layer} Head {head}: score {score:.4f}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
