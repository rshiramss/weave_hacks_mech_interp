"""Hidden-state extraction contract for the trained probes (Stage 2/4).

PARITY-CRITICAL: this MUST match how the saved probes in data/probes/*.pkl were
trained (scripts/probe_ab_test.py:extract_hidden_states on branch
abe/probe-training). Any drift here silently wrecks routing accuracy.

Contract (see PROBE_PLAN.md §G.1):
  - model      Qwen/Qwen2.5-7B-Instruct (local weights; W&B Inference can't emit
               hidden states)
  - layer      24  -> hidden_states[LAYER + 1] (index 0 = embeddings)
  - tokenize   RAW text, truncation, max_length=128 (NO chat template)
  - padding    left; pad_token = eos if unset
  - token      last position hs[:, -1, :]  (left-pad => last real token)
  - dtype      fp16 forward -> float32 features; d_model = 3584

`compute_h` is pure torch and runs inside the Modal GPU container.
`extract_h_remote` is the laptop-side accessor that calls that container.
"""

import numpy as np

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 24
MAX_LENGTH = 128
D_MODEL = 3584
BATCH_SIZE = 16
EXTRACTOR_APP = "probe-extractor"


def compute_h(model, tokenizer, query_texts: list[str],
              layer: int = LAYER, max_length: int = MAX_LENGTH) -> np.ndarray:
    """Residual-stream vectors at (layer, last-token) for each query. [N, d_model].

    Requires the model loaded with output_hidden_states=True and the tokenizer
    with padding_side='left' and a pad_token set.
    """
    import torch

    device = next(model.parameters()).device
    vectors = []
    for start in range(0, len(query_texts), BATCH_SIZE):
        batch = query_texts[start:start + BATCH_SIZE]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        hidden = outputs.hidden_states[layer + 1]  # [B, seq, H]
        last_token = hidden[:, -1, :]              # left-pad => final index is real
        vectors.extend(last_token.float().cpu().numpy())
    return np.stack(vectors)


def extract_h_remote(query_texts: list[str]) -> np.ndarray:
    """Laptop-side: run the forward pass on the deployed Modal GPU extractor."""
    import modal

    # Coerce to plain str: when called inside a @weave.op the args are weave-boxed
    # subclasses, which the Modal container (no weave installed) can't deserialize.
    texts = [str(q) for q in query_texts]
    extractor = modal.Cls.from_name(EXTRACTOR_APP, "Extractor")()
    vectors = extractor.extract.remote(texts)
    return np.asarray(vectors, dtype=np.float32)
