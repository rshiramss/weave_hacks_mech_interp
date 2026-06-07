"""Modal GPU hidden-state extractor for the probe router (Stage 2/4).

Loads frozen Qwen2.5-7B once per container and returns residual-stream vectors
using the exact training contract (src/probes/extract_h.compute_h). Used both for
batch activation extraction and for single-query runtime forward passes — the
laptop never loads the 15 GB model.

Deploy once (so the runtime can call it):
    modal deploy modal_app/extract_activations.py
Smoke test:
    modal run modal_app/extract_activations.py --query "pull the failed auth logs for host web-03"
Batch extract a labeled dataset -> activations.npy on the data Volume:
    modal run modal_app/extract_activations.py::batch --dataset data/registry_clean.jsonl
"""

import json
from pathlib import Path

import modal

from src.probes.extract_h import MODEL_ID

APP_NAME = "probe-extractor"
HF_CACHE_VOLUME = "mi-agent-hf-cache"  # reuse existing HF weight cache
DATA_VOLUME = "probe-router-data"      # shared with the dataset generator

# Default artifact names on the data Volume (consumed by scripts/train_probes.py).
DEFAULT_ACTIVATIONS = "activations_layer24.npy"
DEFAULT_LABELS = "activation_labels.jsonl"

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)
data_vol = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("transformers>=4.44", "torch", "accelerate", "numpy")
    .env({"HF_HOME": "/hf"})
    .add_local_python_source("src")  # keep local source as the LAST image layer
)


@app.cls(image=image, gpu="A10G", volumes={"/hf": hf_cache, "/data": data_vol},
         timeout=1800, scaledown_window=300)
class Extractor:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            output_hidden_states=True,
            device_map="cuda",
        ).eval()
        hf_cache.commit()  # persist downloaded weights for next cold start

    @modal.method()
    def extract(self, query_texts: list) -> list:
        """Single/few-query forward pass for the runtime router (returns vectors)."""
        from src.probes.extract_h import compute_h

        return compute_h(self.model, self.tokenizer, query_texts).tolist()

    @modal.method()
    def extract_dataset(self, rows: list, activations_name: str,
                        labels_name: str) -> dict:
        """Batch-extract h for every row and write npy + aligned labels to /data.

        `rows` are the labeled records ({query_text, agent_id, tool_id, ...}).
        The activation matrix and the labels JSONL share the SAME row order, so
        the trainer can zip them by index without a join key. Writing inside the
        container (not returning the matrix) avoids shipping ~250 MB back.
        """
        import numpy as np

        from src.probes.extract_h import compute_h

        queries = [str(row["query_text"]) for row in rows]
        vectors = compute_h(self.model, self.tokenizer, queries)  # [N, 3584] f32

        np.save(f"/data/{activations_name}", vectors)
        with open(f"/data/{labels_name}", "w") as labels_file:
            for row in rows:
                labels_file.write(json.dumps(row) + "\n")
        data_vol.commit()

        return {
            "rows": len(rows),
            "shape": list(vectors.shape),
            "activations": activations_name,
            "labels": labels_name,
        }


def _load_rows(path: Path) -> list:
    """Load a JSON array or JSONL file into a list of row dicts (local helper)."""
    text = path.read_text()
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _download_from_volume(name: str, dest: Path) -> None:
    """Pull a Volume file to a local path so the CPU trainer can read it."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as out:
        for chunk in data_vol.read_file(name):
            out.write(chunk)


@app.local_entrypoint()
def main(query: str = "pull the failed auth logs for host web-03"):
    vectors = Extractor().extract.remote([query])
    import numpy as np

    arr = np.asarray(vectors, dtype="float32")
    print(f"[extractor] query={query!r}")
    print(f"[extractor] h shape={arr.shape}  (expect [1, 3584])")
    print(f"[extractor] h[:5]={arr[0, :5]}")


@app.local_entrypoint()
def batch(
    dataset: str = "data/registry_clean.jsonl",
    activations: str = DEFAULT_ACTIVATIONS,
    labels: str = DEFAULT_LABELS,
    download: bool = True,
):
    """Extract activations for a cleaned dataset and stage them for training.

    Reads the labeled JSONL/JSON locally, ships the queries to the GPU container,
    writes `activations.npy` + `labels.jsonl` to the `probe-router-data` Volume,
    and (by default) downloads both back so scripts/train_probes.py can load them.
    """
    rows = _load_rows(Path(dataset))
    print(f"[batch] dataset={dataset} rows={len(rows)} -> extracting on A10G...")

    result = Extractor().extract_dataset.remote(rows, activations, labels)
    print(f"[batch] wrote {result['shape']} to volume '{DATA_VOLUME}': "
          f"{result['activations']}, {result['labels']}")

    if download:
        for name in (activations, labels):
            dest = Path("data") / name
            _download_from_volume(name, dest)
            print(f"[batch] downloaded {name} -> {dest}")
        print("[batch] ready: python scripts/train_probes.py "
              f"--activations data/{activations} --labels data/{labels}")
