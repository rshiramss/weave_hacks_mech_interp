"""Modal GPU hidden-state extractor for the probe router (Stage 2/4).

Loads frozen Qwen2.5-7B once per container and returns residual-stream vectors
using the exact training contract (src/probes/extract_h.compute_h). Used both for
batch activation extraction and for single-query runtime forward passes — the
laptop never loads the 15 GB model.

Deploy once (so the runtime can call it):
    modal deploy modal_app/extract_activations.py
Smoke test:
    modal run modal_app/extract_activations.py --query "pull the failed auth logs for host web-03"
"""

import modal

from src.probes.extract_h import MODEL_ID

APP_NAME = "probe-extractor"
HF_CACHE_VOLUME = "mi-agent-hf-cache"  # reuse existing HF weight cache

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("transformers>=4.44", "torch", "accelerate", "numpy")
    .env({"HF_HOME": "/hf"})
    .add_local_python_source("src")  # keep local source as the LAST image layer
)


@app.cls(image=image, gpu="A10G", volumes={"/hf": hf_cache}, timeout=1800,
         scaledown_window=300)
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
        from src.probes.extract_h import compute_h

        return compute_h(self.model, self.tokenizer, query_texts).tolist()


@app.local_entrypoint()
def main(query: str = "pull the failed auth logs for host web-03"):
    vectors = Extractor().extract.remote([query])
    import numpy as np

    arr = np.asarray(vectors, dtype="float32")
    print(f"[extractor] query={query!r}")
    print(f"[extractor] h shape={arr.shape}  (expect [1, 3584])")
    print(f"[extractor] h[:5]={arr[0, :5]}")
