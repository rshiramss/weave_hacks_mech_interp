# Status + Probe Integration Plan

Companion to [agent_architecture.md](agent_architecture.md). Reflects the repo as
built through this session (Stages 3, 5, 6, 7, 8) and lays out exactly what to add
to land the linear probes (Stages 1–2) and the real router (Stage 4).

> **UPDATE — probes are trained.** Branch `abe/probe-training` ships
> `data/probes/orchestrator_probe.pkl` + `tool_probe.pkl` (Qwen2.5-7B, **layer 24**,
> raw-text last-token). Training is DONE — **§G below is the authoritative
> integration spec** and supersedes §D Steps 2–3. §C/§D's "extract + train from
> scratch" remain only as background.

> **PROGRESS (this session) — probes integrated + routing verified.**
> Critical path is largely landed:
> - **Stage 1 ✅** — `data/eval_holdout.jsonl` (50 rows, stratified from the
>   ground-truth `registry_ground.json`, 18k labeled queries) + `scripts/split_holdout.py`
>   + `scripts/validate_jsonl.py`. Published to Weave as `soc-routing-holdout`.
> - **Stage 2 ✅** — `.pkl`s pulled in, `scikit-learn` upgraded to 1.9.0 (was a hard
>   blocker), `src/probes/extract_h.py` (exact layer-24 raw-token contract),
>   `modal_app/extract_activations.py` (GPU extractor, deployed), `src/probes/inference.py`
>   (+ masking), `data/probes/probe_config.json`. **Validated: 5/5 agent top-1, 5/5
>   recall@2** on cross-agent probes via the live Modal extractor.
> - **Stage 4 ✅** — `src/probes/runtime.py` (@weave.op forward_pass/agent_probe/
>   tool_probe/probe_route), `src/routing/agent_picker.py`, `src/flow.py`
>   (`ProbeRoutedFlow`), `scripts/run_probe_router.py`; `probe` backend wired into
>   `src/routers.py` + `src/demo_pipeline.py`. Weave-op probe path verified
>   (fixed: weave-boxed args broke Modal deserialization → coerce to `str`).
> - **Stage 6 ✅** — 3-arm `weave.Evaluation` (incontext/rag/probe) + `wandb.Table`
>   ran over the 50-row holdout; all 3 runs live under `soc-routing-benchmark`.
>   (Fix: `incontext_router` now never raises mid-row — a failed LLM call had made
>   one row a Weave error-dict and broke output summarization.)
> - **Stage 7 ✅** — `scripts/make_leaderboard.py` published `soc-routing-leaderboard`
>   programmatically (routing_exact_match ↑, cost/latency ↓).
>
> **Probe-arm metrics (50-row holdout):** agent top-1 **0.96**, agent recall@2
> **1.00**, tool recall@5 **0.88**, routing-exact **0.82**. (Optimistic if the
> holdout overlaps probe training — see caveat above.)
>
> **Data source note:** `registry_ground.json` (root, 18k rows
> `{query_text, agent_id, tool_id, style, source}`, all registry-valid) is the
> labeled query set — use it for holdout/eval (and any re-extraction). Caveat: the
> probes may have trained on overlapping rows, so probe-arm metrics on a random
> holdout slice can be optimistic; for a clean number, hold out rows the probe
> never saw.

---

## A. What we HAVE vs DON'T HAVE (by stage)

| Stage | Status | Have | Missing |
|-------|--------|------|---------|
| **0 Foundation** | ⚠️ partial | `data/registry.json`, `src/probe/query_contract.py`, `.env.example`; `WEAVE_PROJECT` in `src/weave_setup.py` | `src/registry.py` loader, `src/config.py`, pinned `requirements.txt`/`pyproject.toml`, shared `RouteResult` types. (Today `src/tools/factory.py` reads `registry.json` directly.) |
| **1 Pilot data** | ⚠️ code only | `modal_app/generate_dataset.py` (written, deployable) | **Never run** → no `data/agent_probe_train.jsonl`, `tool_probe_train.jsonl`, `eval_holdout.jsonl`; no `scripts/split_holdout.py`, `scripts/validate_jsonl.py` |
| **2 Probes** | ✅ trained on `abe/probe-training` | `data/probes/orchestrator_probe.pkl` (5 agents), `tool_probe.pkl` (200 tools) — Qwen2.5-7B layer 24; `scripts/train_probes.py`, `scripts/probe_ab_test.py` | extraction-parity module + inference/masking wrapper to USE them (see **§G**); no `probe_config.json` |
| **3 Crews + tools** | ✅ done (verified) | `src/crews/specialist.py`, `src/tools/factory.py`, `src/config/{agents,tasks}.yaml`, `scripts/smoke_crew.py` | — |
| **4 Router E2E (Arm 3)** | ⚠️ stub only | `src/demo_pipeline.py` dispatcher (`stub`/`rag`/`incontext`), `src/weave_setup.py` | `src/flow.py` (`ProbeRoutedFlow`), `src/probes/runtime.py` (`@weave.op` forward_pass/agent_probe/tool_probe), `src/routing/agent_picker.py`, `scripts/run_probe_router.py` |
| **5 Baselines 1 & 2** | ✅ done (verified) | `src/baselines/*`, `modal_app/incontext_route.py`, `modal_app/precompute_embeddings.py`, `scripts/run_baseline.py` | — (deviation: Arm 1 runs **local Ollama Qwen2.5-7B**, not Modal→W&B-Inference; the Modal `incontext_route` app is deployed but bypassed) |
| **6 Holdout + Eval + Table** | ⚠️ code only | `scripts/publish_holdout.py`, `eval/scorers.py`, `src/routers.py`, `scripts/run_eval.py` | **Never run** → needs `data/eval_holdout.jsonl`; no published `soc-routing-holdout`, no eval runs, no `wandb.Table` |
| **7 Leaderboard + cost + compare** | ⚠️ partial | cost `add_cost` polish in `weave_setup.py`, `scripts/demo_comparison.py` | Leaderboard UI view (blocked on Stage 6 eval runs) |
| **8 Demo UI** | ✅ done (verified) | `src/formatters.py`, `src/demo_pipeline.py`, `app/server.py`, `scripts/demo_ui.py`, `README.md` | — |

---

## B. Incomplete phases (critical path order)

1. **Stage 1 data** — produce `eval_holdout.jsonl` (+ train JSONL). Unblocks Stage 6/7 *and* probe training.
2. **Stage 2 probes** — the missing core. Extract activations on a GPU, train two LogReg probes. ← largest piece.
3. **Stage 4 real router** — `ProbeRoutedFlow` + probe runtime + agent picker; flip `ROUTER_BACKEND=probe`.
4. **Stage 6 run** — publish holdout, run 3-arm eval (incontext, rag, **probe**), log Table.
5. **Stage 7 leaderboard** — configure the saved Weave leaderboard from the eval runs.
6. **Stage 0 cleanup** (low priority) — extract `src/registry.py` + `src/config.py`, pin deps.

Everything else (Stages 3, 5, 8) is done. Stages 6/7 code is done — they only need data + the probe arm.

---

## C. Hard constraint that shapes the plan

**W&B Serverless Inference returns text + token counts only — NO hidden states, NO logprobs**
(confirmed in `wandb-docs/pages/inference/api-reference.md`, `weave/quickstart-inference.md`).

➡️ Activations for probes **must** come from **local model weights** loaded with
`transformers` (`output_hidden_states=True`) on a **Modal GPU**. The W&B Inference
endpoint used for dataset generation / Arm-1 text cannot produce `h`.

**Backbone choice (confirmed by the trained probes):** **`Qwen/Qwen2.5-7B-Instruct`**
(open weights, not gated, consistent with crew/routing). `d_model = 3584` (matches
both `.pkl` `n_features_in_`). This deviates from architecture.md §2 (Llama-3.1-8B).

**The trained probes also pin the extraction contract — see §G.1.** It differs from
architecture.md §6.1 in two ways that MUST be honored: **layer 24** (not the final
layer) and **raw tokenization** (`tokenizer(text, max_length=128)`, last token —
NOT a chat template). Match exactly or accuracy collapses.

---

## D. Probe integration plan (step by step)

### Step 1 — Produce the dataset (Stage 1)
- Deploy + run the existing generator:
  `modal deploy modal_app/generate_dataset.py` then `modal run modal_app/generate_dataset.py`
  (pilot: 5 tools × 50 = 250 rows; scale later with `--no-pilot`).
- Add `scripts/split_holdout.py`: from `raw_dataset.jsonl`, take the `split=="test"`
  rows → write `data/eval_holdout.jsonl` as `{query_text, agent_id, tool_id}`.
- Add `scripts/validate_jsonl.py`: assert every row passes `validate_query()` and
  that `agent_id`/`tool_id` exist in `registry.json` (never invent ids).
- **Gate:** ≥20 holdout rows, disjoint from train.

### Step 2 — Extract activations on Modal GPU (Stage 2)
Single shared extractor so train == inference (architecture.md §6.1).

- `src/probes/extract_h.py` — pure function, stdlib + torch/transformers:
  ```python
  # one implementation imported by BOTH the Modal extractor and runtime inference
  input_ids = tokenizer.apply_chat_template(
      [{"role": "user", "content": normalize_query(query_text)}],
      tokenize=True, add_generation_prompt=False, return_tensors="pt")
  out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
  L = model.config.num_hidden_layers - 1
  h = out.hidden_states[L + 1][0, input_ids.shape[-1] - 1, :].float()  # [d_model]
  ```
- `modal_app/extract_activations.py` — Modal GPU class (patterns from
  `modal_docs/guide/{memory-snapshots,volumes,images,secrets}.md`):
  ```python
  image = (modal.Image.debian_slim(python_version="3.12")
           .pip_install("transformers[torch]", "numpy", "scikit-learn", "joblib")
           .add_local_python_source("src")
           .env({"HF_HOME": "/hf"}))
  hf_cache = modal.Volume.from_name("mi-agent-hf-cache", create_if_missing=True)
  data_vol = modal.Volume.from_name("probe-router-data", create_if_missing=True)

  @app.cls(gpu="A10G", image=image, volumes={"/hf": hf_cache, "/data": data_vol},
           enable_memory_snapshot=True, timeout=1800)
  class Extractor:
      @modal.enter()
      def load(self):
          self.model = AutoModelForCausalLM.from_pretrained(
              "Qwen/Qwen2.5-7B-Instruct", torch_dtype="bfloat16", device_map="cuda").eval()
          self.tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
      @modal.method()
      def extract(self, query_texts: list[str]) -> list:  # batched extract_h -> vectors
          ...
  ```
  - Read train JSONL from the Volume (or pass rows in), batch through `extract`,
    write `activations.npy` (`float32 [N, d_model]`) + a row index aligned to the
    JSONL, `data_vol.commit()`. (Return-then-single-writer or in-container write —
    see `generate_dataset.py` for the commit-limit caveat.)
  - First run downloads ~15 GB Qwen weights into `mi-agent-hf-cache` → cached after.
- **Sanity checks before training** (architecture.md §6.1): `h.shape[-1] == hidden_size`;
  `len(hidden_states) == num_hidden_layers + 1`; re-extract a query twice → identical `h`.

### Step 3 — Train the two probes (Stage 2)
- `scripts/train_probes.py`:
  - Load `activations.npy` + labels (agent_id, tool_id) aligned by row index.
  - `agent_probe` = `LogisticRegression(multinomial)` over 5 classes.
  - `tool_probe` = `LogisticRegression(multinomial)` over 200 classes (**train unmasked**).
  - Log to W&B (`wandb.log`, project `weavehacks-soc-probes`): `agent_recall_at_2`,
    `tool_recall_at_5`, top-1 accs (architecture.md §7.6).
  - Save `probes/agent_probe.joblib`, `probes/tool_probe.joblib`, `probes/probe_config.json`
    (model_id, layer_index, token strategy, `add_generation_prompt=false`, max_length,
    truncation_side, dtype, d_model).
  - **Gate:** pilot val agent recall@2 ≥ 0.80, tool recall@5 ≥ 0.50. If near-random,
    debug tokenization/chat-template parity FIRST, not the LogReg.
- `src/probes/inference.py`:
  ```python
  class AgentProbe:  # .top_k(h, k=2) -> [(agent_id, score), ...]
  class ToolProbe:   # .top_k_masked(h, agent_id, k=5): score 200 -> zero non-agent -> top5
  def load_probes(dir) -> (AgentProbe, ToolProbe)
  ```
  Masking (architecture.md §6.4): zero scores for tools whose `agent_id` ≠ chosen
  agent BEFORE argsort/top-5.

### Step 4 — Real router + Weave traces (Stage 4)
- `src/probes/runtime.py` — `@weave.op` wrappers `forward_pass(query)->h`,
  `agent_probe(h)->top2`, `tool_probe(h,agent_id)->top5`. **`forward_pass` runs the
  local Qwen** — either invoke the Modal `Extractor.extract.remote([query])` (keeps
  the 15 GB model off the laptop) or load locally if a GPU is present. Prefer the
  Modal call for parity with training.
- `src/routing/agent_picker.py` — `@weave.op(name="agent_picker")`; tiny prompt to the
  local Ollama Qwen (or W&B Inference) to pick 1 of the 2 shortlisted agents.
- `src/flow.py` — `ProbeRoutedFlow(Flow[RouterState])` (crew_AI_docs `concepts/flows.md`):
  `@start ingest → normalize_query`; `@listen route_and_execute`: forward_pass →
  agent_probe → agent_picker → tool_probe(masked) → `build_specialist_crew(agent, top5)`
  → `crew.kickoff(inputs={"context": query})`. `weave.init()` auto-traces the Flow +
  Crew subtree (wandb-docs `weave/guides/integrations/crewai.md`); the `@weave.op`
  wrappers nest the probe steps (`weave/tutorial-tracing_2.md`).
- `scripts/run_probe_router.py --query "..."` → prints route + result; confirm trace
  tree `ingest_event → forward_pass → agent_probe → agent_picker → tool_probe → crewai.*`.
- **Wire-in:** add a `probe` branch to `src/demo_pipeline._route()` and
  `src/routers.probe_router` that calls this Flow / probe inference. Then everything
  flips with `ROUTER_BACKEND=probe` — no UI changes (return shape already stable).

### Step 5 — Benchmark the probe arm (Stage 6) and leaderboard (Stage 7)
- `python scripts/publish_holdout.py` (needs Step 1 `eval_holdout.jsonl`).
- `python scripts/run_eval.py --arms incontext,rag,probe` — the `probe_router` adapter
  now returns `agent_candidates` (top-2) and `tool_candidates` (top-5), so
  `agent_recall_at_2` / `tool_recall_at_5` score on Arm 3 (they return `None` for 1–2).
- Configure the Weave leaderboard (architecture.md §7.3) from the
  `soc-routing-benchmark` runs: `routing_exact_match` ↑, `estimated_cost` ↓, `latency_ms` ↓.

---

## E. Deviations from architecture.md to reconcile

| Topic | architecture.md | As built / proposed | Action |
|-------|-----------------|---------------------|--------|
| Probe backbone | Llama-3.1-8B (gated) | **Qwen2.5-7B-Instruct** (open) | Use Qwen; note in `probe_config.json`. Keep train==inference. |
| Arm 1 LLM | Modal → W&B Inference | **local Ollama Qwen2.5-7B** | OK for hackathon; `incontext_route` Modal app remains available if you switch back. |
| Activations source | (implied via local model) | **must be local weights on Modal GPU** | W&B Inference can't emit hidden states — confirmed. |
| `d_model` / layers | 4096 / 32 (Llama) | **3584 / 28 (Qwen2.5-7B)** | Read from `model.config` at runtime; never hardcode. |
| Holdout | generated (Stage 1) | not yet produced | Run Stage 1 or hand-author (deferred earlier). |

---

## F. New files to add (summary)

```
scripts/split_holdout.py        # Stage 1 — test split -> eval_holdout.jsonl
scripts/validate_jsonl.py       # Stage 1 — registry-valid label check
src/probes/extract_h.py         # Stage 2 — shared forward-pass extractor
modal_app/extract_activations.py# Stage 2 — Modal GPU activation extraction
scripts/train_probes.py         # Stage 2 — LogReg x2 + wandb.log + joblibs
src/probes/inference.py         # Stage 2 — AgentProbe / ToolProbe (+ masking)
probes/{agent_probe,tool_probe}.joblib, probes/probe_config.json  # Stage 2 artifacts
src/probes/runtime.py           # Stage 4 — @weave.op forward_pass/agent_probe/tool_probe
src/routing/agent_picker.py     # Stage 4 — LLM picks 1 of 2
src/flow.py                     # Stage 4 — ProbeRoutedFlow
scripts/run_probe_router.py     # Stage 4 — CLI
# Stage 0 cleanup (optional): src/registry.py, src/config.py, requirements.txt
```

---

## G. Integrating the trained probes (branch `abe/probe-training`) — AUTHORITATIVE

The probes are **trained and saved**. Do NOT retrain. What remains is reproducing
the extraction at inference, loading the `.pkl`s, adding masking, and wiring Stage 4.

**Artifacts** (`data/probes/`):

| File | Probe | sklearn | classes | features |
|------|-------|---------|---------|----------|
| `orchestrator_probe.pkl` | agent | LogisticRegression | 5 `agent_id`s (sorted) | 3584 |
| `tool_probe.pkl` | tool | LogisticRegression | 200 `tool_id`s (sorted) | 3584 |

Classes match `registry.json` exactly (verified). Built by `scripts/train_probes.py`
from a cached layer-24 activation matrix; live extraction lives in
`scripts/probe_ab_test.py:extract_hidden_states`.

### G.1 Exact extraction contract (train == inference — non-negotiable)

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct` (LOCAL weights — W&B Inference can't emit hidden states) |
| **Layer** | **24** → `hidden_states[LAYER + 1]` = `hidden_states[25]` (index 0 = embeddings) |
| **Tokenization** | **raw** `tokenizer(text, truncation=True, max_length=128)` — **NO chat template, NO `add_generation_prompt`** |
| Padding | `padding_side="left"`; `pad_token = eos_token` if unset |
| **Token read** | **last position** `hs[:, -1, :]` (left-pad ⇒ last real token is the final index) |
| dtype | fp16 on GPU → `.float()` (fp32) for the probe |
| d_model | 3584 |

> ⚠️ Overrides architecture.md §6.1 (chat-template + final layer). The saved probes
> use **raw text + layer 24 + last token**. `probe_ab_test.py` defaults `LAYER=27`
> for its sweep, but `train_probes.py` loaded `..._layer24.npy` → production = **24**.

### G.2 Reference extraction (paste-ready)

```python
LAYER = 24
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.float16,
    output_hidden_states=True, device_map="auto").eval()

inputs = tok(normalize_query(query_text), return_tensors="pt",
             truncation=True, max_length=128).to(model.device)
with torch.no_grad():
    out = model(**inputs)
h = out.hidden_states[LAYER + 1][:, -1, :].float().cpu().numpy()[0]  # shape [3584]
```

### G.3 Inference wrapper — `src/probes/inference.py` (NEW)

```python
import joblib, numpy as np

class AgentProbe:
    def __init__(self, clf): self.clf = clf
    def top_k(self, h, k=2):
        p = self.clf.predict_proba(h.reshape(1, -1))[0]
        order = np.argsort(-p)[:k]
        return [(self.clf.classes_[i], float(p[i])) for i in order]

class ToolProbe:
    def __init__(self, clf, tool_to_agent): self.clf, self.t2a = clf, tool_to_agent
    def top_k_masked(self, h, agent_id, k=5):                 # §6.4 masking
        p = self.clf.predict_proba(h.reshape(1, -1))[0]
        masked = np.where(
            np.array([self.t2a.get(c) == agent_id for c in self.clf.classes_]), p, 0.0)
        order = np.argsort(-masked)[:k]
        return [(self.clf.classes_[i], float(masked[i])) for i in order if masked[i] > 0]

def load_probes(dir="data/probes", tool_to_agent=None):
    agent = AgentProbe(joblib.load(f"{dir}/orchestrator_probe.pkl"))
    tool = ToolProbe(joblib.load(f"{dir}/tool_probe.pkl"), tool_to_agent or {})
    return agent, tool
```

Masking is NOT in their scripts — add it here using `registry.json` `tool→agent`
(`{t["tool_id"]: t["agent_id"] for t in registry["tools"]}`).

### G.4 Where to run the forward pass

Qwen2.5-7B (~15 GB) shouldn't reload per query on a laptop. Preferred:

- **Modal GPU** — `@app.cls(gpu="A10G")` `Extractor`, `@modal.enter` loads the model
  once (+ memory snapshot), method runs the G.2 contract, returns the vector;
  `runtime.forward_pass` calls `Extractor.extract.remote([query])`. Keeps the model
  off the laptop and guarantees the same code path that built the cache.
- **Local** only if a GPU box is available. Either way: ONE `extract_h` used everywhere.

### G.5 Stage 4 wiring (now concrete)

- `src/probes/runtime.py`: `@weave.op` `forward_pass(query)->h` (Modal Extractor),
  `agent_probe(h)->top2`, `tool_probe(h, agent_id)->top5_masked`.
- `src/routing/agent_picker.py`: local Ollama Qwen picks 1 of the 2 shortlisted agents.
- `src/flow.py` `ProbeRoutedFlow` (auto-traced) + `scripts/run_probe_router.py`.
- Add a `probe` branch to `src/demo_pipeline._route()` **and** `src/routers.probe_router`,
  returning `agent_candidates` (top-2) + `tool_candidates` (top-5) so the Stage-6
  scorers (`agent_recall_at_2`, `tool_recall_at_5`) light up for Arm 3.
- Flip everything with `ROUTER_BACKEND=probe` — no UI change (return shape stable).

### G.6 Gotchas

- **sklearn version — HARD BLOCKER (not just a warning):** `.pkl`s were trained on
  **1.9.0**; loading under local **1.6.1** lets `joblib.load` succeed but
  `predict_proba` then **crashes**: `'LogisticRegression' object has no attribute
  'multi_class'`. The probes are unusable until you **`pip install "scikit-learn>=1.9.0"`**
  (match the training minor to be safe). Verify with a dummy `predict_proba` after upgrading.
- **No `probe_config.json`** shipped — create one recording the G.1 contract so
  inference can't silently drift from training.
- **Two eval harnesses exist.** Their `scripts/weave_evaluation.py` already builds
  `weave.Dataset("queries_mixed_v1")` + scorers (Naive / RAG / Probe / big-Qwen) with
  output `{predicted_agent, predicted_tool, token_count}`. Ours is `src/routers.py` +
  `scripts/run_eval.py` (`incontext`/`rag`/`probe`). **Pick ONE** to avoid divergent
  leaderboards — easiest is to adopt their `data/queries_mixed.json` as our holdout
  and run it through our `ROUTERS`.
- **Data not committed:** `data/queries_mixed.json`, `queries_nl.json`,
  `data/probe_cache/*.npy` are absent on the branch — needed to re-extract/retrain,
  NOT to run inference with the saved `.pkl`s. Grab `queries_mixed.json` if you want
  it as the eval holdout (it has `{query_text, agent_id, tool_id}` — same schema).

### G.7 Pull the artifacts onto the agent branch

```bash
git checkout origin/abe/probe-training -- data/probes/orchestrator_probe.pkl \
                                          data/probes/tool_probe.pkl
# optional reference scripts:
git checkout origin/abe/probe-training -- scripts/train_probes.py scripts/probe_ab_test.py
```

Confirm `.gitignore` doesn't drop `*.pkl` / `data/probes/`.
