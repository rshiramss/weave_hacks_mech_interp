# Synthetic Data & Probe Training Plan

**Architecture:** Single-hop probe routing (Qwen 1.5B, **last layer only**, **last token**)  
**Goal:** Train agent probe + tool probe from synthetic **natural-language** SOC queries with known ground-truth labels.

**Activation choice (v1):** One forward pass → take the **residual stream at the final transformer layer**, at the **last input token**. No layer sweep, no mean pooling, no multi-layer concat.

**Data generation (v1):** **No templates, no structured ticket format.** Work items come from `registry.json` only. A **Modal** app calls an LLM in parallel to write varied **natural-language** analyst queries; labels are assigned by construction (one `(agent_id, tool_id)` pair per generation job).

---

## Overview

```
registry.json              → 5 agents, 200 tools (40 per agent)
        ↓
modal/generate_dataset.py  → LLM writes natural-language query_text per tool (parallel .map / .starmap)
        ↓
Modal Volume               → raw rows persisted under /data
        ↓
agent_probe_train.jsonl    (query_text + agent label)
tool_probe_train.jsonl     (query_text + tool label)   ← same rows, different label
        ↓
modal/extract_activations.py → forward pass → hidden states h
        ↓
train_probes.py            → LogReg(h → agent), LogReg(h → tool)
```

**One training set, two probes.** Same `query_text`, same hidden state `h`, different label columns.

---

## Step 1: Build the registry

**File:** `data/registry.json`

Define every routable agent and tool. Each tool must have an `agent_id` (used for tool-probe masking at inference). This file is the **only** catalog needed for generation — no separate template files.

```json
{
  "agents": [
    {
      "agent_id": "log_search",
      "name": "SIEM Log Search Specialist",
      "description": "Queries centralized logs and correlates security events."
    }
  ],
  "tools": [
    {
      "tool_id": "search_auth_logs",
      "agent_id": "log_search",
      "name": "Search Authentication Logs",
      "description": "Query SIEM for login and MFA events.",
      "integration": "mock"
    }
  ]
}
```

**Target scale:**

- **5 sub-agents**
- **200 tools total** (40 per agent)
- All tools `"integration": "mock"` at execution time; registry descriptions still drive LLM generation quality

**Info you need per agent:** `agent_id`, short name, one-line description  
**Info you need per tool:** `tool_id`, `agent_id`, name, one-line description

---

## Step 2: Natural-language query contract

**Critical:** Train and inference must use the **same kind of input** — plain natural language, what an analyst would type into the agent. No tool schemas in the probe forward pass. No rigid `NEW SOC EVENT` / labeled-field templates.

`query_text` **is** the user message — store the LLM output directly:

```python
def normalize_query(text: str) -> str:
    """Light cleanup only. Do not reformat into structured tickets."""
    return text.strip()
```

Use the same string everywhere: Modal generator, probe training, probe inference, CrewAI `{context}`.

**Good examples of `query_text`:**

- "We're seeing 847 failed SSH logins to prod-bastion-01 from 203.0.113.44 in the last 15 minutes, mostly root — can someone pull auth logs?"
- "EDR flagged a suspicious exe on finance-ws-07, hash 3f8a9c…, parent was outlook.exe — need this checked out."
- "User reported a phishing email from paypa1-update.com with subject 'Urgent: Password Reset Required'."

**Avoid:**

- Labeled fields (`Type:`, `Severity:`, `Indicators:`)
- JSON blobs in `query_text`
- A shared formatter that wraps NL into a ticket schema

---

## Step 3: Registry-driven generation spec (no templates)

Each training row is produced by a **generation job** keyed to one tool:


| Field            | Source                                      |
| ---------------- | ------------------------------------------- |
| `agent_id` label | `tool.agent_id` from registry               |
| `tool_id` label  | `tool.tool_id` from registry                |
| `query_text`     | LLM output — natural-language analyst query |


**Prompt pattern (per job):** give the LLM the target agent name/description, target tool name/description, and ask for a **realistic message a tier-1 analyst would type** that requires **that specific tool**. The label is fixed before generation — we are not asking the model to classify.

```text
Write ONE short SOC analyst message (1–4 sentences) that a human would type into a triage agent.

The message MUST require this specialist tool:
  agent: {agent_name} — {agent_description}
  tool: {tool_name} — {tool_description}

Rules:
- Natural language only — write like a person, not a ticket template
- Include concrete details (IPs, hostnames, hashes, users) where relevant
- Use realistic RFC 5737 IPs; vary wording and scenario
- Do NOT use labeled fields (Type:, Severity:, etc.) or JSON
- Return only the message text, no markdown fences
```

**Work list construction (local, before Modal):**

```python
import json
from itertools import repeat

registry = json.load(open("data/registry.json"))
EXAMPLES_PER_TOOL = 75  # pilot: 50; full scale: 50–100

jobs = [
    (tool["tool_id"], tool["agent_id"], tool["name"], tool["description"], i)
    for tool in registry["tools"]
    for i in range(EXAMPLES_PER_TOOL)
]
# len(jobs) == 200 * 75 == 15_000 at full scale
# pilot: filter to 5 tools × 50 == 250 jobs
```

**Hard negatives:** the LLM naturally varies wording across tools. Optionally pass `seed` / example index in the prompt so parallel jobs diverge. Reuse indicator values across *different* tools in post-processing only if eval needs it — not required for v1.

**Sizing:**

- 50–100 examples **per tool**
- Full scale: 200 × 75 ≈ **15,000 rows**
- Pilot: **5 tools × 50 = 250 rows** (one tool per agent)

---

## Step 4: Generate JSONL on Modal

Generation runs on Modal: parallel containers, GPU optional (depends on whether you use a local HF model or an API). Pattern follows `modal_docs/pages/docs/guide/scale.md` (`.map()` / `.starmap()`) and `modal_docs/pages/docs/guide/volumes.md` (persist output).

### Output schema

**Probe training (minimal):**

```json
{"query_text": "We're seeing 847 failed SSH logins to prod-bastion-01 from 203.0.113.44 in the last 15 minutes — can someone check auth logs?", "label": "log_search"}
```

**Raw row written by generator (before split into two JSONL files):**

```json
{
  "query_text": "We're seeing 847 failed SSH logins to prod-bastion-01 from 203.0.113.44 in the last 15 minutes — can someone check auth logs?",
  "agent_id": "log_search",
  "tool_id": "search_auth_logs",
  "split": "train"
}
```

Derive the two probe files locally after download:

```python
# agent_probe_train.jsonl
{"query_text": row["query_text"], "label": row["agent_id"]}
# tool_probe_train.jsonl
{"query_text": row["query_text"], "label": row["tool_id"]}
```

Split: 80% train / 10% val / 10% test, stratified by `tool_id`.

### Modal app sketch (`modal/generate_dataset.py`)

```python
import json
from pathlib import Path

import modal

app = modal.App("probe-router-generate")
vol = modal.Volume.from_name("probe-router-data", create_if_missing=True)
DATA_DIR = Path("/data")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pydantic", "httpx")  # add transformers if running HF on GPU
)

# Option A: @app.cls + GPU — load Qwen (or other) once per container
# See modal_docs/pages/docs/guide/developing-with-llms.md (@modal.enter)
#
# Option B: @app.function + Secret — call OpenAI/Anthropic API for generation
# See modal_docs/pages/docs/reference/modal.Secret.md


@app.function(
    image=image,
    volumes={DATA_DIR: vol},
    timeout=600,
    max_containers=50,  # cap parallel spend; see modal_docs/pages/docs/guide/scale.md
)
def generate_one(
    tool_id: str,
    agent_id: str,
    tool_name: str,
    tool_description: str,
    agent_name: str,
    agent_description: str,
    example_idx: int,
) -> dict:
    """One natural-language query for one tool. Label is fixed by job args, not model classification."""
    # ... call LLM with prompt from Step 3 ...
    query_text = normalize_query(...)  # LLM response text
    row = {
        "query_text": query_text,
        "agent_id": agent_id,
        "tool_id": tool_id,
        "example_idx": example_idx,
    }
    out = DATA_DIR / "raw" / f"{tool_id}_{example_idx:04d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row))
    vol.commit()  # persist Volume writes — modal_docs/pages/docs/guide/volumes.md
    return row


@app.local_entrypoint()
def main(examples_per_tool: int = 50, pilot: bool = True):
    registry = json.loads(Path("data/registry.json").read_text())
    agent_by_id = {a["agent_id"]: a for a in registry["agents"]}

    tools = registry["tools"]
    if pilot:
        seen = set()
        tools = [t for t in tools if t["agent_id"] not in seen and not seen.add(t["agent_id"])]

    jobs = [
        (
            t["tool_id"],
            t["agent_id"],
            t["name"],
            t["description"],
            agent_by_id[t["agent_id"]]["name"],
            agent_by_id[t["agent_id"]]["description"],
            i,
        )
        for t in tools
        for i in range(examples_per_tool)
    ]

    # Parallel generation — modal_docs/pages/docs/guide/scale.md#parallel-execution-of-inputs
    rows = list(generate_one.starmap(jobs, return_exceptions=True))

    # Merge raw JSON → JSONL on local disk (or second Modal function)
    write_probe_jsonl(rows, "data/agent_probe_train.jsonl", label_key="agent_id")
    write_probe_jsonl(rows, "data/tool_probe_train.jsonl", label_key="tool_id")
```

### Run commands

```bash
modal run modal/generate_dataset.py              # pilot: 5 tools × 50
modal run modal/generate_dataset.py --no-pilot   # all 200 tools
modal volume ls probe-router-data                # inspect Volume
```

### Validation before training

Reject rows where:

- `query_text` is empty or under ~20 characters
- Output looks like JSON or labeled ticket fields (`Type:`, `Severity:`, `NEW SOC EVENT`)
- Duplicate `(query_text, tool_id)` exact matches (keep one)

Log rejection rate in Weave or stdout; retry failed jobs with `.starmap(..., return_exceptions=True)` and filter exceptions.

### Example rows

`**data/agent_probe_train.jsonl**`

```json
{"query_text": "We're seeing 847 failed SSH logins to prod-bastion-01 from 203.0.113.44 in the last 15 minutes, mostly root — can someone pull auth logs?", "label": "log_search"}
```

**`data/tool_probe_train.jsonl`** (same `query_text`)

```json
{"query_text": "We're seeing 847 failed SSH logins to prod-bastion-01 from 203.0.113.44 in the last 15 minutes, mostly root — can someone pull auth logs?", "label": "search_auth_logs"}
```

---

## Step 5: Extract hidden states (Modal)

**Input:** `query_text` from JSONL  
**Model:** Qwen 1.5B (frozen, no generation)

Run on Modal with the same Volume or a dedicated `modal/extract_activations.py` app: `@app.cls(gpu="T4")` loads the model once per container, `.map()` over all `query_text` rows, writes `h` vectors to `/data/activations/`.


| Approach                   | v1?   | Notes                                      |
| -------------------------- | ----- | ------------------------------------------ |
| **Last layer, last token** | ✅ Yes | `h = residual[L, -1]`                      |
| Layer sweep                | ❌ No  | Research later                             |
| Mean pooling               | ❌ No  | Last token sees full context via attention |


```python
L = model.config.num_hidden_layers - 1
h = model.forward(row["query_text"], layer=L, token=-1)
```

Store `[N, d_model]` aligned with labels; `vol.commit()` after each batch.

---

## Step 6: Train both probes

Same activations, different labels. Can run locally or as a lightweight Modal function (CPU-only is fine for LogReg).

```python
from sklearn.linear_model import LogisticRegression

agent_probe = LogisticRegression(max_iter=1000, multi_class="multinomial")
agent_probe.fit(X_train, y_agent_train)

tool_probe = LogisticRegression(max_iter=1000, multi_class="multinomial")
tool_probe.fit(X_train, y_tool_train)
```

**Metrics on val/test:**

- Agent probe: top-1 accuracy, **recall@2** (5-way)
- Tool probe: top-1 accuracy, **recall@5** (200-way train; **40-way** masked at inference)

Save: `probes/agent_probe.joblib`, `probes/tool_probe.joblib`, `probes/probe_config.json`.

---

## Step 7: Inference (how probes are used)

```
query_text
    → forward pass → h
    → agent_probe: top-2 agents
    → picker: top-1 (or LLM over 2)
    → tool_probe: score 200, mask to 40, top-5
    → specialist crew: kickoff(inputs={"context": query_text}), tools=top_5
```

Training is **unmasked** on all tools. Masking only at inference.

---

## Build order (hackathon)


| Step | Task                                          | Done when               |
| ---- | --------------------------------------------- | ----------------------- |
| 1    | Registry with 5 agents + 40 tools each        | `registry.json`         |
| 2    | Lock NL query contract (`normalize_query`)    | function in repo        |
| 3    | Generation prompt + job list from registry    | spec reviewed           |
| 4    | `modal run modal/generate_dataset.py` (pilot) | 250-row JSONL pair      |
| 5    | `modal run modal/extract_activations.py`      | `h` matrix on Volume    |
| 6    | Train both probes                             | val recall@2 / recall@5 |
| 7    | Scale generation to 50+ examples/tool         | ~15k rows               |
| 8    | Retrain + Weave eval vs frontier baseline     | leaderboard             |


---

## Checklist: what info you need before generating

- [ ] `data/registry.json` with **5** agents and **200** tools (40 per agent)
- [ ] Natural-language generation prompt finalized (no structured ticket output)
- [ ] Modal app + Volume (`probe-router-data`) configured
- [ ] LLM backend chosen (HF on GPU **or** API + `modal.Secret`)
- [ ] Pilot run: 5 tools × 50 examples before full 15k generation
- [ ] Base model locked (Qwen 1.5B); activations **last layer, last token**

---

## Common mistakes to avoid

1. **Different input style at train vs inference** — e.g. structured tickets in train, chat at demo — silent accuracy collapse
2. **Including tool schemas in probe forward pass** — changes hidden states
3. **Asking the LLM to pick agent/tool** — label must come from the job, not model classification
4. **Structured ticket templates** — not used; natural language only
5. **Training tool probe with masking** — mask only at inference
6. **Forgetting `vol.commit()`** — Volume writes lost on container exit
7. **Mean pooling or mid-layer activations** — use last layer + last token only

---

## File reference


| File                           | Purpose                                       |
| ------------------------------ | --------------------------------------------- |
| `data/registry.json`           | Agent/tool catalog — sole input to generation |
| `data/agent_probe_train.jsonl` | Agent probe labels                            |
| `data/tool_probe_train.jsonl`  | Tool probe labels                             |
| `data/agent_probe_val.jsonl`   | Optional val split                            |
| `data/tool_probe_val.jsonl`    | Optional val split                            |
| `modal/generate_dataset.py`    | Modal LLM generation → JSONL                  |
| `modal/extract_activations.py` | Modal forward pass → `h`                      |
| `scripts/train_probes.py`      | `h` → LogReg × 2                              |
| `probes/probe_config.json`     | layer, model, query_format: natural_language |


**Modal docs (local):** `modal_docs/pages/docs/guide/scale.md`, `modal_docs/pages/docs/guide/volumes.md`, `modal_docs/pages/docs/guide/developing-with-llms.md`