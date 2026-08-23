"""
Benchmark: Naive vs RAG vs Probe-Routed tool selection.

20 tools · 4 categories · 1000 queries · 800 train / 200 test (stratified by tool)

Models
------
Routers (hidden states): swept via ROUTER_CONFIGS  (each unloaded after use)
Final selection:         meta-llama/Llama-3.1-8B-Instruct
Embeddings (RAG):        all-MiniLM-L6-v2

Approaches
----------
1. Naive        – all 20 tool schemas in context → Llama picks.
2. RAG          – sentence-transformer retrieves top-5 schemas → Llama picks.
3. Probe-Routed – router hidden state → LogisticRegression probe over 20 tool
                  classes → top-5 by probe probability injected into context →
                  Llama picks.  Repeated for each entry in ROUTER_CONFIGS.

Probe note: 40 train examples per tool (800 total / 20 classes).
"""

import warnings
warnings.filterwarnings("ignore")

import gc
import time
import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
SELECTOR_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
EMBED_MODEL    = "all-MiniLM-L6-v2"
TOP_K_RAG      = 5
TOP_K_PROBE    = 5
MAX_NEW_TOKS   = 20
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SEED           = 42

# Router configs to sweep.  layer=-1 means the final hidden layer.
ROUTER_CONFIGS = [
    {"model": "Qwen/Qwen2.5-1.5B-Instruct",        "layer": 8,  "C": 0.01, "label": "Qwen1.5B-L8"},
    {"model": "Qwen/Qwen2.5-1.5B-Instruct",        "layer": 27, "C": 0.01, "label": "Qwen1.5B-L27"},
    {"model": "Qwen/Qwen2.5-0.5B-Instruct",        "layer": -1, "C": 0.01, "label": "Qwen0.5B-Last"},
    {"model": "microsoft/deberta-v3-base",           "layer": -1, "C": 0.01, "label": "DeBERTa-Last"},
    {"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "layer": -1, "C": 0.01, "label": "TinyLlama-Last"},
]

# Standardised model_type values for encoder-only architectures.
# These use [CLS] token (index 0); all others use the last real token.
ENCODER_ONLY_TYPES = {
    "bert", "deberta", "deberta-v2", "distilbert", "roberta",
    "electra", "albert", "camembert", "xlm-roberta", "ernie",
}

# ─────────────────────────────────────────────────────────────────────────────
# Tools — 20 across 4 categories
# ─────────────────────────────────────────────────────────────────────────────
TOOLS = [
    # file_operations
    {"name": "read_file",            "cat": "file_operations",  "desc": "Read the contents of a file from the local filesystem.",                         "params": {"path": "str", "encoding": "str"}},
    {"name": "write_file",           "cat": "file_operations",  "desc": "Write or overwrite a file with the provided content.",                            "params": {"path": "str", "content": "str", "mode": "str"}},
    {"name": "list_directory",       "cat": "file_operations",  "desc": "List files and subdirectories inside a directory path.",                          "params": {"path": "str", "recursive": "bool"}},
    {"name": "delete_file",          "cat": "file_operations",  "desc": "Permanently delete a file or empty directory.",                                   "params": {"path": "str"}},
    {"name": "compress_files",       "cat": "file_operations",  "desc": "Compress one or more files into a zip or tar archive.",                           "params": {"sources": "list[str]", "output": "str", "format": "str"}},
    # web_search
    {"name": "web_search",           "cat": "web_search",       "desc": "Search the internet and return ranked results with snippets.",                    "params": {"query": "str", "num_results": "int"}},
    {"name": "image_search",         "cat": "web_search",       "desc": "Search for images by keyword and return image URLs.",                             "params": {"query": "str", "safe_search": "bool"}},
    {"name": "news_search",          "cat": "web_search",       "desc": "Search recent news articles and return headlines and summaries.",                 "params": {"query": "str", "days_back": "int"}},
    {"name": "academic_search",      "cat": "web_search",       "desc": "Search academic databases for papers and citations.",                             "params": {"query": "str", "year_from": "int"}},
    {"name": "video_search",         "cat": "web_search",       "desc": "Search for videos by keyword and return links and metadata.",                     "params": {"query": "str", "platform": "str"}},
    # communication
    {"name": "send_email",           "cat": "communication",    "desc": "Compose and send an email to one or more recipients.",                            "params": {"to": "list[str]", "subject": "str", "body": "str", "cc": "list[str]"}},
    {"name": "send_sms",             "cat": "communication",    "desc": "Send a text message to a phone number.",                                          "params": {"to": "str", "message": "str"}},
    {"name": "post_slack",           "cat": "communication",    "desc": "Post a message to a Slack channel or direct message thread.",                     "params": {"channel": "str", "message": "str", "thread_ts": "str"}},
    {"name": "create_calendar_event","cat": "communication",    "desc": "Create a calendar event with a title, time range, and optional attendees.",       "params": {"title": "str", "start": "str", "end": "str", "attendees": "list[str]"}},
    {"name": "create_meeting",       "cat": "communication",    "desc": "Schedule a video meeting and return the join link.",                              "params": {"topic": "str", "start_time": "str", "duration": "int", "participants": "list[str]"}},
    # code_execution
    {"name": "execute_python",       "cat": "code_execution",   "desc": "Run a Python code snippet and return stdout and stderr.",                         "params": {"code": "str", "timeout": "int"}},
    {"name": "execute_bash",         "cat": "code_execution",   "desc": "Execute a shell command or bash script and return output.",                       "params": {"command": "str", "cwd": "str"}},
    {"name": "run_unit_tests",       "cat": "code_execution",   "desc": "Discover and run unit tests in a project directory and report results.",          "params": {"path": "str", "framework": "str", "verbose": "bool"}},
    {"name": "format_code",          "cat": "code_execution",   "desc": "Auto-format source code according to language style conventions.",                "params": {"code": "str", "language": "str", "style": "str"}},
    {"name": "lint_code",            "cat": "code_execution",   "desc": "Run a linter over source code and return warnings and errors.",                   "params": {"code": "str", "language": "str", "rules": "list[str]"}},
]

assert len(TOOLS) == 20
TOOL_NAMES   = [t["name"] for t in TOOLS]
TOOL_BY_NAME = {t["name"]: t for t in TOOLS}
CATEGORIES   = sorted({t["cat"] for t in TOOLS})

# ─────────────────────────────────────────────────────────────────────────────
# Load 1000 queries from queries.json (50 per tool, 20 tools)
# ─────────────────────────────────────────────────────────────────────────────
import json as _json
with open("queries.json") as _f:
    _raw = _json.load(_f)
QUERIES = [(_r["query"], _r["tool"]) for _r in _raw]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def fmt_schema(t: dict) -> str:
    p = ", ".join(f"{k}: {v}" for k, v in t["params"].items())
    return f"{t['name']}({p}) — {t['desc']}"


def fmt_tools_block(tool_list: list[dict]) -> str:
    return "\n".join(f"  {fmt_schema(t)}" for t in tool_list)


def build_prompt(query: str, tool_list: list[dict], llama_tok) -> str:
    block = fmt_tools_block(tool_list)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a tool-selection assistant. "
                "Given a user request and a list of available tools, "
                "respond with ONLY the exact name of the single best tool. "
                "No explanation, no punctuation, just the tool name."
            ),
        },
        {
            "role": "user",
            "content": f"Tools:\n{block}\n\nRequest: {query}",
        },
    ]
    return llama_tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def parse_pred(raw: str, candidates: list[str]) -> str:
    s = raw.strip().lower()
    for name in candidates:
        if s.startswith(name.lower()):
            return name
    for name in candidates:
        if name.lower() in s:
            return name
    return ""


def count_tokens(prompt: str, tok) -> int:
    return tok(prompt, return_tensors="pt").input_ids.shape[1]


# ─────────────────────────────────────────────────────────────────────────────
# 800 / 200 stratified split  (40 train + 10 test per tool)
# ─────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
train_idx, test_idx = [], []
for tool_name in TOOL_NAMES:
    idx = [i for i, (_, tn) in enumerate(QUERIES) if tn == tool_name]
    perm = rng.permutation(idx)
    n_train = int(len(idx) * 0.8)
    train_idx.extend(perm[:n_train].tolist())
    test_idx.extend(perm[n_train:].tolist())

train_queries = [QUERIES[i] for i in train_idx]   # 800
test_queries  = [QUERIES[i] for i in test_idx]    # 200
print(f"Split: {len(train_queries)} train / {len(test_queries)} test")

tool_enc = LabelEncoder().fit(TOOL_NAMES)
y_train  = tool_enc.transform([tn for _, tn in train_queries])
gt_tools = [tn for _, tn in test_queries]
test_y   = tool_enc.transform(gt_tools)

# ─────────────────────────────────────────────────────────────────────────────
# Router loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_encoder_only(model_type: str) -> bool:
    return model_type.lower() in ENCODER_ONLY_TYPES


def load_router(model_name: str):
    """Return (tokenizer, model, is_encoder_only)."""
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    encoder_only = _is_encoder_only(cfg.model_type)

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.padding_side = "right" if encoder_only else "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ModelCls = AutoModel if encoder_only else AutoModelForCausalLM
    model = ModelCls.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        output_hidden_states=True,
        trust_remote_code=True,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    model.eval()
    if DEVICE == "cpu":
        model = model.to(DEVICE)
    return tok, model, encoder_only


def unload_router(model):
    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def _resolve_layer(layer_idx: int) -> int:
    """Convert a transformer block number to a hidden_states tuple index.

    hidden_states is a tuple of length num_hidden_layers + 1:
      [0]        embedding output
      [1]        transformer block 0 output
      ...
      [N]        transformer block N-1 output  (last layer)

    layer_idx=-1 → -1  (Python negative indexing already selects the last element)
    layer_idx=N  → N+1 (skip the embedding entry at index 0)
    """
    if layer_idx < 0:
        return layer_idx   # pass through; -1 selects last, -2 second-to-last, etc.
    return layer_idx + 1   # +1 to skip embedding entry at index 0


def extract_hidden_states(
    texts: list[str],
    tok,
    model,
    encoder_only: bool,
    layer_idx: int,
    batch_size: int = 16,
) -> np.ndarray:
    """Return (N, H) float32 array of hidden states."""
    hs_index = _resolve_layer(layer_idx)

    parts = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tok(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(DEVICE)
        with torch.no_grad():
            out = model(**enc)

        if i == 0:
            n_hs = len(out.hidden_states)
            print(f"  [debug] hidden_states tuple length={n_hs}  "
                  f"config.num_hidden_layers={model.config.num_hidden_layers}  "
                  f"resolved hs_index={hs_index}")
            assert -n_hs <= hs_index < n_hs, (
                f"hs_index {hs_index} out of range for hidden_states of length {n_hs} "
                f"(layer_idx={layer_idx}, config.num_hidden_layers={model.config.num_hidden_layers})"
            )

        hs = out.hidden_states[hs_index]   # (B, seq, H)

        if encoder_only:
            # CLS token is always position 0 for BERT-family encoders
            vecs = hs[:, 0, :]
        else:
            # last real (non-padding) token; padding_side="left" so padding is on the left
            seq_len = enc["attention_mask"].sum(dim=1) - 1   # (B,)
            idx = seq_len.view(-1, 1, 1).expand(-1, 1, hs.size(-1))
            vecs = hs.gather(1, idx).squeeze(1)

        parts.append(vecs.float().cpu().numpy())
        print(f"  {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")

    print()
    return np.vstack(parts)


def compute(r: dict, gt: list[str]) -> dict:
    preds = r["preds"]
    acc   = accuracy_score(gt, preds)
    present = sorted(set(gt))
    f1    = f1_score(gt, preds, labels=present, average="macro", zero_division=0)
    return {
        "acc":  acc,
        "f1":   f1,
        "tok":  float(np.mean(r["tok"])),
        "ms":   float(np.mean(r["ms"])),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — sentence-transformer RAG index
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[Phase 1] Building RAG index with {EMBED_MODEL} ...")
embedder    = SentenceTransformer(EMBED_MODEL)
schema_txts = [f"{t['name']} {t['desc']} {' '.join(t['params'].keys())}" for t in TOOLS]
schema_embs = embedder.encode(schema_txts, batch_size=64, show_progress_bar=False)  # (20, 384)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — load Llama once; run Naive + RAG inference once
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[Phase 2] Loading {SELECTOR_MODEL} ...")
llama_tok = AutoTokenizer.from_pretrained(SELECTOR_MODEL)
llama_tok.padding_side = "left"
if llama_tok.pad_token is None:
    llama_tok.pad_token = llama_tok.eos_token

llama = AutoModelForCausalLM.from_pretrained(
    SELECTOR_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
)
llama.eval()
if DEVICE == "cpu":
    llama = llama.to(DEVICE)


def llama_generate(prompt: str) -> str:
    enc = llama_tok(
        prompt, return_tensors="pt", truncation=True, max_length=4096
    ).to(DEVICE)
    with torch.no_grad():
        out = llama.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKS,
            do_sample=False,
            pad_token_id=llama_tok.pad_token_id,
            eos_token_id=llama_tok.eos_token_id,
        )
    new_ids = out[0][enc["input_ids"].shape[1]:]
    return llama_tok.decode(new_ids, skip_special_tokens=True)


W = 90

naive_result = {"preds": [], "tok": [], "ms": []}
rag_result   = {"preds": [], "tok": [], "ms": []}

print(f"\nRunning Naive and RAG inference on {len(test_queries)} test queries ...")
for qi, (query, _) in enumerate(test_queries):
    print(f"  [{qi+1:3d}/{len(test_queries)}] {query[:72]}", end="\r")

    t0    = time.perf_counter()
    p_n   = build_prompt(query, TOOLS, llama_tok)
    tok_n = count_tokens(p_n, llama_tok)
    raw_n = llama_generate(p_n)
    ms_n  = (time.perf_counter() - t0) * 1000
    naive_result["preds"].append(parse_pred(raw_n, TOOL_NAMES))
    naive_result["tok"].append(tok_n)
    naive_result["ms"].append(ms_n)

    t0      = time.perf_counter()
    q_emb   = embedder.encode([query], show_progress_bar=False)
    sims    = cosine_similarity(q_emb, schema_embs)[0]
    top5_i  = np.argsort(sims)[::-1][:TOP_K_RAG]
    rag_pool = [TOOLS[i] for i in top5_i]
    p_r     = build_prompt(query, rag_pool, llama_tok)
    tok_r   = count_tokens(p_r, llama_tok)
    raw_r   = llama_generate(p_r)
    ms_r    = (time.perf_counter() - t0) * 1000
    rag_result["preds"].append(parse_pred(raw_r, [t["name"] for t in rag_pool]))
    rag_result["tok"].append(tok_r)
    rag_result["ms"].append(ms_r)

print()
m_naive = compute(naive_result, gt_tools)
m_rag   = compute(rag_result,   gt_tools)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — router sweep
# ─────────────────────────────────────────────────────────────────────────────
router_summary = []   # one row per ROUTER_CONFIGS entry

for cfg in ROUTER_CONFIGS:
    model_name  = cfg["model"]
    layer_idx   = cfg["layer"]
    C_VAL       = cfg["C"]
    label       = cfg["label"]

    print(f"\n{'━' * W}")
    print(f"  Router: {label}  (model={model_name}, layer={layer_idx}, C={C_VAL})")
    print(f"{'━' * W}")

    # ── Load router ──────────────────────────────────────────────────────────
    print(f"Loading {model_name} ...")
    r_tok, r_model, enc_only = load_router(model_name)

    # ── Extract hidden states ────────────────────────────────────────────────
    print(f"Extracting training hidden states ({len(train_queries)} queries) ...")
    X_train = extract_hidden_states(
        [q for q, _ in train_queries], r_tok, r_model, enc_only, layer_idx
    )

    print(f"Extracting test hidden states ({len(test_queries)} queries) ...")
    X_test_hs = extract_hidden_states(
        [q for q, _ in test_queries], r_tok, r_model, enc_only, layer_idx
    )

    # ── Unload router before Llama inference ─────────────────────────────────
    print("Releasing router from GPU memory ...")
    unload_router(r_model)

    # ── Train probe ──────────────────────────────────────────────────────────
    print(f"Training tool-level probe  C={C_VAL} ...")
    probe = LogisticRegression(
        max_iter=2000, C=C_VAL,
        solver="lbfgs", class_weight="balanced",
    )
    probe.fit(X_train, y_train)

    train_acc   = probe.score(X_train, y_train)
    train_probs = probe.predict_proba(X_train)
    top5_train  = np.argsort(train_probs, axis=1)[:, -TOP_K_PROBE:]
    r5_train    = np.mean([y_train[i] in top5_train[i] for i in range(len(y_train))])

    test_probs  = probe.predict_proba(X_test_hs)
    probe_top1  = np.mean(test_probs.argmax(axis=1) == test_y)
    top5_test   = np.argsort(test_probs, axis=1)[:, -TOP_K_PROBE:]
    probe_r5    = np.mean([test_y[i] in top5_test[i] for i in range(len(gt_tools))])

    print(f"Probe  train top-1: {train_acc:.3f}   train recall@5: {r5_train:.3f}")
    print(f"Probe  test  top-1: {probe_top1:.3f}   test  recall@5: {probe_r5:.3f}")

    # ── Probe-routed inference (reuses precomputed X_test_hs) ────────────────
    probe_result = {"preds": [], "tok": [], "ms": []}

    print(f"\nRunning probe-routed inference on {len(test_queries)} test queries ...")
    for qi, (query, _) in enumerate(test_queries):
        t0         = time.perf_counter()
        hs_q       = X_test_hs[qi : qi + 1]
        tool_probs = probe.predict_proba(hs_q)[0]
        top5_t_i   = np.argsort(tool_probs)[::-1][:TOP_K_PROBE]
        top5_names = tool_enc.inverse_transform(top5_t_i)
        probe_pool = [TOOL_BY_NAME[n] for n in top5_names]
        p_probe    = build_prompt(query, probe_pool, llama_tok)
        tok_p      = count_tokens(p_probe, llama_tok)
        raw_p      = llama_generate(p_probe)
        ms_p       = (time.perf_counter() - t0) * 1000
        probe_result["preds"].append(parse_pred(raw_p, [t["name"] for t in probe_pool]))
        probe_result["tok"].append(tok_p)
        probe_result["ms"].append(ms_p)

    m_probe = compute(probe_result, gt_tools)

    router_summary.append({
        "label":       label,
        "layer":       layer_idx,
        "C":           C_VAL,
        "train_top1":  train_acc,
        "test_top1":   probe_top1,
        "test_r5":     probe_r5,
        "e2e_acc":     m_probe["acc"],
        "ms":          m_probe["ms"],
    })

    print(f"\n  {label}  probe top-1={probe_top1:.3f}  R@5={probe_r5:.3f}  "
          f"E2E-acc={m_probe['acc']:.3f}  ms/q={m_probe['ms']:.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# Final summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n" + "█" * W)
print("  ROUTER SWEEP SUMMARY")
print("█" * W)
print(f"{'Router':<20} {'Layer':>6} {'C':>6} {'TrainTop1':>10} {'TestTop1':>10} "
      f"{'TestR@5':>9} {'E2E-Acc':>9} {'ms/q':>8}")
print("─" * W)

# Naive and RAG reference rows (no probe metrics)
print(f"{'Naive (20 tools)':<20} {'—':>6} {'—':>6} {'—':>10} {'—':>10} "
      f"{'—':>9} {m_naive['acc']:>9.3f} {m_naive['ms']:>8.1f}")
print(f"{'RAG (top-5)':<20} {'—':>6} {'—':>6} {'—':>10} {'—':>10} "
      f"{'—':>9} {m_rag['acc']:>9.3f} {m_rag['ms']:>8.1f}")
print("─" * W)

for row in router_summary:
    print(f"{row['label']:<20} {row['layer']:>6} {row['C']:>6} "
          f"{row['train_top1']:>10.3f} {row['test_top1']:>10.3f} "
          f"{row['test_r5']:>9.3f} {row['e2e_acc']:>9.3f} {row['ms']:>8.1f}")

print("█" * W)
print("\nDone.")
