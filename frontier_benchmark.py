"""
Frontier Benchmark: Probe-Routed system vs GPT-4o / GPT-4o-mini.

Approaches compared
-------------------
1. Naive (Llama)          — all 20 tools in context, Llama-3.1-8B selects
2. RAG (Llama)            — top-5 via sentence-transformer, Llama selects
3. Probe-Routed (Llama)   — Qwen2.5-1.5B-L27 probe top-5, Llama selects
4. GPT-4o Naive           — all 20 tools in context, GPT-4o selects
5. GPT-4o-mini Naive      — all 20 tools in context, GPT-4o-mini selects
6. GPT-4o-mini + Probe    — probe narrows to top-5, GPT-4o-mini selects
7. Qwen72B Naive (local)  — all 20 tools in context, Qwen2.5-72B via Ollama selects
8. Probe + Qwen72B (local)— probe narrows to top-5, Qwen2.5-72B via Ollama selects

Cost config (USD per 1M input tokens)
--------------------------------------
  GPT-4o:      $2.50
  GPT-4o-mini: $0.15

Router: Qwen/Qwen2.5-1.5B-Instruct, layer 27, C=0.01
"""

import warnings
warnings.filterwarnings("ignore")

import gc
import json
import os
import time
import pickle
import urllib.request
from groq import Groq

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
SEED           = 42
TOP_K_RAG      = 5
TOP_K_PROBE    = 5
MAX_NEW_TOKS   = 20
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

SELECTOR_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
EMBED_MODEL    = "all-MiniLM-L6-v2"
ROUTER_MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"
ROUTER_LAYER   = 27
PROBE_C        = 0.01

GPT4O_MODEL = "llama-3.3-70b-versatile"      # replaces gpt-4o
GPT4O_MINI_MODEL = "llama-3.1-8b-instant"    # replaces gpt-4o-mini

OLLAMA_BASE_URL  = "http://localhost:11434/api/chat"
OLLAMA_MODEL     = "qwen2.5:72b"

# USD per 1M input tokens (local Ollama = $0.00)
COST_PER_1M = {
    GPT4O_MODEL:      2.50,
    GPT4O_MINI_MODEL: 0.15,
    OLLAMA_MODEL:     0.00,
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

# ─────────────────────────────────────────────────────────────────────────────
# Load queries and split
# ─────────────────────────────────────────────────────────────────────────────
with open("queries.json") as f:
    _raw = json.load(f)
QUERIES = [(_r["query"], _r["tool"]) for _r in _raw]

rng = np.random.default_rng(SEED)
train_idx, test_idx = [], []
for tool_name in TOOL_NAMES:
    idx   = [i for i, (_, tn) in enumerate(QUERIES) if tn == tool_name]
    perm  = rng.permutation(idx)
    n_tr  = int(len(idx) * 0.8)
    train_idx.extend(perm[:n_tr].tolist())
    test_idx.extend(perm[n_tr:].tolist())

train_queries = [QUERIES[i] for i in train_idx]
test_queries  = [QUERIES[i] for i in test_idx]
print(f"Split: {len(train_queries)} train / {len(test_queries)} test")

tool_enc = LabelEncoder().fit(TOOL_NAMES)
y_train  = tool_enc.transform([tn for _, tn in train_queries])
gt_tools = [tn for _, tn in test_queries]
test_y   = tool_enc.transform(gt_tools)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers — schema formatting
# ─────────────────────────────────────────────────────────────────────────────

def fmt_schema(t: dict) -> str:
    p = ", ".join(f"{k}: {v}" for k, v in t["params"].items())
    return f"{t['name']}({p}) — {t['desc']}"


def fmt_tools_block(tool_list: list[dict]) -> str:
    return "\n".join(f"  {fmt_schema(t)}" for t in tool_list)


def build_prompt_llama(query: str, tool_list: list[dict], tok) -> str:
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
        {"role": "user", "content": f"Tools:\n{block}\n\nRequest: {query}"},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_messages_openai(query: str, tool_list: list[dict]) -> list[dict]:
    block = fmt_tools_block(tool_list)
    return [
        {
            "role": "system",
            "content": (
                "You are a tool-selection assistant. "
                "Given a user request and a list of available tools, "
                "respond with ONLY the exact name of the single best tool. "
                "No explanation, no punctuation, just the tool name."
            ),
        },
        {"role": "user", "content": f"Tools:\n{block}\n\nRequest: {query}"},
    ]


def parse_pred(raw: str, candidates: list[str]) -> str:
    s = raw.strip().lower()
    for name in candidates:
        if s.startswith(name.lower()):
            return name
    for name in candidates:
        if name.lower() in s:
            return name
    return ""


def count_tokens_local(prompt: str, tok) -> int:
    return tok(prompt, return_tensors="pt").input_ids.shape[1]


def compute_metrics(preds: list[str], tok_list: list[int], ms_list: list[float],
                    gt: list[str], cost_per_1m: float = 0.0) -> dict:
    acc  = accuracy_score(gt, preds)
    f1   = f1_score(gt, preds, labels=sorted(set(gt)), average="macro", zero_division=0)
    tok  = float(np.mean(tok_list))
    ms   = float(np.mean(ms_list))
    cost = tok / 1_000_000 * cost_per_1m * 1000  # cost per 1k queries
    return {"acc": acc, "f1": f1, "tok": tok, "ms": ms, "cost1k": cost}


CACHE = "phase3_results.pkl"

if os.path.exists(CACHE):
    print("Loading Phase 1-3 from cache, skipping recompute...")
    with open(CACHE, "rb") as f:
        _ck = pickle.load(f)
    naive_preds, naive_toks, naive_ms   = _ck["naive_preds"],  _ck["naive_toks"],  _ck["naive_ms"]
    rag_preds,   rag_toks,   rag_ms     = _ck["rag_preds"],    _ck["rag_toks"],    _ck["rag_ms"]
    probe_preds, probe_toks, probe_ms   = _ck["probe_preds"],  _ck["probe_toks"],  _ck["probe_ms"]
    probe_probs_test                    = _ck["probe_probs_test"]
    gt_tools                            = _ck["gt_tools"]
    test_queries                        = _ck["test_queries"]
    tool_enc                            = _ck["tool_enc"]
else:
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1 — RAG index
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[Phase 1] Building RAG index with {EMBED_MODEL} ...")
    embedder    = SentenceTransformer(EMBED_MODEL)
    schema_txts = [f"{t['name']} {t['desc']} {' '.join(t['params'].keys())}" for t in TOOLS]
    schema_embs = embedder.encode(schema_txts, batch_size=64, show_progress_bar=False)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2 — Qwen router: extract hidden states + train probe
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[Phase 2] Loading router {ROUTER_MODEL} (layer {ROUTER_LAYER}) ...")

    cfg_router   = AutoConfig.from_pretrained(ROUTER_MODEL, trust_remote_code=True)
    r_tok        = AutoTokenizer.from_pretrained(ROUTER_MODEL, trust_remote_code=True)
    r_tok.padding_side = "left"
    if r_tok.pad_token is None:
        r_tok.pad_token = r_tok.eos_token

    r_model = AutoModelForCausalLM.from_pretrained(
        ROUTER_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        output_hidden_states=True,
        trust_remote_code=True,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    r_model.eval()
    if DEVICE == "cpu":
        r_model = r_model.to(DEVICE)

    hs_index = ROUTER_LAYER + 1  # +1 to skip embedding entry

    def extract_hidden_states(texts: list[str], batch_size: int = 16) -> np.ndarray:
        parts = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = r_tok(
                chunk, return_tensors="pt", padding=True, truncation=True, max_length=128
            ).to(DEVICE)
            with torch.no_grad():
                out = r_model(**enc)
            hs      = out.hidden_states[hs_index]          # (B, seq, H)
            seq_len = enc["attention_mask"].sum(dim=1) - 1  # last real token
            idx     = seq_len.view(-1, 1, 1).expand(-1, 1, hs.size(-1))
            vecs    = hs.gather(1, idx).squeeze(1)
            parts.append(vecs.float().cpu().numpy())
            print(f"  {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")
        print()
        return np.vstack(parts)

    print(f"Extracting train hidden states ({len(train_queries)} queries) ...")
    X_train = extract_hidden_states([q for q, _ in train_queries])

    print(f"Extracting test hidden states ({len(test_queries)} queries) ...")
    X_test_hs = extract_hidden_states([q for q, _ in test_queries])

    print("Releasing router from GPU memory ...")
    del r_model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print(f"Training probe  C={PROBE_C} ...")
    probe = LogisticRegression(
        max_iter=2000, C=PROBE_C, solver="lbfgs", class_weight="balanced"
    )
    probe.fit(X_train, y_train)
    probe_probs_test = probe.predict_proba(X_test_hs)  # (200, 20) — reused throughout

    probe_top1 = np.mean(probe_probs_test.argmax(axis=1) == test_y)
    top5_test  = np.argsort(probe_probs_test, axis=1)[:, -TOP_K_PROBE:]
    probe_r5   = np.mean([test_y[i] in top5_test[i] for i in range(len(test_y))])
    print(f"Probe test top-1={probe_top1:.3f}  R@5={probe_r5:.3f}")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3 — Load Llama; run Naive + RAG + Probe-Routed
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[Phase 3] Loading {SELECTOR_MODEL} ...")
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

    naive_preds, naive_toks, naive_ms   = [], [], []
    rag_preds,   rag_toks,   rag_ms     = [], [], []
    probe_preds, probe_toks, probe_ms   = [], [], []

    print(f"\nRunning Naive / RAG / Probe-Routed on {len(test_queries)} test queries ...")
    for qi, (query, _) in enumerate(test_queries):
        print(f"  [{qi+1:3d}/{len(test_queries)}]", end="\r")

        # Naive
        t0    = time.perf_counter()
        p_n   = build_prompt_llama(query, TOOLS, llama_tok)
        tok_n = count_tokens_local(p_n, llama_tok)
        raw_n = llama_generate(p_n)
        naive_preds.append(parse_pred(raw_n, TOOL_NAMES))
        naive_toks.append(tok_n)
        naive_ms.append((time.perf_counter() - t0) * 1000)

        # RAG
        t0       = time.perf_counter()
        q_emb    = embedder.encode([query], show_progress_bar=False)
        sims     = cosine_similarity(q_emb, schema_embs)[0]
        top5_i   = np.argsort(sims)[::-1][:TOP_K_RAG]
        rag_pool = [TOOLS[i] for i in top5_i]
        p_r      = build_prompt_llama(query, rag_pool, llama_tok)
        tok_r    = count_tokens_local(p_r, llama_tok)
        raw_r    = llama_generate(p_r)
        rag_preds.append(parse_pred(raw_r, [t["name"] for t in rag_pool]))
        rag_toks.append(tok_r)
        rag_ms.append((time.perf_counter() - t0) * 1000)

        # Probe-Routed
        t0         = time.perf_counter()
        tool_probs = probe_probs_test[qi]
        top5_t_i   = np.argsort(tool_probs)[::-1][:TOP_K_PROBE]
        top5_names = tool_enc.inverse_transform(top5_t_i)
        probe_pool = [TOOL_BY_NAME[n] for n in top5_names]
        p_pr       = build_prompt_llama(query, probe_pool, llama_tok)
        tok_pr     = count_tokens_local(p_pr, llama_tok)
        raw_pr     = llama_generate(p_pr)
        probe_preds.append(parse_pred(raw_pr, [t["name"] for t in probe_pool]))
        probe_toks.append(tok_pr)
        probe_ms.append((time.perf_counter() - t0) * 1000)

    print()

    # Free Llama memory before API calls
    del llama
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    with open(CACHE, "wb") as f:
        pickle.dump({
            "naive_preds": naive_preds, "naive_toks": naive_toks, "naive_ms": naive_ms,
            "rag_preds":   rag_preds,   "rag_toks":   rag_toks,   "rag_ms":   rag_ms,
            "probe_preds": probe_preds, "probe_toks": probe_toks, "probe_ms": probe_ms,
            "probe_probs_test": probe_probs_test,
            "gt_tools": gt_tools, "test_queries": test_queries,
            "tool_enc": tool_enc,
        }, f)
    print(f"Phase 1-3 results saved to {CACHE}")

# ─────────────────────────────────────────────────────────────────────────────
# Metrics for Phases 1-3 (runs on both cache-hit and fresh paths)
# ─────────────────────────────────────────────────────────────────────────────
m_naive = compute_metrics(naive_preds, naive_toks, naive_ms, gt_tools, cost_per_1m=0.0)
m_rag   = compute_metrics(rag_preds,   rag_toks,   rag_ms,   gt_tools, cost_per_1m=0.0)
m_probe = compute_metrics(probe_preds, probe_toks, probe_ms, gt_tools, cost_per_1m=0.0)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — Groq-hosted frontier models
# ─────────────────────────────────────────────────────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def openai_query(model: str, messages: list[dict]) -> tuple[str, int, float]:
    """Return (raw_text, prompt_tokens, latency_ms)."""
    t0   = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=20,
        temperature=0,
    )
    ms          = (time.perf_counter() - t0) * 1000
    raw         = resp.choices[0].message.content or ""
    prompt_toks = resp.usage.prompt_tokens
    return raw, prompt_toks, ms


def ollama_query(messages: list[dict]) -> tuple[str, int, float]:
    """Call local Ollama /api/chat. Return (raw_text, prompt_tokens, latency_ms).

    Token count comes from response.usage.prompt_tokens when present;
    falls back to eval_count (number of tokens in the prompt fed to the model)
    reported by Ollama in the response body.
    """
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 20},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_BASE_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        body = json.loads(resp.read())
    ms  = (time.perf_counter() - t0) * 1000
    raw = body.get("message", {}).get("content", "")
    # Ollama reports prompt_eval_count = tokens in the prompt
    prompt_toks = body.get("prompt_eval_count", 0)
    return raw, prompt_toks, ms


gpt4o_preds,      gpt4o_toks,      gpt4o_ms      = [], [], []
gpt4o_mini_preds, gpt4o_mini_toks, gpt4o_mini_ms = [], [], []
gpt4m_probe_preds, gpt4m_probe_toks, gpt4m_probe_ms = [], [], []

print(f"\n[Phase 4] Running GPT-4o / GPT-4o-mini on {len(test_queries)} test queries ...")
for qi, (query, _) in enumerate(test_queries):
    print(f"  [{qi+1:3d}/{len(test_queries)}]", end="\r")

    # GPT-4o Naive
    msgs = build_messages_openai(query, TOOLS)
    raw, ptoks, ms = openai_query(GPT4O_MODEL, msgs)
    gpt4o_preds.append(parse_pred(raw, TOOL_NAMES))
    gpt4o_toks.append(ptoks)
    gpt4o_ms.append(ms)

    # GPT-4o-mini Naive
    raw, ptoks, ms = openai_query(GPT4O_MINI_MODEL, msgs)
    gpt4o_mini_preds.append(parse_pred(raw, TOOL_NAMES))
    gpt4o_mini_toks.append(ptoks)
    gpt4o_mini_ms.append(ms)

    # GPT-4o-mini + Probe (probe narrows to top-5)
    tool_probs = probe_probs_test[qi]
    top5_t_i   = np.argsort(tool_probs)[::-1][:TOP_K_PROBE]
    top5_names = tool_enc.inverse_transform(top5_t_i)
    mini_probe_pool = [TOOL_BY_NAME[n] for n in top5_names]
    msgs_short = build_messages_openai(query, mini_probe_pool)
    raw, ptoks, ms = openai_query(GPT4O_MINI_MODEL, msgs_short)
    gpt4m_probe_preds.append(parse_pred(raw, [t["name"] for t in mini_probe_pool]))
    gpt4m_probe_toks.append(ptoks)
    gpt4m_probe_ms.append(ms)

    time.sleep(0.5)

print()

m_gpt4o      = compute_metrics(gpt4o_preds,      gpt4o_toks,      gpt4o_ms,      gt_tools, COST_PER_1M[GPT4O_MODEL])
m_gpt4o_mini = compute_metrics(gpt4o_mini_preds, gpt4o_mini_toks, gpt4o_mini_ms, gt_tools, COST_PER_1M[GPT4O_MINI_MODEL])
m_gpt4m_probe= compute_metrics(gpt4m_probe_preds, gpt4m_probe_toks, gpt4m_probe_ms, gt_tools, COST_PER_1M[GPT4O_MINI_MODEL])

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — Qwen2.5-72B via local Ollama
# ─────────────────────────────────────────────────────────────────────────────
qwen72_naive_preds,  qwen72_naive_toks,  qwen72_naive_ms  = [], [], []
qwen72_probe_preds,  qwen72_probe_toks,  qwen72_probe_ms  = [], [], []

print(f"\n[Phase 5] Running {OLLAMA_MODEL} via Ollama on {len(test_queries)} test queries ...")
for qi, (query, _) in enumerate(test_queries):
    print(f"  [{qi+1:3d}/{len(test_queries)}]", end="\r")

    # Qwen72B Naive — all tools in context
    msgs_naive = build_messages_openai(query, TOOLS)
    raw, ptoks, ms = ollama_query(msgs_naive)
    qwen72_naive_preds.append(parse_pred(raw, TOOL_NAMES))
    qwen72_naive_toks.append(ptoks)
    qwen72_naive_ms.append(ms)

    # Probe + Qwen72B — probe top-5 shortlist
    tool_probs    = probe_probs_test[qi]
    top5_t_i      = np.argsort(tool_probs)[::-1][:TOP_K_PROBE]
    top5_names    = tool_enc.inverse_transform(top5_t_i)
    qwen_probe_pool = [TOOL_BY_NAME[n] for n in top5_names]
    msgs_short    = build_messages_openai(query, qwen_probe_pool)
    raw, ptoks, ms = ollama_query(msgs_short)
    qwen72_probe_preds.append(parse_pred(raw, [t["name"] for t in qwen_probe_pool]))
    qwen72_probe_toks.append(ptoks)
    qwen72_probe_ms.append(ms)

print()

m_qwen72_naive = compute_metrics(qwen72_naive_preds, qwen72_naive_toks, qwen72_naive_ms, gt_tools, COST_PER_1M[OLLAMA_MODEL])
m_qwen72_probe = compute_metrics(qwen72_probe_preds, qwen72_probe_toks, qwen72_probe_ms, gt_tools, COST_PER_1M[OLLAMA_MODEL])

# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────
W = 80
rows = [
    ("Naive (Llama)",          m_naive,        naive_preds),
    ("RAG (Llama)",            m_rag,          rag_preds),
    ("Probe-Routed (Llama)",   m_probe,        probe_preds),
    ("GPT-4o Naive",           m_gpt4o,        gpt4o_preds),
    ("GPT-4o-mini Naive",      m_gpt4o_mini,   gpt4o_mini_preds),
    ("GPT-4o-mini + Probe",    m_gpt4m_probe,  gpt4m_probe_preds),
    ("Qwen72B Naive (local)",  m_qwen72_naive, qwen72_naive_preds),
    ("Probe + Qwen72B (local)",m_qwen72_probe, qwen72_probe_preds),
]

print("\n\n" + "█" * W)
print("  FRONTIER BENCHMARK SUMMARY")
print("█" * W)
hdr = f"{'Approach':<26} {'Accuracy':>9} {'MacroF1':>9} {'Tok/q':>8} {'ms/q':>8} {'Cost/1kq':>10}"
print(hdr)
print("─" * W)
for label, m, _ in rows:
    cost_str = f"${m['cost1k']:.4f}" if m["cost1k"] > 0 else "—"
    print(
        f"{label:<26} {m['acc']:>9.3f} {m['f1']:>9.3f} "
        f"{m['tok']:>8.1f} {m['ms']:>8.1f} {cost_str:>10}"
    )
print("█" * W)

# ─────────────────────────────────────────────────────────────────────────────
# Per-query breakdown
# ─────────────────────────────────────────────────────────────────────────────
COL_W = 22

def ck(pred: str, gt: str) -> str:
    return "✓" if pred == gt else "✗"


print("\n\nPER-QUERY BREAKDOWN")
print("─" * (COL_W * 9 + 10))
hdr2 = (
    f"{'#':>4}  "
    f"{'N':^{COL_W}}"
    f"{'R':^{COL_W}}"
    f"{'P':^{COL_W}}"
    f"{'G4':^{COL_W}}"
    f"{'G4M':^{COL_W}}"
    f"{'G4M+P':^{COL_W}}"
    f"{'Q72':^{COL_W}}"
    f"{'Q72+P':^{COL_W}}"
    f"  GT"
)
print(hdr2)
print("─" * (COL_W * 9 + 10))

def cell(pred: str, gt: str) -> str:
    mark = "✓" if pred == gt else "✗"
    s = f"{mark} {pred}"
    return s[:COL_W - 1].ljust(COL_W)

for qi, (query, gt) in enumerate(test_queries):
    print(
        f"{qi+1:>4}  "
        f"{cell(naive_preds[qi],       gt)}"
        f"{cell(rag_preds[qi],         gt)}"
        f"{cell(probe_preds[qi],       gt)}"
        f"{cell(gpt4o_preds[qi],       gt)}"
        f"{cell(gpt4o_mini_preds[qi],  gt)}"
        f"{cell(gpt4m_probe_preds[qi], gt)}"
        f"{cell(qwen72_naive_preds[qi],gt)}"
        f"{cell(qwen72_probe_preds[qi],gt)}"
        f"  {gt}"
    )

print("─" * (COL_W * 9 + 10))
print("\nLegend: N=Naive(Llama)  R=RAG(Llama)  P=Probe-Routed(Llama)  "
      "G4=GPT-4o Naive  G4M=GPT-4o-mini Naive  G4M+P=GPT-4o-mini+Probe  "
      "Q72=Qwen72B Naive(local)  Q72+P=Probe+Qwen72B(local)")
print("\nDone.")
