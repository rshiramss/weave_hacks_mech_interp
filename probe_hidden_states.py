"""
Probe whether Qwen2.5-1.5B-Instruct hidden states cluster by tool intent.
Layers probed: 8, 16, 24, and the final layer (27 for 1.5B, not 32 which is
out-of-range for this model — adjust LAYERS if you use a deeper variant).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS = {
    "read_file": {
        "description": "Read the contents of a file from the local filesystem.",
        "params": {"path": "str", "encoding": "str"},
    },
    "web_search": {
        "description": "Search the internet and return ranked results with snippets.",
        "params": {"query": "str", "num_results": "int"},
    },
    "send_email": {
        "description": "Compose and send an email to one or more recipients.",
        "params": {"to": "list[str]", "subject": "str", "body": "str", "cc": "list[str]"},
    },
    "execute_python": {
        "description": "Run a snippet of Python code and return stdout/stderr.",
        "params": {"code": "str", "timeout": "int"},
    },
    "query_database": {
        "description": "Execute a SQL query against a relational database and return rows.",
        "params": {"sql": "str", "connection_string": "str"},
    },
}

# ---------------------------------------------------------------------------
# 20 realistic queries per tool  (naturally phrased, tool name never mentioned)
# ---------------------------------------------------------------------------

QUERIES = {
    "read_file": [
        "What's inside the config.yaml file in my project root?",
        "Show me what the README says about installation.",
        "Can you pull up the contents of /etc/hosts?",
        "I need to see the last few lines of app.log.",
        "What does the requirements.txt say?",
        "Open the .env file and tell me which variables are set.",
        "Grab the text from notes.txt on my desktop.",
        "Look at the Dockerfile and describe its stages.",
        "What does my crontab currently contain?",
        "Display the contents of schema.sql.",
        "Check what's inside the secrets folder's credentials file.",
        "I want to review the migration script from last week.",
        "What's in the package.json under the scripts section?",
        "Show me the full content of the terms_of_service.txt.",
        "Tell me what the Makefile target 'build' does.",
        "Can you look at the error log from yesterday?",
        "What version string is in VERSION.txt?",
        "Inspect the nginx.conf and tell me the server block.",
        "Read through my journal entry from 2024-03-15.txt.",
        "What does the manifest.json file declare?",
    ],
    "web_search": [
        "What's the current stock price of NVIDIA?",
        "Who won the Champions League last season?",
        "What are the latest updates on the merger between those two airlines?",
        "Find me some reviews for the new Arc browser.",
        "What is the recommended daily intake of vitamin D?",
        "Are there any good tutorials for learning Rust in 2025?",
        "What time does the Apple keynote start tonight?",
        "What's the weather like in Tokyo this weekend?",
        "Find recent papers on diffusion models for audio generation.",
        "What are the top Python libraries for time-series forecasting?",
        "Is there news about the next-gen PlayStation?",
        "Look up symptoms of magnesium deficiency.",
        "What's the current exchange rate for USD to EUR?",
        "Are there any open-source alternatives to Notion?",
        "What happened at CES this year?",
        "Find me documentation on the OpenAI assistants API.",
        "What movies are trending on streaming platforms right now?",
        "Who is the current prime minister of Canada?",
        "What are people saying about the new M4 MacBook?",
        "Get me the latest Python release notes.",
    ],
    "send_email": [
        "Let my manager know I'll be out sick tomorrow.",
        "Remind the team about the standup at 10 AM.",
        "Send Alice a follow-up on the proposal we discussed.",
        "Draft a thank-you note to the interviewer from yesterday.",
        "Notify the client that the project is delayed by one week.",
        "Shoot a message to HR asking about the parental leave policy.",
        "Write to the vendor requesting a new quote for Q3.",
        "Let Bob know his pull request has been approved.",
        "Send a meeting invite recap to everyone who attended.",
        "Ping the design team asking for updated mockups by Friday.",
        "Message the on-call engineer about the production incident.",
        "Reach out to the speaker confirming their slot at the conference.",
        "Tell the recruiter we'd like to move forward with the candidate.",
        "Compose a newsletter update for our subscribers.",
        "Inform the finance team that the invoice has been paid.",
        "Write a quick apology to the client for the miscommunication.",
        "Let Sarah know that her access has been provisioned.",
        "Drop a note to the whole engineering org about the API deprecation.",
        "Send a birthday greeting to my colleague.",
        "Message the board summarising this quarter's highlights.",
    ],
    "execute_python": [
        "Calculate the SHA-256 hash of the string 'hello world'.",
        "What's the 1000th Fibonacci number?",
        "Parse this CSV string and tell me the column names.",
        "Convert this list of timestamps to UTC.",
        "Sort this dictionary by value in descending order.",
        "Find all prime numbers between 1 and 500.",
        "Benchmark how long it takes to sort a million random integers.",
        "Flatten this nested list three levels deep.",
        "What's the base64 encoding of my API key string?",
        "Run a linear regression on these two lists and return the slope.",
        "Compress this JSON payload and report the size reduction.",
        "Generate a UUID and print it.",
        "Count the word frequency in this paragraph.",
        "Validate whether this string is a valid IPv4 address.",
        "Render a simple bar chart from these values and save it.",
        "Compute the cosine similarity between these two vectors.",
        "What's the current Unix timestamp?",
        "Parse this HTML snippet and extract all hyperlinks.",
        "Decode this base64-encoded string back to plaintext.",
        "Simulate rolling two dice 10 000 times and show the distribution.",
    ],
    "query_database": [
        "How many users signed up in the last 30 days?",
        "What's the total revenue for Q2 this year?",
        "List the top 10 products by number of orders.",
        "Which customers haven't logged in for over 90 days?",
        "Show me average session duration broken down by country.",
        "Find all orders that are still in 'pending' status.",
        "What's the churn rate for last month?",
        "Pull the names and emails of users on the premium plan.",
        "How many support tickets were opened versus closed this week?",
        "Give me a count of active subscriptions per pricing tier.",
        "Which sales rep closed the most deals this quarter?",
        "Show me inventory levels for items below the reorder threshold.",
        "What's the month-over-month growth in new accounts?",
        "Find duplicate email addresses in the users table.",
        "Which pages have the highest average load time in the analytics table?",
        "List all failed payment attempts from the past 7 days.",
        "How many API calls did each tenant make yesterday?",
        "Identify any accounts where the billing address is missing.",
        "What's the average order value segmented by acquisition channel?",
        "Show me the most-used feature flags in the experiments table.",
    ],
}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# Qwen2.5-1.5B has 28 transformer layers (indices 0–27).
# Layer 32 is out-of-range; using 27 (final) in its place.
LAYERS = [8, 16, 24, 27]
LAYER_LABELS = ["Layer 8", "Layer 16", "Layer 24", "Layer 27 (final)"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8

# ---------------------------------------------------------------------------
# Load model + tokenizer
# ---------------------------------------------------------------------------

print(f"Loading {MODEL_NAME} on {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    output_hidden_states=True,
    trust_remote_code=True,
    device_map="auto" if DEVICE == "cuda" else None,
)
model.eval()
if DEVICE == "cpu":
    model = model.to(DEVICE)

print(f"Model loaded. Num hidden layers: {model.config.num_hidden_layers}")
# Sanity-check layer indices
max_layer = model.config.num_hidden_layers  # hidden_states has shape (num_layers+1,)
for l in LAYERS:
    assert l <= max_layer, f"Layer {l} out of range (max index = {max_layer})"

# ---------------------------------------------------------------------------
# Build flat query/label lists
# ---------------------------------------------------------------------------

all_queries: list[str] = []
all_labels: list[str] = []
for tool_name, queries in QUERIES.items():
    assert len(queries) == 20, f"{tool_name} has {len(queries)} queries, expected 20"
    all_queries.extend(queries)
    all_labels.extend([tool_name] * len(queries))

print(f"Total queries: {len(all_queries)}")

# ---------------------------------------------------------------------------
# Extract hidden states with batched inference
# ---------------------------------------------------------------------------

# hidden_states_by_layer[layer_idx] will be a list of numpy vectors
hidden_states_by_layer: dict[int, list[np.ndarray]] = {l: [] for l in LAYERS}


def extract_last_token_hidden(batch_texts: list[str]) -> dict[int, np.ndarray]:
    """Return dict mapping layer index → (batch, hidden_dim) numpy array."""
    inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.hidden_states: tuple of (num_layers+1) tensors, each (B, seq, H)
    # Index 0 is the embedding layer; index k is after transformer block k-1.
    # hidden_states[k+1] corresponds to the output of layer k (0-indexed).
    hidden_states = outputs.hidden_states  # length = num_layers + 1

    # Identify the actual last (non-padding) token for each item
    attention_mask = inputs["attention_mask"]  # (B, seq)
    # last real token position per sequence
    seq_lens = attention_mask.sum(dim=1) - 1  # (B,)

    result: dict[int, np.ndarray] = {}
    for layer_idx in LAYERS:
        hs = hidden_states[layer_idx + 1]  # (B, seq, H)  — +1 for embedding offset
        # Gather last real token
        idx = seq_lens.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hs.size(-1))
        last_tok = hs.gather(1, idx).squeeze(1)  # (B, H)
        result[layer_idx] = last_tok.float().cpu().numpy()
    return result


print("Running forward passes ...")
for batch_start in range(0, len(all_queries), BATCH_SIZE):
    batch = all_queries[batch_start : batch_start + BATCH_SIZE]
    batch_result = extract_last_token_hidden(batch)
    for layer_idx, vecs in batch_result.items():
        hidden_states_by_layer[layer_idx].extend(vecs)
    if (batch_start // BATCH_SIZE + 1) % 5 == 0 or batch_start + BATCH_SIZE >= len(all_queries):
        done = min(batch_start + BATCH_SIZE, len(all_queries))
        print(f"  Processed {done}/{len(all_queries)} queries")

# Stack into (N, H) arrays
X_by_layer: dict[int, np.ndarray] = {
    l: np.stack(vecs) for l, vecs in hidden_states_by_layer.items()
}

# Encode labels
le = LabelEncoder()
y = le.fit_transform(all_labels)
tool_names = le.classes_
colors_map = {
    name: plt.cm.tab10(i / len(tool_names)) for i, name in enumerate(tool_names)
}
point_colors = [colors_map[name] for name in all_labels]

# ---------------------------------------------------------------------------
# TSNE + silhouette + probe
# ---------------------------------------------------------------------------

print("\n--- Results ---")
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle("Qwen2.5-1.5B-Instruct — Hidden State Clusters by Tool Intent", fontsize=13)

for ax, layer_idx, label in zip(axes, LAYERS, LAYER_LABELS):
    X = X_by_layer[layer_idx]  # (100, H)

    # TSNE
    tsne = TSNE(n_components=2, perplexity=20, random_state=42, max_iter=1000)
    X_2d = tsne.fit_transform(X)

    # Silhouette
    sil = silhouette_score(X, y, metric="cosine")
    print(f"{label}  |  silhouette (cosine) = {sil:.4f}")

    # Linear probe  (80/20 split, stratified)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(X_tr, y_tr)
    acc = clf.score(X_te, y_te)
    print(f"{label}  |  probe accuracy (80/20) = {acc:.4f}")

    # Plot
    for i, tool in enumerate(tool_names):
        mask = np.array(all_labels) == tool
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            c=[colors_map[tool]],
            label=tool,
            s=30,
            alpha=0.85,
            edgecolors="none",
        )
    ax.set_title(f"{label}\nsil={sil:.3f}  acc={acc:.3f}")
    ax.set_xticks([])
    ax.set_yticks([])

# Shared legend
handles = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors_map[t], markersize=8, label=t)
    for t in tool_names
]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("layer_viz.png", dpi=150, bbox_inches="tight")
print("\nPlot saved to layer_viz.png")
