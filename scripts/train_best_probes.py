import numpy as np
import json
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

print("Loading hidden states...")
X = np.load("data/probe_cache/mixed_Qwen2.5-7B-Instruct_layer24.npy")
print(f"X shape: {X.shape}")

print("Loading labels...")
rows = json.load(open("data/queries_mixed.json"))
y_tool = np.array([r["tool_id"] for r in rows])
y_agent = np.array([r["agent_id"] for r in rows])
print(f"Labels loaded: {len(y_tool)} tool, {len(y_agent)} agent")

print("Splitting...")
X_tr, X_te, yt_tr, yt_te, ya_tr, ya_te = train_test_split(
    X, y_tool, y_agent, test_size=0.2, random_state=42, stratify=y_tool
)
print(f"Train: {len(X_tr)}, Test: {len(X_te)}")

print("Training orchestrator probe (C=1.0)...")
orch = LogisticRegression(C=1.0, solver="saga", max_iter=1000, 
                          class_weight="balanced", verbose=1, n_jobs=-1)
orch.fit(X_tr, ya_tr)
print("Orchestrator done.")

print("Training tool probe (C=1.0)...")
tool = LogisticRegression(C=1.0, solver="saga", max_iter=1000,
                          class_weight="balanced", verbose=1, n_jobs=-1)
tool.fit(X_tr, yt_tr)
print("Tool probe done.")

os.makedirs("data/probes", exist_ok=True)
joblib.dump(orch, "data/probes/orchestrator_probe_best.pkl")
joblib.dump(tool, "data/probes/tool_probe_best.pkl")
print("Saved. Done.")
