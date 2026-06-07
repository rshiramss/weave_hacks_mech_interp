"""Generate natural-language SOC analyst queries on Modal (Plan Steps 3-4).

One generation job == one tool == one row. The label (agent_id, tool_id) is fixed
by the job args BEFORE the model runs; the model only writes a realistic message
that *requires* that tool. Jobs fan out across Modal containers via `.starmap`.

Backend: W&B Inference API (OpenAI-compatible) calling meta-llama/Llama-3.1-8B-Instruct
— the same model used later for activation extraction (Plan Step 5). `WANDB_API_KEY`
is injected into containers via a Modal Secret built from the local environment.

Persistence: to avoid the Volumes-v1 concurrent-commit limit (~5 writers), workers
RETURN rows instead of each committing a file. The local entrypoint collects all
rows, writes the probe JSONL files locally, then does a single `vol.batch_upload()`
so downstream Modal steps (extract_activations) can read the dataset from the Volume.

Run:
    modal run modal_app/generate_dataset.py                 # pilot: 5 tools x 50
    modal run modal_app/generate_dataset.py --no-pilot      # full: 200 tools x 75
    modal volume ls probe-router-data                       # inspect Volume
"""

import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import modal

from src.probe.query_contract import (
    build_generation_messages,
    normalize_query,
    validate_query,
)

# --- App / resource names (kebab-case per Modal convention) ------------------
APP_NAME = "probe-router-generate"
VOLUME_NAME = "probe-router-data"

# --- Generation backend ------------------------------------------------------
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"  # served by W&B Inference
WANDB_BASE_URL = "https://api.inference.wandb.ai/v1"
TEMPERATURE = 0.9  # high → varied wording across a tool's many examples
MAX_TOKENS = 220  # a 1-4 sentence analyst message
MAX_ATTEMPTS = 5  # per job: API error OR invalid output both consume an attempt

# --- Local output layout -----------------------------------------------------
SPLIT_SEED = 1337  # deterministic train/val/test split
VAL_FRAC = 0.10
TEST_FRAC = 0.10

app = modal.App(APP_NAME)

# Persist generated data so extract_activations.py can read it from the Volume.
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# `add_local_python_source("src")` ships the shared query_contract module into the
# container (Modal 1.0+ no longer auto-mounts local source).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("openai>=1.40")
    .add_local_python_source("src")
)

# Build the Secret from the local environment so no manual `modal secret create`
# step is needed. Only resolve it locally — on the remote container the key is
# already present as an env var, and `.env` does not exist there.
if modal.is_local():
    from dotenv import load_dotenv

    load_dotenv()
    _secret_values = {"WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "")}
    if os.environ.get("WANDB_PROJECT"):
        _secret_values["WANDB_PROJECT"] = os.environ["WANDB_PROJECT"]
    wandb_secret = modal.Secret.from_dict(_secret_values)
else:
    wandb_secret = modal.Secret.from_dict({})


def _client():
    """Return a process-cached OpenAI client pointed at the W&B endpoint.

    Cached on the function object so each container builds it once and reuses it
    across the concurrent inputs it handles.
    """
    cached = getattr(_client, "_instance", None)
    if cached is not None:
        return cached

    from openai import OpenAI

    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError("WANDB_API_KEY missing in container — check Modal Secret.")

    kwargs = {"base_url": WANDB_BASE_URL, "api_key": api_key}
    if os.environ.get("WANDB_PROJECT"):
        kwargs["project"] = os.environ["WANDB_PROJECT"]  # attribute usage in W&B

    _client._instance = OpenAI(**kwargs)
    return _client._instance


@app.function(
    image=image,
    secrets=[wandb_secret],
    volumes={"/data": vol},
    timeout=600,
    max_containers=20,  # cap parallel spend / W&B rate pressure (Plan Step 4)
    retries=2,  # infra-level retry on top of the in-function attempt loop
)
@modal.concurrent(max_inputs=8)  # generation is API-bound → many calls per container
def generate_one(
    tool_id: str,
    agent_id: str,
    tool_name: str,
    tool_description: str,
    agent_name: str,
    agent_description: str,
    example_idx: int,
    model: str,
) -> dict:
    """Produce one natural-language query for one tool. Label comes from args.

    Retries on transient API errors (with backoff) and on outputs that fail the
    NL contract. Raises after MAX_ATTEMPTS so `.starmap(return_exceptions=True)`
    can record the failure without aborting the whole batch.
    """
    messages = build_generation_messages(
        agent_name, agent_description, tool_name, tool_description, example_idx
    )
    client = _client()

    last_error = "unknown"
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            query_text = normalize_query(response.choices[0].message.content or "")
            is_valid, reason = validate_query(query_text)
            if is_valid:
                return {
                    "query_text": query_text,
                    "agent_id": agent_id,
                    "tool_id": tool_id,
                    "example_idx": example_idx,
                }
            last_error = f"invalid:{reason}"
        except Exception as exc:  # noqa: BLE001 — surface any API/SDK failure to retry
            last_error = f"api_error:{type(exc).__name__}:{exc}"
            time.sleep(min(2**attempt, 30))  # exponential backoff, capped at 30s

    raise RuntimeError(
        f"generate_one failed for {tool_id} idx={example_idx}: {last_error}"
    )


# --- Local helpers (run on your machine inside the entrypoint) ----------------


def _load_registry(path: Path) -> dict:
    return json.loads(path.read_text())


def _select_tools(registry: dict, pilot: bool) -> list[dict]:
    """All tools, or one tool per agent for a cheap pilot (Plan Step 3 sizing)."""
    tools = registry["tools"]
    if not pilot:
        return tools

    first_per_agent: dict[str, dict] = {}
    for tool in tools:
        first_per_agent.setdefault(tool["agent_id"], tool)
    return list(first_per_agent.values())


def _build_jobs(
    tools: list[dict], agent_by_id: dict, examples_per_tool: int, model: str
) -> list[tuple]:
    """Flatten (tool x examples) into starmap-ready argument tuples."""
    return [
        (
            tool["tool_id"],
            tool["agent_id"],
            tool["name"],
            tool["description"],
            agent_by_id[tool["agent_id"]]["name"],
            agent_by_id[tool["agent_id"]]["description"],
            i,
            model,
        )
        for tool in tools
        for i in range(examples_per_tool)
    ]


def _dedup(rows: list[dict]) -> list[dict]:
    """Drop exact (query_text, tool_id) duplicates, keeping the first (Plan Step 4)."""
    seen: set[tuple[str, str]] = set()
    unique = []
    for row in rows:
        key = (row["query_text"], row["tool_id"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def _split_rows(rows: list[dict]) -> list[dict]:
    """Stratified 80/10/10 split by tool_id; tags each row with row['split']."""
    by_tool: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_tool[row["tool_id"]].append(row)

    rng = random.Random(SPLIT_SEED)
    tagged: list[dict] = []
    for items in by_tool.values():
        rng.shuffle(items)
        n = len(items)
        n_test = int(n * TEST_FRAC)
        n_val = int(n * VAL_FRAC)
        for i, row in enumerate(items):
            if i < n_test:
                split = "test"
            elif i < n_test + n_val:
                split = "val"
            else:
                split = "train"
            tagged.append({**row, "split": split})
    return tagged


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for obj in lines:
            f.write(json.dumps(obj) + "\n")


def _write_probe_files(rows: list[dict], out_dir: Path) -> list[Path]:
    """Write raw + per-probe, per-split JSONL files. Returns produced paths.

    Same rows, two label columns (Plan: one training set, two probes):
      agent_probe_*  → label = agent_id
      tool_probe_*   → label = tool_id
    """
    produced: list[Path] = []

    raw_path = out_dir / "raw_dataset.jsonl"
    _write_jsonl(raw_path, rows)
    produced.append(raw_path)

    for probe, label_key in (("agent_probe", "agent_id"), ("tool_probe", "tool_id")):
        for split in ("train", "val", "test"):
            split_rows = [r for r in rows if r["split"] == split]
            lines = [
                {"query_text": r["query_text"], "label": r[label_key]}
                for r in split_rows
            ]
            path = out_dir / f"{probe}_{split}.jsonl"
            _write_jsonl(path, lines)
            produced.append(path)

    return produced


def _persist_to_volume(paths: list[Path]) -> None:
    """Single-writer upload of the JSONL files to the Volume root (/data on mount).

    Avoids the v1 concurrent-commit limit by uploading everything in one batch
    instead of committing per worker.
    """
    with vol.batch_upload(force=True) as batch:
        for path in paths:
            batch.put_file(str(path), f"/{path.name}")


@app.local_entrypoint()
def main(
    examples_per_tool: int = 50,  # pilot default per Plan Step 3 sizing
    pilot: bool = True,  # True → 5 tools (one per agent); --no-pilot → all 200
    model: str = DEFAULT_MODEL,
    registry_path: str = "data/registry.json",
    out_dir: str = "data",
    persist_volume: bool = True,  # also push JSONL to the Modal Volume
):
    """Orchestrate generation, validate, split, and write the dataset.

    PILOT NOTES (confirmed first run = 250 rows):
      - `pilot=True` keeps one tool per agent → 5 tools.
      - 5 tools x examples_per_tool(50) = 250 generation jobs.
      - Run `modal run modal_app/generate_dataset.py` with no flags to get exactly
        this. Scale up later with `--no-pilot --examples-per-tool 75` (~15k rows).
    """
    registry = _load_registry(Path(registry_path))
    agent_by_id = {a["agent_id"]: a for a in registry["agents"]}

    tools = _select_tools(registry, pilot)
    jobs = _build_jobs(tools, agent_by_id, examples_per_tool, model)

    print(
        f"[generate] mode={'pilot' if pilot else 'full'} "
        f"tools={len(tools)} examples/tool={examples_per_tool} "
        f"jobs={len(jobs)} model={model}"
    )

    # Parallel fan-out. return_exceptions=True keeps failed jobs from aborting the
    # batch — we filter and report them below (Plan Step 4 retry/filter guidance).
    results = list(generate_one.starmap(jobs, return_exceptions=True))

    rows = [r for r in results if isinstance(r, dict)]
    failures = [r for r in results if not isinstance(r, dict)]
    rejection_rate = len(failures) / len(results) if results else 0.0
    print(
        f"[generate] returned={len(results)} ok={len(rows)} "
        f"failed={len(failures)} rejection_rate={rejection_rate:.1%}"
    )

    rows = _dedup(rows)
    rows = _split_rows(rows)

    counts = defaultdict(int)
    for r in rows:
        counts[r["split"]] += 1
    print(
        f"[generate] after dedup+split: total={len(rows)} "
        f"train={counts['train']} val={counts['val']} test={counts['test']}"
    )

    produced = _write_probe_files(rows, Path(out_dir))
    print("[generate] wrote:")
    for path in produced:
        print(f"  - {path}")

    if persist_volume:
        _persist_to_volume(produced)
        print(f"[generate] uploaded {len(produced)} files to volume '{VOLUME_NAME}'")
