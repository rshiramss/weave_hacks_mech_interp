"""Weave init + cost registration (Stage 5 control plane; reused by Stage 4).

`init_weave()` starts the Weave project once and registers per-token pricing for
the W&B-Inference model so Llama/Qwen calls show $ in traces and eval summaries
(§7.5). Falls back to the bare project name if the configured entity is not
accessible to this account.
"""

import os

WEAVE_PROJECT = os.environ.get("WEAVE_PROJECT", "weavehacks/soc-probe-router")

# Cost is attributed by the model id that appears in traces. Qwen2.5-7B runs
# locally via Ollama, so litellm may report any of these forms; register all so
# $ attaches on Arms 1 & 3 (Arm 2 embedding cost is local/negligible).
LLM_COST_IDS = (
    "ollama_chat/qwen2.5:7b-instruct",
    "ollama/qwen2.5:7b-instruct",
    "qwen2.5:7b-instruct",
    "Qwen/Qwen2.5-7B-Instruct",  # if pointed at a hosted endpoint
)
LLM_ID = LLM_COST_IDS[0]

# Notional $/token (local model has ~no marginal cost; this powers the relative
# cheaper-vs-frontier story). Convert from $/1M tokens.
_PROMPT_COST_PER_TOKEN = 0.20 / 1_000_000
_COMPLETION_COST_PER_TOKEN = 0.20 / 1_000_000

_client = None


def init_weave():
    """Idempotently init Weave and register model cost. Returns the client (or None)."""
    global _client
    if _client is not None:
        return _client

    import weave

    try:
        _client = weave.init(WEAVE_PROJECT)
    except Exception as exc:  # noqa: BLE001 — entity may not exist for this user
        bare = WEAVE_PROJECT.split("/")[-1]
        try:
            _client = weave.init(bare)
            print(f"[weave] entity fallback -> project '{bare}'")
        except Exception:
            print(f"[weave] WARNING: weave.init failed, no tracing: {exc}")
            return None

    for llm_id in LLM_COST_IDS:
        try:
            _client.add_cost(
                llm_id=llm_id,
                prompt_token_cost=_PROMPT_COST_PER_TOKEN,
                completion_token_cost=_COMPLETION_COST_PER_TOKEN,
            )
        except Exception as exc:  # noqa: BLE001 — add_cost is best-effort
            print(f"[weave] add_cost skipped for {llm_id}: {exc}")

    return _client
