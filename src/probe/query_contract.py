"""Natural-language query contract — the single source of truth for probe input.

Plan Step 2: train and inference must feed the model the *same kind of input* —
plain natural language, exactly what a SOC analyst would type. No tool schemas,
no labeled-field tickets, no JSON. Every stage (Modal generator, probe training,
probe inference, CrewAI `{context}`) imports `normalize_query` from here so the
string is identical everywhere.

This module is intentionally dependency-free (stdlib only) so it can be imported
both locally and inside a Modal container without extra image layers.
"""

import re

# --- Query validation thresholds (Plan Step 4: "Validation before training") ---
MIN_QUERY_CHARS = 20  # shorter than this is almost never a real analyst message

# Labeled-field prefixes that signal a structured ticket leaked through. The probe
# must never see these — train/inference are natural language only.
_LABELED_FIELD = re.compile(
    r"^\s*(type|severity|indicators?|priority|status|category|summary|"
    r"description|source|destination|new soc event)\s*:",
    re.IGNORECASE | re.MULTILINE,
)

# A response the model wrapped in a ```fence``` — strip it during normalization.
_CODE_FENCE = re.compile(r"^```[a-zA-Z]*\n?|\n?```$")

# Collapse any run of whitespace (including newlines) into single spaces.
_WHITESPACE = re.compile(r"\s+")


def normalize_query(text: str) -> str:
    """Light cleanup only — do NOT reformat into a structured ticket.

    Removes wrapping code fences and surrounding quotes that small instruct
    models sometimes add, then collapses whitespace and strips. The semantic
    content is left untouched so the same string is used at train and inference.
    """
    cleaned = _CODE_FENCE.sub("", text.strip()).strip()

    # Drop a single pair of wrapping quotes ("..." or '...') if present.
    if len(cleaned) >= 2 and cleaned[0] in "\"'" and cleaned[-1] == cleaned[0]:
        cleaned = cleaned[1:-1].strip()

    return _WHITESPACE.sub(" ", cleaned).strip()


def validate_query(text: str) -> tuple[bool, str]:
    """Check a normalized query against the Plan Step 4 rejection rules.

    Returns (is_valid, reason). `reason` is empty when valid, otherwise a short
    machine-friendly tag for aggregating rejection stats.
    """
    if not text or len(text) < MIN_QUERY_CHARS:
        return False, "too_short"
    if _LABELED_FIELD.search(text):
        return False, "labeled_fields"
    # A bare JSON blob in query_text breaks the NL contract.
    if text.startswith("{") or text.startswith("["):
        return False, "looks_like_json"
    return True, ""


# --- Generation prompt (Plan Step 3) -----------------------------------------
# The label is fixed BEFORE generation by the job args. We never ask the model to
# classify or pick a tool — it only writes a realistic message that *requires*
# the given tool. This keeps ground-truth labels clean (Plan "Common mistake #3").

GENERATION_SYSTEM_PROMPT = (
    "You are simulating tier-1 SOC (Security Operations Center) analysts writing "
    "short, realistic messages into a triage system. You write like a busy human, "
    "never like a ticket template."
)

# Rotating scenario angles so parallel jobs for the same tool diverge in wording
# and framing instead of collapsing to one phrasing. Indexed by example_idx.
_SCENARIO_ANGLES = [
    "You just noticed this yourself and are asking for help.",
    "A user or another team escalated this to you; relay it in your own words.",
    "An automated alert fired and you are summarizing it conversationally.",
    "You are mid-investigation and need this one specific check next.",
    "It is a busy shift; keep it terse and slightly informal.",
    "You are double-checking something that looks suspicious but might be benign.",
    "You are handing off and flagging this for whoever picks it up.",
    "A VIP or critical asset is involved and there is mild urgency.",
]


def build_generation_messages(
    agent_name: str,
    agent_description: str,
    tool_name: str,
    tool_description: str,
    example_idx: int,
) -> list[dict]:
    """Build the chat messages for one generation job (one query for one tool).

    `example_idx` selects a scenario angle (deterministic) so a tool's many
    examples vary in tone and detail rather than repeating.
    """
    angle = _SCENARIO_ANGLES[example_idx % len(_SCENARIO_ANGLES)]

    user_prompt = f"""Write ONE short SOC analyst message (1-4 sentences) that a human would type into a triage agent.

The message MUST require this specialist tool:
  agent: {agent_name} - {agent_description}
  tool: {tool_name} - {tool_description}

Scenario angle: {angle}

Rules:
- Natural language only - write like a person, not a ticket template
- Include concrete details (IPs, hostnames, hashes, usernames) where relevant
- Use realistic RFC 5737 IPs (192.0.2.x, 198.51.100.x, 203.0.113.x); vary wording and scenario
- Do NOT use labeled fields (Type:, Severity:, etc.) or JSON
- Do NOT name the tool or agent explicitly; describe the situation, not the tooling
- Return only the message text, no markdown fences, no quotes"""

    return [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
