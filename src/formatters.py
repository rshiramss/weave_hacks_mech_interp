"""Structured SOC event -> natural-language analyst message (Stage 8, §4).

The probe / router contract is plain natural language. If a sim feed or UI emits
a structured alert, convert it to NL exactly ONCE here at the Flow boundary
before any routing — never feed labeled-field tickets to the router.
"""

from src.probe.query_contract import normalize_query


def _join_context(event: dict) -> str:
    """Build trailing context fragments (from/on/for) from common alert fields."""
    fragments = []
    source = event.get("source_ip") or event.get("src_ip") or event.get("ip")
    host = event.get("host") or event.get("hostname") or event.get("asset")
    user = event.get("user") or event.get("username") or event.get("account")
    if source:
        fragments.append(f"from {source}")
    if host:
        fragments.append(f"on {host}")
    if user:
        fragments.append(f"for {user}")
    return " ".join(fragments)


def to_analyst_message(event) -> str:
    """Render a structured alert as a plain-language message an analyst would type.

    Accepts a dict (structured event) or a string (already NL). A pre-written
    `message`/`text` field is used verbatim. Output is normalized NL — no
    labeled fields (Type:, Severity:), JSON, or tool schemas (§4).
    """
    if isinstance(event, str):
        return normalize_query(event)
    if not isinstance(event, dict):
        return normalize_query(str(event))

    for key in ("message", "text", "query_text"):
        if event.get(key):
            return normalize_query(str(event[key]))

    summary = (
        event.get("summary")
        or event.get("title")
        or event.get("description")
        or "Possible security event"
    )
    indicators = event.get("indicators")
    if isinstance(indicators, (list, tuple)):
        indicators = ", ".join(str(i) for i in indicators)

    parts = [str(summary).rstrip(".")]
    context = _join_context(event)
    if context:
        parts.append(context)
    if indicators:
        parts.append(f"indicators: {indicators}")

    message = " ".join(parts).strip() + " — can someone investigate?"
    return normalize_query(message)
