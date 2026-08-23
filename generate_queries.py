import os
import json
import random
import time
from openai import OpenAI

# Support both env var names
api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPEN_AI_KEY")
if not api_key:
    raise RuntimeError("No API key found. Set OPENAI_API_KEY or OPEN_AI_KEY.")

client = OpenAI(api_key=api_key)

TOOLS = [
    "read_file",
    "write_file",
    "list_directory",
    "delete_file",
    "compress_files",
    "web_search",
    "image_search",
    "news_search",
    "academic_search",
    "video_search",
    "send_email",
    "send_sms",
    "post_slack",
    "create_calendar_event",
    "create_meeting",
    "execute_python",
    "execute_bash",
    "run_unit_tests",
    "format_code",
    "lint_code",
]

TOOL_DESCRIPTIONS = {
    "read_file": "opens and reads the contents of a file on disk",
    "write_file": "creates or overwrites a file with given content",
    "list_directory": "lists files and folders in a directory",
    "delete_file": "permanently removes a file or folder",
    "compress_files": "zips or archives files into a compressed bundle",
    "web_search": "searches the internet for general information",
    "image_search": "searches for images online",
    "news_search": "finds recent news articles on a topic",
    "academic_search": "finds scholarly papers and academic publications",
    "video_search": "searches for videos on the internet",
    "send_email": "composes and sends an email message",
    "send_sms": "sends a text message to a phone number",
    "post_slack": "posts a message to a Slack channel or user",
    "create_calendar_event": "adds an event or appointment to a calendar",
    "create_meeting": "schedules a video or phone meeting with participants",
    "execute_python": "runs a Python script or code snippet",
    "execute_bash": "runs a shell command or bash script",
    "run_unit_tests": "executes the test suite for a codebase",
    "format_code": "auto-formats source code to match style conventions",
    "lint_code": "analyzes code for errors, warnings, and style issues",
}

PROMPT_TEMPLATE = """You are generating training data for an AI agent that routes user requests to tools.

Tool: {tool}
Description: {description}

Generate as many realistic, natural-language queries as you can (aim for 200) that a real user would send to an AI assistant when they need this tool's capability.

Rules:
- Do NOT mention the tool name or technical function names in the query
- Queries should sound like things a real person would actually type
- Include ambiguity and variety: casual phrasing, domain-specific contexts, different user types (developer, manager, student, etc.)
- Vary query length: some short (under 8 words), some medium, some longer and descriptive
- Cover diverse real-world use cases and scenarios for this tool
- Do not make it obvious which tool is needed — the query should describe the desired outcome, not the mechanism

Return ONLY a JSON array of strings, no other text, no markdown, no explanation.
Example format: ["query one", "query two", ...]"""


def generate_queries_for_tool(tool: str) -> list[str]:
    prompt = PROMPT_TEMPLATE.format(
        tool=tool,
        description=TOOL_DESCRIPTIONS[tool],
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        max_tokens=4096,
    )
    content = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if content.startswith("```"):
        lines = content.splitlines()
        content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    queries = json.loads(content)
    if not isinstance(queries, list):
        raise ValueError(f"Expected list, got {type(queries)} for tool {tool}")
    return [str(q) for q in queries]


def main():
    all_entries: list[dict] = []

    for i, tool in enumerate(TOOLS, 1):
        print(f"[{i:2d}/{len(TOOLS)}] Generating queries for: {tool} ...", end=" ", flush=True)
        retries = 3
        for attempt in range(retries):
            try:
                queries = list(dict.fromkeys(generate_queries_for_tool(tool)))
                for q in queries:
                    all_entries.append({"query": q, "tool": tool})
                print(f"done ({len(queries)} queries)")
                break
            except Exception as e:
                if attempt < retries - 1:
                    wait = 2 ** attempt * 3
                    print(f"error ({e}), retrying in {wait}s...", end=" ", flush=True)
                    time.sleep(wait)
                else:
                    print(f"FAILED after {retries} attempts: {e}")
                    raise

    random.shuffle(all_entries)

    output_path = os.path.join(os.path.dirname(__file__), "queries.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_entries)} queries to {output_path}")


if __name__ == "__main__":
    main()
