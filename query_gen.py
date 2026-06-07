import json
import random
import sys
import modal

app = modal.App("tooloptim-query-gen")

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

SYSTEM_PROMPT = """You are a SOC analyst. Generate realistic natural language queries that would cause a security tool to be invoked.
Return ONLY a JSON array of strings."""

def build_prompt(tool: dict) -> str:
    return f"""Tool ID: {tool["tool_id"]}
Description: {tool["description"]}

Generate 50 realistic queries a SOC analyst might use to invoke this tool. Mix these styles:
- Some phrased as urgent triage requests
- Some as routine monitoring checks
- Some as investigations/detections
- Some as remediations
- Some from a junior analyst describing a problem without knowing the tool name

Each query should sound like a natural analyst request, not a restatement of the tool name.

Return a JSON array of exactly 50 strings, no explanation, no markdown, no backticks.
Example format:
["query one", "query two", "query three", "query four"]"""


@app.function(
    image=image,
    gpu="A10G",
    timeout=1200,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate_queries(tool: dict) -> tuple[str, str, list[str]]:
    from vllm import LLM, SamplingParams
    model = LLM(model=MODEL, max_model_len=8192)
    sampling = SamplingParams(temperature=0.7, max_tokens=4096)
    prompt = build_prompt(tool)
    outputs = model.generate([prompt], sampling)
    raw = outputs[0].outputs[0].text

    clean = raw.strip()
    try:
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        start = clean.index("[")
        end = clean.rindex("]")
        queries = json.loads(clean[start:end + 1])
    except (ValueError, json.JSONDecodeError):
        print(f"WARNING: failed to parse queries for tool_id={tool['tool_id']}")
        return (tool["tool_id"], tool["agent_id"], [])

    return (tool["tool_id"], tool["agent_id"], queries)


@app.local_entrypoint()
def main(test: bool = False):
    with open("data/registry.json") as f:
        registry = json.load(f)

    if test:
        tools = random.sample(registry["tools"], 3)
        for tool in tools:
            tool_id, agent_id, queries = generate_queries.remote(tool)

            print(f"Tool: {tool_id}")
            print(f"Description: {tool['description']}")
            print("Queries:")
            for q in queries:
                print(f"  - {q}")
    else:
        results = list(generate_queries.map(registry["tools"]))

        rows = []
        for tool_id, agent_id, queries in results:
            for query in queries:
                rows.append({"tool_id": tool_id, "agent_id": agent_id, "query_text": query})

        output_path = "/home/abraham/Desktop/ToolOptim/data/queries_200tool.json"
        with open(output_path, "w") as f:
            json.dump(rows, f, indent=2)

        print(f"Tools processed: {len(results)}")
        print(f"Total queries generated: {len(rows)}")
        print(f"Output written to: {output_path}")
