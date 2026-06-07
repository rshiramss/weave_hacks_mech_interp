import json
import modal

app = modal.App("tooloptim-registry-gen")

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

AGENTS = [
    {
        "agent_id": "ITAC",
        "name": "Access Control",
        "description": "Manages user logins, passwords, and account access to stop attackers from breaking in.",
        "seed_tools": [
            {"tool_id": "log_out_user", "description": "Signs a user out of all active sessions."},
            {"tool_id": "suspend_user", "description": "Locks a user account so it can't be used."},
            {"tool_id": "reset_user_2fa", "description": "Clears a user's two-factor login devices."},
            {"tool_id": "list_logged_in_users", "description": "Shows who is currently logged in and from where."},
            {"tool_id": "get_login_history", "description": "Shows recent login attempts for a user."},
            {"tool_id": "list_user_groups", "description": "Shows which groups a user belongs to."},
            {"tool_id": "remove_user_from_group", "description": "Takes a user out of a group."},
            {"tool_id": "reset_user_password", "description": "Forces a user to set a new password."},
        ]
    },
    {
        "agent_id": "EDRH",
        "name": "Endpoint Defense",
        "description": "Watches laptops and servers for attacks and shuts down threats fast.",
        "seed_tools": [
            {"tool_id": "isolate_device", "description": "Cuts a device off from the network to stop an attack."},
            {"tool_id": "reconnect_device", "description": "Restores network access to a device."},
            {"tool_id": "list_running_programs", "description": "Shows what programs are running on a device."},
            {"tool_id": "remove_bad_files", "description": "Deletes harmful files from a device."},
            {"tool_id": "list_network_connections", "description": "Shows what a device is connecting to online."},
            {"tool_id": "download_device_logs", "description": "Grabs activity logs from a device for review."},
            {"tool_id": "get_device_info", "description": "Shows basic details about a device."},
            {"tool_id": "stop_program", "description": "Shuts down a suspicious program on a device."},
        ]
    },
    {
        "agent_id": "TIRA",
        "name": "Threat Intel",
        "description": "Checks files, links, and addresses to see if they're dangerous.",
        "seed_tools": [
            {"tool_id": "check_file_safety", "description": "Checks if a file is known to be malicious."},
            {"tool_id": "check_ip_address", "description": "Checks if an internet address is dangerous."},
            {"tool_id": "check_website", "description": "Checks if a website is known to be malicious."},
            {"tool_id": "scan_link", "description": "Tests a suspicious link in a safe sandbox."},
            {"tool_id": "get_link_scan_results", "description": "Shows results from a previous link scan."},
            {"tool_id": "look_up_hacker_group", "description": "Shows known info about a hacking group."},
            {"tool_id": "get_website_history", "description": "Shows past internet addresses used by a website."},
            {"tool_id": "scan_file", "description": "Tests a suspicious file in a safe sandbox."},
        ]
    },
    {
        "agent_id": "SSCA",
        "name": "Supply Chain",
        "description": "Checks code and software for security problems before they cause harm.",
        "seed_tools": [
            {"tool_id": "list_vulnerable_packages", "description": "Shows software packages with known security holes."},
            {"tool_id": "list_code_warnings", "description": "Shows security problems found in the code."},
            {"tool_id": "create_security_notice", "description": "Writes up a warning about a security problem."},
            {"tool_id": "list_leaked_passwords", "description": "Shows passwords or keys accidentally posted in code."},
            {"tool_id": "list_repo_members", "description": "Shows who has access to a code repository."},
            {"tool_id": "check_merge_rules", "description": "Checks the rules for approving code changes."},
            {"tool_id": "check_commit_signature", "description": "Checks whether a code change was signed by its author."},
            {"tool_id": "list_security_rules", "description": "Shows the company's security checklist for code."},
        ]
    },
    {
        "agent_id": "SICL",
        "name": "Incident Comms",
        "description": "Alerts the right people and keeps everyone updated during a security incident.",
        "seed_tools": [
            {"tool_id": "send_alert", "description": "Posts a warning message to a chat channel."},
            {"tool_id": "open_incident_room", "description": "Creates a chat room for handling an incident."},
            {"tool_id": "send_email_alert", "description": "Emails an urgent warning to admins."},
            {"tool_id": "check_for_approval", "description": "Checks if an admin has approved an action by email."},
            {"tool_id": "close_incident_room", "description": "Closes and archives a finished incident chat room."},
            {"tool_id": "add_person_to_incident", "description": "Brings someone into an active incident chat."},
            {"tool_id": "open_ticket", "description": "Creates a tracking ticket for an incident."},
            {"tool_id": "update_ticket", "description": "Adds notes and progress to an incident ticket."},
        ]
    },
]

SYSTEM_PROMPT = """You are a security engineering architect designing a SOC automation platform.
Your job is to generate realistic, distinct security tool definitions in JSON format.
Return ONLY valid JSON, no explanation, no markdown, no backticks."""

def build_prompt(agent: dict) -> str:
    seed_json = json.dumps(agent["seed_tools"], indent=2)
    return f"""Agent: {agent["name"]}
Description: {agent["description"]}

Here are 8 existing tools for this agent:
{seed_json}

Generate 32 MORE tools for this agent, written for a live security dashboard that
non-technical staff (support, ops, managers) will read and click on. Each tool must:
- Be immediately understandable to a non-technical audience
- Have a tool_id that is self-explanatory in plain English (e.g. suspend_user, send_alert, block_ip)
- Have a description that is one short sentence, max 10 words, that a non-engineer could read and instantly understand what it does
- Avoid jargon like "telemetry", "cryptographic", "entitlement auditing", "normalization"
- Be things you'd see in a live security dashboard — actions and lookups that feel real and immediate
- Be genuinely distinct from the 8 existing tools and from each other
- Still be realistic for a {agent["name"]} in an enterprise SOC environment
- NOT overlap with tools from other SOC domains (identity, endpoint, threat intel, supply chain, comms)

Return a JSON array of exactly 32 objects, each with "tool_id" and "description" fields only.
Example format:
[
  {{"tool_id": "block_ip", "description": "Stops traffic from a dangerous address."}}
]"""


@app.function(
    image=image,
    gpu="A10G",
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate_tools(agent: dict) -> tuple[str, str]:
    from vllm import LLM, SamplingParams
    model = LLM(model=MODEL, max_model_len=8192)
    sampling = SamplingParams(temperature=0.7, max_tokens=4096)
    prompt = build_prompt(agent)
    outputs = model.generate([prompt], sampling)
    return (agent["agent_id"], outputs[0].outputs[0].text)


@app.local_entrypoint()
def main():
    # test mode: just run one agent
    test_mode = False
    agents_to_run = AGENTS[:1] if test_mode else AGENTS

    results = generate_tools.map(agents_to_run)

    registry = {"agents": [], "tools": []}

    for agent, (agent_id, result) in zip(agents_to_run, results):
        registry["agents"].append({
            "agent_id": agent["agent_id"],
            "name": agent["name"],
            "description": agent["description"],
        })

        # add seed tools
        for t in agent["seed_tools"]:
            registry["tools"].append({
                "tool_id": t["tool_id"],
                "agent_id": agent["agent_id"],
                "description": t["description"],
                "integration": "real" if t["tool_id"] in [
                    "send_alert", "open_incident_room",
                    "send_email_alert", "check_for_approval",
                    "list_vulnerable_packages", "list_code_warnings",
                    "list_leaked_passwords", "list_repo_members",
                ] else "mock"
            })

        # parse and add generated tools
        try:
            # strip any markdown fences if model added them
            clean = result.strip()
            if "```" in clean:
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            generated = json.loads(clean)
            for t in generated:
                registry["tools"].append({
                    "tool_id": t["tool_id"],
                    "agent_id": agent["agent_id"],
                    "description": t["description"],
                    "integration": "mock"
                })
            print(f"✓ {agent['agent_id']}: {len(generated)} tools generated")
        except Exception as e:
            print(f"✗ {agent['agent_id']}: failed to parse — {e}")
            print(f"Raw output: {result[:500]}")

    out_path = "/home/abraham/Desktop/ToolOptim/data/registry.json"
    with open(out_path, "w") as f:
        json.dump(registry, f, indent=2)

    total_tools = len(registry["tools"])
    print(f"\nSaved registry to {out_path}")
    print(f"Agents: {len(registry['agents'])}, Tools: {total_tools}")