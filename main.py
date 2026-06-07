import os

import weave
from dotenv import load_dotenv
from openai import OpenAI

def format_prompt(user_message: str) -> list:
    """
    Format the prompt for the OpenAI chat completion API.

    Args:
        user_message (str): The user's message for the assistant.

    Returns:
        list: A list of messages formatted for the model.
    """
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_message},
    ]


load_dotenv()

WANDB_PROJECT = os.environ.get(
    "WANDB_PROJECT", "abrahambhatti525-santa-clara-university/ToolOptim"
)

if not os.environ.get("WANDB_API_KEY"):
    raise RuntimeError(
        "WANDB_API_KEY is not set. Get your API key from https://wandb.ai/settings, "
        "then either:\n"
        "  1. Create a .env file with: WANDB_API_KEY=your_key_here\n"
        "  2. Or run: wandb login\n"
        "Also ensure WANDB_PROJECT points to a project you have access to."
    )

# weave.init must run before OpenAI calls so Weave can auto-trace them.
weave.init(WANDB_PROJECT)

client = OpenAI(
    base_url="https://api.inference.wandb.ai/v1",
    api_key=os.environ["WANDB_API_KEY"],
    project=WANDB_PROJECT,
)


@weave.op()
def create_completion(message: str) -> str:
    prompt = format_prompt(message)
    response = client.chat.completions.create(
        model="OpenPipe/Qwen3-14B-Instruct",
        messages=prompt,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    result = create_completion("Tell me a joke.")
    print(result)

    weave_client = weave.get_client()
    if weave_client is not None:
        weave_client.flush()
