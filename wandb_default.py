import os
import weave
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

weave.init('abrahambhatti525-santa-clara-university/ToolOptim')

@weave.op
def create_completion(message: str) -> str:
    client = OpenAI(
        base_url='https://api.inference.wandb.ai/v1',
        api_key=os.environ['WANDB_API_KEY'],
        project='abrahambhatti525-santa-clara-university/ToolOptim',
    )
    response = client.chat.completions.create(
        model="OpenPipe/Qwen3-14B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}
        ],
    )
    return response.choices[0].message.content

message = "Tell me a joke."
create_completion(message)