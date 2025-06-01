import json
from openai import OpenAI

from envs.kernelbench import KernelBenchEnv

"""
Evaluating multi-turn reasoning before/after training.

CUDA_VISIBLE_DEVICES=0,1 vllm serve 'Qwen/Qwen2.5-7B-Instruct' --tensor_parallel_size 2 --max_model_len 8192 --dtype bfloat16 \
    --gpu_memory_utilization 0.9 --enable_prefix_caching \
    --host 0.0.0.0 --port 8001

uv run verifiers/examples/demo_kernelbench.py
"""

PROMPT = """
Think step-by-step inside <reasoning>...</reasoning> tags, then either write your code inside <code>...</code> tags, or give your final answer inside <answer>...</answer> tags.
You will be given a prompt containing a code snippet. This code snippet contains a reference model, named `Model`. Your response has to include a `torch.nn.Module` class called `NewModel` that implements the same functionality as the reference model. Your response has to include all required imports. It shouldn't contain any other definitions, after finishing your response 
in <code>...</code> tags, you will be given a response containing how your code performed against the reference model. When you are done, give your final answer inside <answer>...</answer> tags.
"""


vf_env = KernelBenchEnv(
    system_prompt=PROMPT,
    few_shot=[],
    max_steps=5,
)
print(vf_env.system_prompt)

model_name = "Qwen/Qwen2.5-7B-Instruct"
base_url = "http://0.0.0.0:8001/v1"
client = OpenAI(base_url=base_url, api_key="EMPTY")
results, rewards = vf_env.eval_api(
    client, model_name, max_concurrent=20, sampling_args={"temperature": 0.6}
)


with open("results.json", "w") as f:
    json.dump(results, f)
