import json
from openai import OpenAI


def get_all_prompts(file_path):
    prompts = []
    with open(file_path, 'r') as f:
        for row in f:
            cur_prompt = json.loads(row)
            prompts.append(cur_prompt)
    return prompts


prompts = get_all_prompts("/data/sc159/LLaVARad/MyData/ReportProcess/prompts/valid_prompts_0_1000.jsonl")

for prompt in prompts:
    if prompt['custom_id'] != "12641488_51982061":
        continue
    picked_prompt_message = prompt['body']['messages']
print(picked_prompt_message)

# send the request
client = OpenAI(api_key="sk-mfsoglkmixpwsdirmpwetxphavpespdrihkapsuueifntnet", base_url="https://api.siliconflow.cn/v1")
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3",
    messages=picked_prompt_message,
    temperature=0.6,
    max_tokens=4096
)
print(response)
