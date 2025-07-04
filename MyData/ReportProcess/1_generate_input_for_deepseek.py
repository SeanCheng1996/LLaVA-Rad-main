"""
1. Extract report from llavaRad.
2. Generate prompt for Deepseek, including system prompt and the report
3. Save the generated prompt
"""
import json
import os

from tqdm import tqdm


def extractReport(file_path):
    """
    file_path: llavaRad json file

    return:
    id_report_json: {unique report id : report, ... }
    """

    id_report_json = {}
    with open(file_path, 'r') as f:
        reports = json.load(f)

    for report in reports:
        cur_id = report['id']

        if cur_id in id_report_json:
            continue

        id_report_json[cur_id] = report["conversations"][1]["value"]

    return id_report_json


def generate_prompt_per_report(id, report, system_prompt):
    """
    Below is an example as required by "https://docs.siliconflow.cn/cn/userguide/guides/batch".
    {
        "custom_id": "request-1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
                    "model": "deepseek-ai/DeepSeek-V3",
                    "messages":
                        [
                            {
                                "role": "system",
                                "content": "You are a highly advanced and versatile AI assistant"
                            },
                            {
                                "role": "user",
                                "content": "How does photosynthesis work?"
                            }
                        ],
                    "stream": true,
                    "max_tokens": 1514
                }
    }
    """
    prompt = {
        "custom_id": id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "deepseek-ai/DeepSeek-V3",
            "messages":
                [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": report
                    }
                ],
            "stream": True,
            "temperature": 0.6,
            "max_tokens": 1514
        }
    }
    return prompt


def get_system_prompt(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    return content


def generate_prompt(id_report_json, system_prompt, start_index, rows=1000):
    prompts = []
    ids = sorted(list(id_report_json.keys()))[start_index:start_index + rows]
    for id in tqdm(ids, desc="Generating prompts"):
        cur_prompt = generate_prompt_per_report(id, id_report_json[id], system_prompt)
        prompts.append(cur_prompt)
    return prompts


if __name__ == '__main__':
    system_prompt_cate = "chexpert_organ_labels"
    system_prompt = get_system_prompt(
        f"/data/sc159/LLaVARad/MyData/ReportProcess/{system_prompt_cate}/report_prompt.txt")

    split = "test"
    id_report_json = extractReport(
        f"/data/sc159/data/MIMIC_III/llava_rad/chat_{split}_MIMIC_CXR_all_gpt4extract_rulebased_v1.json")

    interval = 5000
    for i in tqdm(range(0, len(id_report_json), interval), desc="Processing batches"):
        prompts = generate_prompt(id_report_json, system_prompt, start_index=i, rows=interval)
        dest_file = f"/data/sc159/LLaVARad/MyData/ReportProcess/{system_prompt_cate}/prompts/{split}_prompts_{i}_{i + interval}.jsonl"
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
        with open(dest_file,
                  'w') as f:
            for prompt in prompts:
                f.write(json.dumps(prompt) + "\n")
