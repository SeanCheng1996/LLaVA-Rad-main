from openai import OpenAI
from glob import glob
import os

for batch_input_file_id_path in glob(
        "/data/sc159/LLaVARad/MyData/ReportProcess/chexpert_organ_labels/prompts/test*_batch_file_id.txt"):
    # for batch_input_file_id_path in glob("/data/sc159/LLaVARad/MyData/ReportProcess/chexpert_organ_labels/prompts/*_batch_file_id.txt"):
    # batch_input_file_id_path = "/data/sc159/LLaVARad/MyData/ReportProcess/organ_labels/prompts/valid_prompts_1000_2000_batch_file_id.txt"

    if os.path.exists(batch_input_file_id_path.replace("batch_file_id.txt", "response.jsonl")):
        continue
    else:
        print(f"creating for {batch_input_file_id_path}")

    with open(batch_input_file_id_path, 'r') as f:
        batch_input_file_id = f.readline().strip()
    print(batch_input_file_id)

    client = OpenAI(
        api_key="sk-mfsoglkmixpwsdirmpwetxphavpespdrihkapsuueifntnet",
        base_url="https://api.siliconflow.cn/v1"
    )

    res = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"processing report {batch_input_file_id}"
        },
        extra_body={"replace": {"model": "deepseek-ai/DeepSeek-V3"}}
    )
    print(res)
    with open(batch_input_file_id_path.replace("file_id.txt", "batch_id.txt"), 'w') as f:
        print(str(res), file=f)
