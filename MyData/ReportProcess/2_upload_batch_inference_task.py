from openai import OpenAI
from glob import glob

for file_path in glob("/data/sc159/LLaVARad/MyData/ReportProcess/chexpert_organ_labels/prompts/*.jsonl"):
    # file_path = "/data/sc159/LLaVARad/MyData/ReportProcess/organ_labels/prompts/valid_prompts_1000_2000.jsonl"

    client = OpenAI(
        api_key="sk-mfsoglkmixpwsdirmpwetxphavpespdrihkapsuueifntnet",
        base_url="https://api.siliconflow.cn/v1"
    )

    batch_input_file = client.files.create(
        file=open(file_path, "rb"),
        purpose="batch"
    )
    print(batch_input_file)
    file_id = batch_input_file.data['id']
    print(file_id)
    with open(file_path.replace(".jsonl", "_batch_file_id.txt"), 'w') as f:
        print(str(file_id), file=f)
