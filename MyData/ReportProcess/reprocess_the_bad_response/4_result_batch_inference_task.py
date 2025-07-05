import json

from openai import OpenAI
import requests
import os
from glob import glob
import time

def download_file(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True
    except Exception as e:
        print("download error")
        return False


if __name__ == '__main__':
    # for batch_id_path in glob("/data/sc159/LLaVARad/MyData/ReportProcess/chexpert_organ_labels/prompts/test*_batch_batch_id.txt"):
    batch_id_path = "/data/sc159/LLaVARad/MyData/ReportProcess/chexpert_organ_labels/re_for_bad_response/train_prompts_bad_0_5000_batch_batch_id.txt"
    print("=" * 10)
    print(batch_id_path)

    with open(batch_id_path, 'r') as f:
        content = f.readline().strip()
        start = content.find("id='") + 4
        end = content.find("'", start)
        batch_id = content[start:end]
    print(batch_id)

    client = OpenAI(
        api_key="sk-mfsoglkmixpwsdirmpwetxphavpespdrihkapsuueifntnet",
        base_url="https://api.siliconflow.cn/v1"
    )

    batch = client.batches.retrieve(batch_id)
    print(batch)


    if batch.status == 'completed':
        print("downloading")
        download_file(batch.output_file_id, batch_id_path.replace("batch_batch_id.txt", "response.jsonl"))
    else:
        print("Not finished, status:", batch.status)

    # time.sleep(1)

