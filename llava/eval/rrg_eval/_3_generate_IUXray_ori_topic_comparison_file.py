"""
# before, the concat_topic.py file is to generate topic_gt vs pred_gt in report level.
# now, this code is generate ori_gt vs pred_gt in report level.

The following operation could be running the commands below:
cd llava/eval/rrg_eval
python run.py the_output.jsonl --output_dir /path/to/some_dir/
"""
import os
import json
from collections import defaultdict


def process_jsonl_files(concated_topic_gt_pred_file, ori_report_file, output_file):
    """
    For each id in concated_topic_gt_pred_file,
    find the corresponding original report in ori_report_file.
    Save them to the merged_data

    """
    merged_data = defaultdict(lambda: {"references": [], "predictions": []})

    ori_id_reference_json = {}
    with open(ori_report_file, 'r', encoding='utf-8') as f:
        reports = json.load(f)
        for cur_report in reports:
            cur_id = cur_report['id']
            cur_reference = cur_report['conversations'][1]['value']
            if cur_id in ori_id_reference_json:
                continue
            else:
                ori_id_reference_json[cur_id] = cur_reference

    with open(concated_topic_gt_pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            image_id = entry["id"]

            merged_data[image_id]["references"].append(ori_id_reference_json[image_id])
            merged_data[image_id]["predictions"].append(entry["prediction"])

    # Write output as a new .jsonl file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for image_id, content in merged_data.items():
            out_entry = {
                "id": image_id,
                "reference": " ".join(content["references"]),
                "prediction": " ".join(content["predictions"])
            }
            f_out.write(json.dumps(out_entry, ensure_ascii=False) + "\n")


# Example usage
if __name__ == "__main__":
    # Replace with your actual list of .jsonl files
    concated_topic_gt_pred_file = "/data/sc159/LLaVARad/results/topic_seg/llavarad_IUXray_onlyMLP/test_merged.jsonl"
    ori_report_file = "/data/sc159/data/IU_Xray_raw/processed/llava_rad_topic/all_llavarad_format_v2.json"
    output_file = "/data/sc159/LLaVARad/results/topic_seg/llavarad_IUXray_onlyMLP/oriReport_predTopic.jsonl"
    process_jsonl_files(concated_topic_gt_pred_file, ori_report_file, output_file)
