import json

from _6_read_result_file import extract_response_file


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

if __name__ == '__main__':
        split = "valid"
        ori_id_report_json = extractReport(
                f"/data/sc159/data/MIMIC_III/llava_rad/chat_{split}_MIMIC_CXR_all_gpt4extract_rulebased_v1.json")

        processed_id_report_json = extract_response_file(
                "/data/sc159/LLaVARad/MyData/ReportProcess/chexpert_labels/prompts/valid_prompts_1000_2000_response.jsonl")

        for id in processed_id_report_json.keys():
            print("="*10)
            print(f"id:{id}")
            ori_report=ori_id_report_json[id].replace('. ','.\n    ')
            print(f"ori:\n    {ori_report}")
            print(f"processed:")
            for topic_sentence_json in processed_id_report_json[id]:
                print(f"  {topic_sentence_json['topic']}")
                for sent in topic_sentence_json['sentences']:
                    print(f"    -{sent}")