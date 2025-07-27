import json
from tqdm import tqdm
from _6_read_result_file import extract_response_file
import matplotlib.pyplot as plt
from collections import Counter


def extractReport(file_path):
    """
    file_path: llavaRad json file

    return:
    id_report_json:
    {
        unique report id :
            {
            image: "",
            ori:"",
            topics: ["","",...],
            topic_sents: ["", "",...]
            }
    }
    """

    id_report_json = {}
    with open(file_path, 'r') as f:
        reports = json.load(f)

    for report in reports:
        cur_id = report['id']

        if cur_id not in id_report_json:
            id_report_json[cur_id] = {"image": report['image'], "ori": report['conversations'][1]['value']}

        if "topics" not in id_report_json[cur_id]:
            id_report_json[cur_id]['topics'] = []
        id_report_json[cur_id]['topics'].append(report['topic'])

        if "topic_sents" not in id_report_json[cur_id]:
            id_report_json[cur_id]['topic_sents'] = []
        id_report_json[cur_id]['topic_sents'].append(report['conversations'][2]['value'])

    return id_report_json


if __name__ == '__main__':
    file_path = "/data/sc159/data/MIMIC_III/llava_rad_topic/chat_test_MIMIC_CXR_all_gpt4extract_rulebased_v3.json"
    save_path = "/data/sc159/data/MIMIC_III/llava_rad_topic/chat_test_MIMIC_CXR_all_gpt4extract_rulebased_v4.json"

    # stats how many repeated topics
    res = []
    visited_id_topics_json = {}  # {id: set(topics), ...}
    with open(file_path, 'r') as f:
        ori_res = json.load(f)
        print(f"previous len: {len(ori_res)}")

        for report in ori_res:
            id = report['id']
            topic = report['topic']
            if id in visited_id_topics_json and topic in visited_id_topics_json[id]:  # repeated
                continue
            else:  # new
                if id not in visited_id_topics_json:
                    visited_id_topics_json[id] = set()
                visited_id_topics_json[id].add(topic)
                res.append(report)


    print(f"current len: {len(res)}")
    with open(save_path, 'w') as f:
        json.dump(res, f, indent=2)
