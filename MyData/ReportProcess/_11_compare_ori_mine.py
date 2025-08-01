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
    split = "valid"
    processed_id_report_json = extractReport(
        "/data/sc159/data/IU_Xray_raw/processed/llava_rad_topic/all_llavarad_format_v1.json")

    # show llavarad and my report side by side
    for id in list(processed_id_report_json.keys())[:50]:
        print("=" * 60)
        print(f"id:{id}")
        ori_report = processed_id_report_json[id]['ori'].replace('. ', '.\n    ')
        print(f"ori:\n    {ori_report}")
        print(f"processed:")
        for topic, topic_sents in zip(processed_id_report_json[id]['topics'],
                                      processed_id_report_json[id]['topic_sents']):
            print(f"  {topic}")
            print(f"    -{topic_sents}")

    # show specific id
    # id="17398573_51909919"
    # print("=" * 60)
    # print(f"id:{id}")
    # ori_report = processed_id_report_json[id]['ori'].replace('. ', '.\n    ')
    # print(f"ori:\n    {ori_report}")
    # print(f"processed:")
    # for topic, topic_sents in zip(processed_id_report_json[id]['topics'],
    #                               processed_id_report_json[id]['topic_sents']):
    #     print(f"  {topic}")
    #     print(f"    -{topic_sents}")

    # stats how many repeated topics
    diff_cnts = []
    for id in tqdm(list(processed_id_report_json.keys())):
        topics_list = processed_id_report_json[id]['topics']
        diff = len(topics_list) - len(set(topics_list))
        if diff > 0:
            diff_cnts.append(diff)
            print("=" * 60)
            for i in range(len(topics_list)):
                print(f"{topics_list[i]}:     {processed_id_report_json[id]['topic_sents'][i]}")

    counts = Counter(diff_cnts)
    numbers = list(counts.keys())
    frequencies = list(counts.values())

    plt.bar(numbers, frequencies, color='skyblue')

    plt.title(f'Redundant topics frequency. {len(diff_cnts)}')
    plt.xlabel('#Redundant topics')
    plt.ylabel('Frequency')
    plt.show()

    # # generate a jsonl in the format of prediction, so that the evaluate code can be applied on the processed reports.
    # with open("/data/sc159/data/MIMIC_III/llava_rad_topic/evaluate/llavarad_vs_topic.jsonl", 'w',
    #           encoding='utf-8') as f_out:
    #     for id in tqdm(list(processed_id_report_json.keys())):
    #         ori_report = processed_id_report_json[id]['ori']
    #         topic_report = " ".join(processed_id_report_json[id]['topic_sents'])
    #         image = processed_id_report_json[id]['image']
    #         out_entry = {
    #             "id": id,
    #             "image": image,
    #             "reference": ori_report,
    #             "prediction": topic_report
    #         }
    #         f_out.write(json.dumps(out_entry, ensure_ascii=False) + "\n")
