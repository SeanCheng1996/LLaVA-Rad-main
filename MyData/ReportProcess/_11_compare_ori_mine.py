import json

from _6_read_result_file import extract_response_file


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
            id_report_json[cur_id]['topics']=[]
        id_report_json[cur_id]['topics'].append(report['topic'])

        if "topic_sents" not in id_report_json[cur_id]:
            id_report_json[cur_id]['topic_sents'] = []
        id_report_json[cur_id]['topic_sents'].append(report['conversations'][2]['value'])


    return id_report_json

if __name__ == '__main__':
        split = "valid"
        processed_id_report_json = extractReport(
                "/data/sc159/data/MIMIC_III/llava_rad_topic/chat_valid_MIMIC_CXR_all_gpt4extract_rulebased_v3.json")

        for id in list(processed_id_report_json.keys())[:10]:
            print("="*60)
            print(f"id:{id}")
            ori_report=processed_id_report_json[id]['ori'].replace('. ','.\n    ')
            print(f"ori:\n    {ori_report}")
            print(f"processed:")
            for topic, topic_sents in zip(processed_id_report_json[id]['topics'], processed_id_report_json[id]['topic_sents']):
                print(f"  {topic}")
                print(f"    -{topic_sents}")