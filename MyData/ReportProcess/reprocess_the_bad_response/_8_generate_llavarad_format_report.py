"""
With the topic_segmented report,
1. Match with original_llavaRad "id" with my "custom_id",
filtering the original_llavaRad with only "generate_method"="gpt4" and "view" in ["AP","PA"]

2. for each organ in the topic_segmented report, do the following and generate a new json for each organ.

    3. copy the json of the original_llavaRad

    4. replace "generate_method": "gpt4" to "generate_method": "topic_segmented"

    5. add another key in the json, e.g.:
    "topic":"organ_heart"

    5. add another conversation key-value pair in the report section.
    conversations[2]={"from":"topic_based", "value":"corresponding reports"}
"""
import os
import json
from glob import glob
from tqdm import tqdm


def get_ori_id_report_json(file_path):
    """
    read the original report file, a json list.
    apply filter: filtering the original_llavaRad with only "generate_method"="gpt4" and "view" in ["AP","PA"]
    move the "id"|"image" outside as key.
    check if the id is duplicated, should not exist duplicated id after filtering.

    return:
    ori_res, a nested json. {id|imagePath: {original json}, ...}
    """
    with open(file_path, 'r') as f:
        original_data = json.load(f)

    ori_res = {}
    for item in original_data:
        if item.get("generate_method") == "gpt4" and item.get("view") in ["AP", "PA"]:
            item_id = item["id"]
            item_imagePath = item["image"]
            item_newId = f"{item_id}|{item_imagePath}"
            if item_newId in ori_res:
                raise ValueError(f"Duplicate ID found in original data: {item_newId}")
            ori_res[item_newId] = item

    return ori_res


def get_topic_id_report_json(file_path):
    """
    read the topic_segmented report file, a jsonl file, each row is a json.

    return:
    topic_res, a 2 nested json. {id: {"$topic1": "$report1", "$topic2": "$report2", ...}, ... }
    """
    topic_res = {}
    print(f"Loading {file_path}")
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            custom_id = item["custom_id"]

            topic_sentence_json_list = item['response']['body']['choices'][0]['message']['content']

            index = topic_sentence_json_list.find("```json")
            if index != -1:
                topic_sentence_json_list = topic_sentence_json_list[index:]  # Slice from the found index

            topic_sentence_json_list = topic_sentence_json_list.strip().removeprefix("```json").removesuffix(
                "```").strip()
            topic_sentence_json_list = topic_sentence_json_list.strip().removeprefix("```").removesuffix(
                "```").strip()
            if len(topic_sentence_json_list) < 1:
                continue
            try:
                topic_sentence_json_list = json.loads(
                    topic_sentence_json_list)  # a json list with each json {"topic":"", "sentence or sentences":["","",...]}
            except:
                print(custom_id)

            for topic_sentence_json in topic_sentence_json_list:

                topic = topic_sentence_json["topic"]
                if "sentence" in item:
                    sentences = topic_sentence_json["sentence"]
                else:
                    sentences = topic_sentence_json["sentences"]
                sentences = " ".join(sentences)

                if custom_id not in topic_res:
                    topic_res[custom_id] = {}

                topic_res[custom_id][topic] = sentences

    return topic_res


def check_ori_topic_res(ori_res, topic_res):
    """
    They should have the same ids.
    """
    ori_ids = [k.split("|")[0] for k, v in ori_res.items()]
    ori_ids_newId_json = {k.split("|")[0]: k for k, v in ori_res.items()}

    ori2topic_bad_id_list = []
    for id in ori_ids:
        if id not in topic_res:
            ori2topic_bad_id_list.append(id)

    topic2ori_bad_id_list = []
    for id in topic_res:
        if id not in ori_ids:
            topic2ori_bad_id_list.append(id)

    if len(ori2topic_bad_id_list) > 0:
        print("-" * 15)
        print("The following id is not in the topic_res:")
        print(ori2topic_bad_id_list)
        print("***** We will delete those files from the new report json file. *****")
    elif len(topic2ori_bad_id_list) > 0:
        print("-" * 15)
        print("The following id is not in the ori_res:")
        print(topic2ori_bad_id_list)

        exit()
    else:
        print("ori and topic results are matching perfectly")


def generate_json_list_for_one_ori_item(original_item, topic_reports):
    """
    1. copy from the ori_res with the specific id.
    2. for each organ in topic_res
     2.1 update "generate_method"="topic_segmented"
     2.2 add a key "topic":"the current organ"
     2.3 add another conversation key-value pair in the report section.
         conversations[2]={"from":"topic_based", "value":"corresponding reports"}
     2.5 append to cur_json_list

    params:
        original_item: json. {"id":, "image":,...}. One json item from original json list.
        topic_reports: json. {"$topic1": "$report1", "$topic2": "$report2", ...}
     return:
      cur_json_list. [ {}, ...]. All json is for the same id. Each json is for different topic.
    """

    cur_json_list = []

    for topic, report in topic_reports.items():
        # Create a deep copy of the original item
        new_item = json.loads(json.dumps(original_item))

        # Update fields
        new_item["generate_method"] = "topic_segmented"
        new_item["topic"] = topic

        # Add the topic-based report as a new conversation
        new_conversation = {
            "from": "topic_based",
            "value": report
        }

        # Make sure conversations exists and is a list
        if "conversations" not in new_item:
            raise ValueError(f"No conversation existed, id: {new_item['id']}")
        if len(new_item["conversations"]) != 2:
            raise ValueError(
                f"Conversation len is {len(new_item['conversations'])}, which should be 2, id: {new_item['id']}")
        new_item["conversations"].append(new_conversation)

        cur_json_list.append(new_item)

    return cur_json_list


if __name__ == '__main__':
    split = "train"
    ori_path_file = f"/data/sc159/data/MIMIC_III/llava_rad/chat_{split}_MIMIC_CXR_all_gpt4extract_rulebased_v1.json"
    topic_file_paths = [f"/data/sc159/LLaVARad/MyData/ReportProcess/chexpert_organ_labels/re_for_bad_response/{split}_prompts_bad_0_5000_response.jsonl"]
    save_file_path = f"/data/sc159/LLaVARad/MyData/ReportProcess/chexpert_organ_labels/re_for_bad_response/chat_{split}_MIMIC_CXR_all_gpt4extract_rulebased_v1.json"

    # load ori report
    ori_res = get_ori_id_report_json(ori_path_file)

    # load topic report
    print(f"Loading topic report from {len(topic_file_paths)} paths. \n E.g. {topic_file_paths[0]}")
    topic_res = {}
    for topic_file_path in topic_file_paths:
        temp_topic_res = get_topic_id_report_json(topic_file_path)
        for key in temp_topic_res:
            if key in topic_res:
                topic_res[key].update(temp_topic_res[key])
            else:
                topic_res[key] = temp_topic_res[key]

    # sanity check
    check_ori_topic_res(ori_res, topic_res)

    # get updated report
    res_json_list = []
    ignored_id_list = []
    print(f"start transfering. {'*' * 15}")
    for id_imagePath, ori_item in tqdm(ori_res.items(), total=len(ori_res)):

        cur_id = id_imagePath.split("|")[0]
        if cur_id not in topic_res:
            continue
        topic_reports = topic_res[cur_id]
        id_json_list = generate_json_list_for_one_ori_item(ori_item, topic_reports)
        res_json_list.extend(id_json_list)

    # save
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
    with open(save_file_path, 'w') as f:
        json.dump(res_json_list, f, indent=2)
