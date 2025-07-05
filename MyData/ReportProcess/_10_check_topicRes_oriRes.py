"""
Compare my generated report and the original report.
Some id is deleted.
Print out all the deleted id and see if it's reasonable to delete them.
"""

import json
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


def get_my_id_report_json(file_path):
    """
    read my report file, a json list.
    move the "id"|"image" outside as key.
    check if the id is duplicated, should not exist duplicated id after filtering.

    return:
    ori_res, a nested json. {id|imagePath: {original json}, ...}
    """
    with open(file_path, 'r') as f:
        original_data = json.load(f)

    ori_res = {}
    for item in original_data:
        item_id = item["id"]
        item_imagePath = item["image"]
        item_newId = f"{item_id}|{item_imagePath}"
        ori_res[item_newId] = item

    return ori_res


if __name__ == '__main__':

    my_res_path = "/data/sc159/data/MIMIC_III/llava_rad_topic/chat_train_MIMIC_CXR_all_gpt4extract_rulebased_v1.json"
    ori_res_path = "/data/sc159/data/MIMIC_III/llava_rad/chat_train_MIMIC_CXR_all_gpt4extract_rulebased_v1.json"

    ori_res = get_ori_id_report_json(ori_res_path)
    my_res = get_my_id_report_json(my_res_path)

    with open("/data/sc159/data/MIMIC_III/llava_rad_topic/train_sanity_check.txt", 'w') as f:
        for newID, report in tqdm(ori_res.items()):
            if newID not in my_res:
                print("-" * 20, file=f)
                print(newID, file=f)
                print(report['conversations'][1]['value'], file=f)
