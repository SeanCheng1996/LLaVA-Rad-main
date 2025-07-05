import json

file_paths = [
    "/data/sc159/data/MIMIC_III/llava_rad_topic/chat_train_MIMIC_CXR_all_gpt4extract_rulebased_v1.json",
    "/data/sc159/LLaVARad/MyData/ReportProcess/chexpert_organ_labels/re_for_bad_response/chat_train_MIMIC_CXR_all_gpt4extract_rulebased_v1.json"
]

save_path="/data/sc159/data/MIMIC_III/llava_rad_topic/chat_train_MIMIC_CXR_all_gpt4extract_rulebased_v2.json"

res = []
for file_path in file_paths:
    with open(file_path, 'r') as f:
        cur_res = json.load(f)
        print(len(cur_res))
        res.extend(cur_res)
print(len(res))
print(res[0])

with open(save_path, 'w') as f:
    json.dump(res, f, indent=2)

