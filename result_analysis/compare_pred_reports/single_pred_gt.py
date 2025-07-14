import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from llava.constants import ORGAN_MASK_MAPPING, MASK_THRESH
import os
import pickle
from scipy import ndimage


def load_gt_preds_json(file_path):
    gt_preds_json_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "organs or diseases: other" in line:
                continue
            gt_preds_json_list.append(json.loads(line))
    return gt_preds_json_list


def load_query_json(file_path):
    with open(file_path, 'r') as f:
        query_json = json.load(f)
    id_image_json = {}
    for cur_json in query_json:
        id_image_json[cur_json['id']] = {"image": cur_json['image'][len('mimic/'):], "topic": cur_json['topic']}
    return id_image_json


def show_eg(gt_preds_json_list, id_image_json, json_ix,
            img_folder="/data/sc159/data/MIMIC_III/physionet.org/files/mimic-cxr-jpg/2.0.0/files",
            mask_folder="/data/sc159/data/MIMIC_III/segmentation_single"):
    cur_json = gt_preds_json_list[json_ix]
    id, query, gt, pred = cur_json['id'], cur_json['query'], cur_json['reference'], cur_json['prediction']
    topic = query.split(":")[-1][:-1].strip()
    """
    get image
    """
    img_path = f"{img_folder}/{id_image_json[id]['image']}"
    image = Image.open(img_path).convert('RGB')
    """
    get masked image
    """
    mask_labels = ORGAN_MASK_MAPPING[topic]
    if "all" in mask_labels:
        masked_img = image
    else:
        mask_json = os.path.join(mask_folder, id_image_json[id]['image']).replace('.jpg', '.pkl')
        with open(mask_json, 'rb') as f:
            mask_json = pickle.load(f)  # {"$organ": np.array, bool,512*512, ...}
        # get mask for specific organs
        merged_mask = np.zeros_like(next(iter(mask_json.values())), dtype=bool)
        for cur_organ in mask_labels:
            mask_arr = mask_json[cur_organ]
            # post process mask: island removal
            labeled, num_features = ndimage.label(mask_arr)
            sizes = ndimage.sum(mask_arr, labeled, range(num_features + 1))
            mask_arr = sizes[labeled] > MASK_THRESH
            merged_mask = np.logical_or(merged_mask, mask_arr)
        # get segmented image
        mask_resized = cv2.resize(merged_mask.astype(np.uint8), image.size, interpolation=cv2.INTER_NEAREST)
        img_array = np.asarray(image)
        masked_img_array = img_array * mask_resized[:, :, np.newaxis]  # Broadcast mask over RGB channels
        masked_img = Image.fromarray(masked_img_array)
    """
    show
    """
    fig, axs = plt.subplots(4, 2, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 2, 1, 1]})

    axs[0, 0].text(0.5, 0.5, query, ha='center', va='center', fontsize=14, wrap=True)
    axs[0, 0].axis('off')

    axs[0, 1].text(0.5, 0.5, topic, ha='center', va='center', fontsize=14, wrap=True)
    axs[0, 1].axis('off')

    axs[1, 0].imshow(image)
    axs[1, 0].axis('off')

    axs[1, 1].imshow(masked_img)
    axs[1, 1].axis('off')

    axs[2, 0].text(0.5, 0.5, f"gt: {gt}", ha='center', va='center', fontsize=14, wrap=True)
    axs[2, 0].axis('off')
    axs[2, 1].axis('off')


    axs[3, 0].text(0.5, 0.5, f"pred: {pred}", ha='center', va='center', fontsize=14, wrap=True)
    axs[3, 0].axis('off')
    axs[3, 1].axis('off')

    plt.subplots_adjust(hspace=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    gt_preds_json_file="/data/sc159/LLaVARad/results/topic_seg/llavarad_MIMIC_onlyMLP/test_0.jsonl"
    query_json_file="/data/sc159/data/MIMIC_III/llava_rad_topic/chat_test_MIMIC_CXR_all_gpt4extract_rulebased_v3.json"
    gt_preds_json_list=load_gt_preds_json(gt_preds_json_file)
    id_image_json=load_query_json(query_json_file)
    np.random.seed(42)
    for json_ix in np.random.choice(len(gt_preds_json_list),20):
        show_eg(gt_preds_json_list, id_image_json,json_ix=json_ix)

