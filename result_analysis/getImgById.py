from PIL import Image
import pickle
import os
import numpy as np
from scipy import ndimage
import cv2
from llava.constants import ORGAN_MASK_MAPPING, MASK_THRESH
import matplotlib.pyplot as plt
import json

def load_query_json(file_path):
    with open(file_path, 'r') as f:
        query_json = json.load(f)
    id_image_json = {}
    for cur_json in query_json:
        id_image_json[cur_json['id']] = {"image": cur_json['image'][len('mimic/'):], "topic": cur_json['topic']}
    return id_image_json

def show_eg(id_image_json, id,
            img_folder="/data/sc159/data/MIMIC_III/physionet.org/files/mimic-cxr-jpg/2.0.0/files",
            mask_folder="/data/sc159/data/MIMIC_III/segmentation_single"):
    """
    get image
    """
    img_path = f"{img_folder}/{id_image_json[id]['image']}"
    image = Image.open(img_path).convert('RGB')
    """
    get masked image
    """
    organ_maskedImg_json = {}

    mask_json = os.path.join(mask_folder, id_image_json[id]['image']).replace('.jpg', '.pkl')
    with open(mask_json, 'rb') as f:
        mask_json = pickle.load(f)  # {"$organ": np.array, bool,512*512, ...}
    # get mask for specific organs
    for cur_organ in list(mask_json.keys()):
        merged_mask = np.zeros_like(next(iter(mask_json.values())), dtype=bool)
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

        # put under the to-show json
        organ_maskedImg_json[cur_organ] = masked_img
    """
    show
    """
    plt.imshow(image)
    plt.show()
    for organ, masked_img in organ_maskedImg_json.items():
        plt.imshow(masked_img)
        plt.title(organ)
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    query_json_file = "/data/sc159/data/MIMIC_III/llava_rad_topic/chat_valid_MIMIC_CXR_all_gpt4extract_rulebased_v4.json"
    id_image_json = load_query_json(query_json_file)

    id="10003502_50084553"
    show_eg(id_image_json, id)
