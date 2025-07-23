import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from llava.constants import ORGAN_MASK_MAPPING, MASK_THRESH
import os
import pickle
from scipy import ndimage
import math
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import textwrap
from matplotlib.patches import Rectangle


def load_gt_preds_json(file_path):
    gt_preds_json_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
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
    id, gt, pred = cur_json['id'], cur_json['reference'], cur_json['prediction']
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
    plot_images(image, organ_maskedImg_json, gt, pred)


def plot_images(image, organ_maskedImg_json, gt, pred):
    gt=gt.replace(". ", ".\n")
    gt="gt\n"+gt
    pred = pred.replace(". ", ".\n")
    pred="pred\n"+pred
    # 计算总行数：器官图片每4个一组，加上原始图片行和文本行
    num_organs = len(organ_maskedImg_json)
    num_rows = 1 + (num_organs + 3) // 4  # 原始图片行 + 器官行 + 文本行

    # 创建figure，设置合适的大小
    fig = plt.figure(figsize=(25, 9 * num_rows))
    gs = fig.add_gridspec(num_rows, 4)

    # 第一行第一列：原始图像
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(image)
    ax.set_title('Original Image', fontsize=30)
    ax.axis('off')

    # 显示器官分割图像
    row = 0
    col = 1
    for i, (organ, img) in enumerate(organ_maskedImg_json.items()):
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img)
        ax.set_title(organ, fontsize=30)
        ax.axis('off')
        col += 1
        if col > 3:
            row += 1
            col = 0

    # 最后一行：合并单元格显示文本
    # GT文本区域（合并第1、2列）
    ax_gt = fig.add_subplot(gs[-1, :2])
    ax_gt.set_facecolor('lightblue')  # 设置背景颜色
    ax_gt.axis('off')

    # 自动换行文本，水平垂直居中
    ax_gt.text(0.5, 0.5, gt, ha='center', va='center',
               wrap=True, fontsize=30, bbox=dict(facecolor='lightblue', alpha=0.5))

    # Pred文本区域（合并第3、4列）
    ax_pred = fig.add_subplot(gs[-1, 2:])
    ax_pred.set_facecolor('lightgreen')  # 设置背景颜色
    ax_pred.axis('off')

    # 自动换行文本，水平垂直居中
    ax_pred.text(0.5, 0.5, pred, ha='center', va='center',
                 wrap=True, fontsize=30, bbox=dict(facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    gt_preds_json_file = "/data/sc159/LLaVARad/results/topic_seg/llavarad_MIMIC_onlyMLP/test_merged.jsonl"
    query_json_file = "/data/sc159/data/MIMIC_III/llava_rad_topic/chat_test_MIMIC_CXR_all_gpt4extract_rulebased_v3.json"
    gt_preds_json_list = load_gt_preds_json(gt_preds_json_file)
    id_image_json = load_query_json(query_json_file)
    np.random.seed(42)
    for json_ix in np.random.choice(len(gt_preds_json_list), 10):
        show_eg(gt_preds_json_list, id_image_json, json_ix=json_ix)
