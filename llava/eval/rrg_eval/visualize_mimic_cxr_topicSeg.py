"""
A model worker executes the model.
"""
import os
import json
import math

import torch
import fire
from tqdm import tqdm
from PIL import Image, ImageFile

# https://stackoverflow.com/questions/12984426/pil-ioerror-image-file-truncated-with-big-images
ImageFile.LOAD_TRUNCATED_IMAGES = True

from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import build_logger, disable_torch_init, data_loaders
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, \
    ORGAN_MASK_MAPPING, MASK_THRESH
import pickle
import numpy as np
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def create_batches(data, batch_size, group_by_length, tokenizer):
    if batch_size == 1 or not group_by_length:
        return [data[i: i + batch_size] for i in range(0, len(data), batch_size)]
    else:
        batches = []
        batch, batch_len = [], None
        for d in data:
            d_len = len(tokenizer(d["conversations"][0]['value']).input_ids)
            if batch_len is None or d_len == batch_len:
                batch_len = d_len
                batch.append(d)
                if len(batch) == batch_size:
                    batches.append(batch)
                    batch, batch_len = [], None
            else:
                assert len(batch)
                batches.append(batch)
                batch, batch_len = [d], d_len
        if len(batch):
            batches.append(batch)
        assert len(data) == sum(len(b) for b in batches)
        return batches


def eval_model(
        query_file: str,
        image_folder: str,
        mask_path: str,
        conv_mode: str,
        prediction_file: str,
        model_path: str,
        model_base: str = None,
        load_8bit: bool = False,
        load_4bit: bool = False,
        device: str = "cuda",
        temperature: float = 0.2,
        top_p: float = None,
        num_beams: int = 1,
        chunk_idx: int = 0,
        num_chunks: int = 1,
        batch_size: int = 8,
        loader: str = "default",
        group_by_length: bool = False,
):
    os.makedirs("logs", exist_ok=True)
    logger = build_logger("model_mimic_cxr", f"logs/model_mimic_cxr_{chunk_idx}.log")

    # load model
    disable_torch_init()
    if model_path.startswith("/"):
        model_path = model_path
    else:
        model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    if not model_name.startswith("llavarad") and not model_name.startswith("llava"):
        # "llava" needs to be in model_name to correctly load the model.
        raise ValueError(f"Model name {model_name} is not 'llavarad' or 'llava'.")
    logger.info(f"Loading the model {model_name} ...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name, load_8bit, load_4bit, device=device)

    all_queries = data_loaders[loader](query_file)

    for query_index in range(len(all_queries)):

        query = all_queries[query_index]
        if query['topic'] == 'other':
            continue

        batch_prompts = []
        batch_input_ids = []
        batch_images = []

        q = query["conversations"][0]["value"]

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], q)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

        # get image
        image = Image.open(os.path.join(image_folder, query["image"])).convert("RGB")
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        # get mask
        mask_labels = ORGAN_MASK_MAPPING[query['topic']]
        if "all" in mask_labels:
            masked_img = image
        else:
            mask_json = os.path.join(mask_path, query["image"]).replace('.jpg', '.pkl').replace(".png", ".pkl")
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
        masked_img_tensor = image_processor.preprocess(masked_img, return_tensors='pt')['pixel_values'][0]

        # build prompts
        batch_prompts.append(prompt)
        batch_input_ids.append(input_ids)
        batch_images.append(image_tensor)
        batch_images.append(masked_img_tensor)

        with torch.inference_mode():
            outputs = model.generate(
                torch.stack(batch_input_ids).cuda(),
                images=torch.stack(batch_images).half().cuda(),
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                output_attentions=True,
                return_dict_in_generate=True,
                num_beams=num_beams,
                max_new_tokens=256,
                use_cache=True)

            # prepare for visualize
            generated_ids = outputs.sequences
            attentions = outputs.attentions

            # Get position of <image> token (id = -200 in your case)
            num_image_tokens = 1369
            image_token_id = -200
            input_ids = batch_input_ids[0]  # Before generation
            image_token_pos = (input_ids == image_token_id).nonzero(as_tuple=True)[0]

            # Loop over generated tokens
            for layer_index in [-2, 0, 10, 20, -1]:
                for i, token_id in enumerate(generated_ids[0][input_ids.shape[0]:]):  # Only new tokens
                    word = tokenizer.decode([token_id])
                    word_clean = word.strip().replace(" ", "_").replace("/", "_").replace("\\", "_")

                    # Get last layer attention for current token
                    step_attention = attentions[i]  # tuple of num_layers
                    if layer_index != -2:
                        cur_layer = step_attention[layer_index].detach().cpu().numpy()  # [B, num_heads, T, T]
                    else:
                        cur_layer = torch.stack(
                            step_attention).detach().cpu().numpy()  # [num_layers, B, num_heads, T, T]
                        cur_layer = cur_layer.max(axis=0)  # NumPy max over layers → [B, num_heads, T, T]
                    attn = cur_layer[0, :, -1,
                           :]  # [num_heads, total_seq_len], attention from current word to all tokens
                    attn_mean = attn.max(axis=0) # [total_seq_len]

                    # Get attention to image patches only
                    attn_image1 = attn_mean[
                                  image_token_pos[0].item(): image_token_pos[
                                                                 0].item() + num_image_tokens]
                    attn_image2 = attn_mean[
                                  image_token_pos[1].item() + num_image_tokens - 1: image_token_pos[
                                                                                        1].item() + 2 * num_image_tokens - 1]

                    # Plot
                    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
                    axs[0, 0].imshow(image)
                    axs[0, 0].set_title("Original Image 1")
                    axs[0, 0].axis('off')

                    axs[0, 1].imshow(image)
                    attn_image1 = attn_image1.reshape(37, 37)
                    attn_image1 = cv2.resize(attn_image1.astype(np.float32), (image.size[0], image.size[1]))
                    axs[0, 1].imshow(attn_image1, cmap="jet", alpha=0.5)
                    axs[0, 1].set_title("Attn on Image 1")
                    axs[0, 1].axis('off')

                    axs[1, 0].imshow(masked_img)
                    axs[1, 0].set_title("Original Image 2")
                    axs[1, 0].axis('off')

                    axs[1, 1].imshow(masked_img)
                    attn_image2 = attn_image2.reshape(37, 37)
                    attn_image2 = cv2.resize(attn_image2.astype(np.float32), (masked_img.size[0], masked_img.size[1]))
                    axs[1, 1].imshow(attn_image2, cmap="jet", alpha=0.5)
                    axs[1, 1].set_title("Attn on Image 2")
                    axs[1, 1].axis('off')

                    plt.tight_layout()
                    save_path = os.path.join(f"/data/sc159/LLaVARad/results/topic_seg/temp/{query_index}/{layer_index}",
                                             f"{i:03d}_{word_clean}.png")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    plt.savefig(save_path)
                    plt.show()
                    plt.close(fig)


if __name__ == "__main__":
    # fire.Fire(eval_model)
    eval_model(
        query_file="/data/sc159/data/MIMIC_III/llava_rad_topic/chat_test_MIMIC_CXR_all_gpt4extract_rulebased_v4.json",
        image_folder="/data/sc159/data/MIMIC_III/physionet.org/files/mimic-cxr-jpg/2.0.0/files",
        mask_path="/data/sc159/data/MIMIC_III/segmentation_single",
        conv_mode="v1",
        prediction_file="/data/sc159/LLaVARad/results/topic_seg/temp/test_0.jsonl",
        model_path="/data/sc159/LLaVARad/checkpoints/llava_biomedclip_cxr_518-pt-1e-1e-3-20250628005426",
        model_base="lmsys/vicuna-7b-v1.5",
        load_8bit=False,
        load_4bit=False,
        device="cuda",
        temperature=0,
        top_p=None,
        num_beams=1,
        chunk_idx=0,
        num_chunks=1,
        batch_size=2,
        loader="mimic_topic_reason_findings",
        group_by_length=True,
    )
