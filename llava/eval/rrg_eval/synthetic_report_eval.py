import json
import os.path

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from open_clip import create_model_from_pretrained, get_tokenizer

# Load BiomedCLIP model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
model.to(device)
model.eval()

context_length = 256

# Load JSONL file
jsonl_path = '/data/sc159/data/MIMIC_III/llava_rad_topic/evaluate/llavarad_vs_topic.jsonl'  # <-- change to your real file
data = []
with open(jsonl_path, 'r') as f:
    for line in f:
        data.append(json.loads(line.strip()))

# Extract features
image_embeddings = []
prediction_embeddings = []

# for entry in tqdm(data[:10000]):
for entry_start_i in tqdm(range(0, len(data), 500)):
    image = torch.stack([
        preprocess(
            Image.open(os.path.join('/data/sc159/data/MIMIC_III/physionet.org/files/mimic-cxr-jpg/2.0.0/files/',
                                    data[entry_i]['image'][len('mimic/'):])).convert('RGB'))
        for entry_i in range(entry_start_i, entry_start_i+500)
    ]).to(device)
    pred_tokens = tokenizer(
        [data[entry_i]['prediction'] for entry_i in range(entry_start_i, entry_start_i+500)],
        context_length=context_length).to(device)

    with torch.no_grad():
        image_feat, pred_feat, logit_scale = model(image, pred_tokens)
        image_feat=image_feat.detach().cpu()
        pred_feat=pred_feat.detach().cpu()
    # Normalize
    image_feat /= np.linalg.norm(image_feat, axis=1, keepdims=True)
    pred_feat /= np.linalg.norm(pred_feat, axis=1, keepdims=True)

    image_embeddings.append(image_feat)
    prediction_embeddings.append(pred_feat)

# Compute cosine similarities
image_embeddings = np.vstack(image_embeddings)
prediction_embeddings = np.vstack(prediction_embeddings)

true_similarities = np.sum(image_embeddings * prediction_embeddings, axis=1)

# Permutation test with 1000 random shuffles
n_perm = 1000
random_similarities_list = []
for _ in tqdm(range(n_perm)):
    shuffled_indices = np.random.permutation(len(prediction_embeddings))
    sim_scores = np.sum(image_embeddings * prediction_embeddings[shuffled_indices], axis=1)
    random_similarities_list.append(sim_scores)
random_similarities = np.mean(random_similarities_list, axis=0)

# P-value calculation
true_mean = np.mean(true_similarities)
random_means = [np.mean(rs) for rs in random_similarities_list]
p_value = np.mean(np.array(random_means) >= true_mean)

# Print table results
print("Evaluation Results:")
print(f"{'Metric':<35}{'Value'}")
print("-" * 50)
print(f"{'Mean similarity (true match)':<35}{np.mean(true_similarities):.4f}")
print(f"{'Std deviation (true match)':<35}{np.std(true_similarities):.4f}")
print(f"{'Mean similarity (random match)':<35}{np.mean(random_similarities):.4f}")
print(f"{'Std deviation (random match)':<35}{np.std(random_similarities):.4f}")
print(f"{'Permutation test p-value':<35}{p_value:.4f}")