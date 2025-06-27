import requests
import torch
from PIL import Image
from io import BytesIO

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
import matplotlib.pyplot as plt

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


# Model
disable_torch_init()

model_path = "microsoft/llava-rad"
model_base = "lmsys/vicuna-7b-v1.5"
model_name = "llavarad"
conv_mode = "v1"

tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)

# Prepare query
image_file = "https://openi.nlm.nih.gov/imgs/512/253/253/CXR253_IM-1045-1001.png"  # CXR w pneumothorax from Open-I
query = "<image>\nDescribe the findings of the chest x-ray.\n"

conv = conv_templates[conv_mode].copy()
conv.append_message(conv.roles[0], query)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

image = load_image(image_file)
image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].half().unsqueeze(0).cuda()

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

stopping_criteria = KeywordsStoppingCriteria(["</s>"], tokenizer, input_ids)

with torch.no_grad():
    outputs = model(
        input_ids,
        images=image_tensor,
        output_attentions=True)

all_attentions = torch.stack(outputs.attentions)  # (num_layers, batch_size, num_heads, seq_len, seq_len)
all_attentions = all_attentions.squeeze(1)  # [layers, heads, seq_len, seq_len]

# 4. 找到所有image_token的index
image_token_id = tokenizer.image_token_id
image_token_indices = torch.where(input_ids.squeeze() == image_token_id)[0].tolist()
end_image_token_index = max(image_token_indices) + 1
start_image_token_index = min(image_token_indices)

# 5. 得到text token对patch的attention
token_patch_attention = all_attentions[:, :, end_image_token_index:,
                        start_image_token_index:end_image_token_index]  # layers*heads*text_tokens*img_tokens
# 6. fuse heads and layers
mean_token_patch_attention = token_patch_attention.mean(dim=0).mean(dim=0)
max_token_patch_attention = token_patch_attention.max(dim=0)[0].max(dim=0)[0]

# 7. get original image
plt.imshow(image)
plt.show()