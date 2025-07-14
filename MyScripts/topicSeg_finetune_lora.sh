#!/bin/bash

# Set the following variables correspondingly to run this script:

################## VICUNA ##################
PROMPT_VERSION=v1

model_base=lmsys/vicuna-7b-v1.5
output_dir="${1:-./checkpoints_topic_seg_a100}"

# PROJECTOR="/PATH_TO/mm_projector.bin" # generated using pretrain.sh
PROJECTOR="/data/sc159/LLaVARad/checkpoints/topic_seg/llava_biomedclip_cxr_518-pt-1e-1e-3-20250710031958/mm_projector.bin" # todo, sirius
#PROJECTOR="/work/devika/sc159/llava_rad/LLaVA-Rad-main/checkpoints_topic_seg_a100/llava_biomedclip_cxr_518-pt-1e-1e-3-20250710031958/mm_projector.bin" # todo, rapid
vision_tower="biomedclip_cxr_518"
vision_tower_config="llava/model/multimodal_encoder/open_clip_encoder/model_configs/biomedclip_cxr_518.json"
vision_tower_checkpoint="biomedclipcxr_518_checkpoint.pt"
################## VICUNA ##################


################## Data ##################
# data_path=/PATH_TO/physionet.org/files/llava-rad-mimic-cxr-annotation/1.0.0/chat_train_MIMIC_CXR_all_gpt4extract_rulebased_v1.json
data_path=/data/sc159/data/MIMIC_III/llava_rad_topic/chat_train_MIMIC_CXR_all_gpt4extract_rulebased_v3.json # todo, sirius
#data_path=/work/devika/data/MIMIC_III/MIMIC_III/llava_rad_topic/chat_train_MIMIC_CXR_all_gpt4extract_rulebased_v3.json #todo, rapid

mask_path=/data/sc159/data/MIMIC_III/segmentation_single # todo, sirius
#mask_path=/work/devika/data/MIMIC_III/MIMIC_III/segmentation_single # todo, rapid

loader="mimic_topic_reason_findings"

# image_folder=/PATH_TO/physionet.org/files/mimic-cxr-jpg/2.0.0/files
image_folder=/data/sc159/data/MIMIC_III/physionet.org/files/mimic-cxr-jpg/2.0.0/files # todo, sirius
#image_folder=/work/devika/data/MIMIC_III/MIMIC_III/physionet.org/files/mimic-cxr-jpg/2.0.0/files # todo, rapid
################## Data ##################

################## Run name ##################
epoch="${2:-3}"
bsz="${3:-8}"
lr="1e-4"
schedule="lora-${epoch}e"
export run_name="${vision_tower}-${schedule}-${lr}-$(date +%Y%m%d%H%M%S)"
echo $run_name > run_name
################## Run name ##################


# Batch size is set for 4-GPU machines, 64 in total.
WANDB_PROJECT="llava_topic_seg" WANDB_RUN_ID="llava-ft-$(date +%Y%m%d%H%M%S)" WANDB_RUN_GROUP=fine-tune \
    deepspeed llava/train/train_mem_forTopicSeg.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_alpha 128 \
    --model_name_or_path ${model_base} \
    --version $PROMPT_VERSION \
    --data_path ${data_path} \
    --loader ${loader} \
    --image_folder ${image_folder} \
    --vision_tower ${vision_tower} \
    --vision_tower_config ${vision_tower_config} \
    --vision_tower_checkpoint ${vision_tower_checkpoint} \
    --pretrain_mm_mlp_adapter ${PROJECTOR} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${output_dir}/${run_name} \
    --num_train_epochs ${epoch} \
    --per_device_train_batch_size ${bsz} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 800 \
    --save_total_limit 2 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --run_name ${run_name}
