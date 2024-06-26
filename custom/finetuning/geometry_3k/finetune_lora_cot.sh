#!/bin/bash

# Assign the arguments to variables
DATA_PATH="/piech/u/joetey/multimodal-reasoning/dataset/training/geometry_3k/geometry_3k_finetuning.json"
IMAGE_PATH="/piech/u/joetey/multimodal-reasoning/dataset/training/geometry_3k/images/"
OUTPUT_DIR="/piech/u/joetey/TinyLLaVABench/custom/checkpoints/fine-tuned/geometry_3k/TinyLLaVA-3.1B-lora-1"

deepspeed /piech/u/joetey/TinyLLaVABench/tinyllava/train/train.py \
    --deepspeed /piech/u/joetey/TinyLLaVABench/scripts/tiny_llava/zero3.json \
    --lora_enable True --lora_r 32 --lora_alpha 64 \
    --model_name_or_path /piech/u/joetey/TinyLLaVABench/custom/checkpoints/pre-trained/TinyLLaVA-3.1B \
    --version phi \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_PATH\
    --vision_tower /piech/u/joetey/TinyLLaVABench/custom/checkpoints/pre-trained/TinyLLaVA-3.1B-SigLIP \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --fp16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 15 \
    --lazy_preprocess True \
    --report_to wandb \
