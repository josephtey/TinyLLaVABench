#!/bin/bash

# data
DATA_FILE="./../../../multimodal-reasoning/dataset/eval/geometry_3k/geometry_3k.json"
IMAGE_FOLDER="./../../../multimodal-reasoning/dataset/eval/geometry_3k/images"
IDX=0

# pretrained model
MODEL_BASE="./../checkpoints/pre-trained/TinyLLaVA-3.1B"

# finetuned model
MODEL_PATH="./../checkpoints/fine-tuned/geometry_3k/TinyLLaVA-3.1B-lora-1"

# results file
RESULTS_FILE="./../../../multimodal-reasoning/results/files/results_geometry_3k_finetuned_$(date +%Y%m%d_%H%M%S).json"
ATTENTION_FILE="./../../../multimodal-reasoning/results/attention_results/results_geometry_3k_finetuned_$(date +%Y%m%d_%H%M%S).json"
ATTENTION_WEIGHTS_FILE="./../../../multimodal-reasoning/results/attention_results/results_geometry_3k_finetuned_$(date +%Y%m%d_%H%M%S).pt"

python -m eval_geometry_3k \
    --image-folder $IMAGE_FOLDER \
    --idx $IDX \
    --data-file $DATA_FILE \
    --model-base $MODEL_BASE \
    --model-path $MODEL_PATH \
    --results-file $RESULTS_FILE \
    --attention-file $ATTENTION_FILE \
    --attention-weights-file $ATTENTION_WEIGHTS_FILE \
    --temperature 0 \
    --conv-mode phi \
    --single
