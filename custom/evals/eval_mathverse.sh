#!/bin/bash

# data
DATA_FILE="./../../multimodal-reasoning/dataset/eval/mathverse/mathverse_visual_only.json"
IMAGE_FOLDER="./../../multimodal-reasoning/dataset/eval/mathverse/images"

# pretrained model
MODEL_BASE="./../checkpoints/pre-trained/TinyLLaVA-3.1B"

# finetuned model
MODEL_PATH="./../checkpoints/finetuned/TinyLLaVA-3.1B-lora"

# results file
RESULTS_FILE="./../../multimodal-reasoning/results/files/results_$(date +%Y%m%d_%H%M%S).json"

python -m eval \
    --image-folder $IMAGE_FOLDER$ \
    --data-file $DATA_FILE$ \
    --model-base $MODEL_BASE \
    --model-path $MODEL_PATH \
    --results-file $RESULTS_FILE \
    --temperature 0 \
    --conv-mode phi