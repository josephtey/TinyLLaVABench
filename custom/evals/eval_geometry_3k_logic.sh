#!/bin/bash

# data
DATA_FILE="./../../../multimodal-reasoning/dataset/eval/geometry_3k/geometry_3k.json"
IMAGE_FOLDER="./../../../multimodal-reasoning/dataset/eval/geometry_3k/images"

# pretrained model
MODEL_BASE="./../checkpoints/pre-trained/TinyLLaVA-3.1B"

# finetuned model
MODEL_PATH="./../checkpoints/fine-tuned/geometry_3k/TinyLLaVA-3.1B-lora-2"

# results file
RESULTS_FOLDER="./../../../multimodal-reasoning/results/files/results_geometry_3k_finetuned_logic_$(date +%Y%m%d_%H%M%S)"

python -m eval_geometry_3k \
    --image-folder $IMAGE_FOLDER \
    --data-file $DATA_FILE \
    --model-base $MODEL_BASE \
    --model-path $MODEL_PATH \
    --results-folder $RESULTS_FOLDER \
    --temperature 0 \
    --conv-mode phi