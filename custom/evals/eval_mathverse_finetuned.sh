#!/bin/bash

# data
DATA_FILE="./../../../multimodal-reasoning/dataset/eval/mathverse/mathverse.json"
IMAGE_FOLDER="./../../../multimodal-reasoning/dataset/eval/mathverse/images"

# pretrained model
MODEL_BASE="./../checkpoints/pre-trained/TinyLLaVA-3.1B"

# finetuned model
MODEL_PATH="./../checkpoints/fine-tuned/mathvista/TinyLLaVA-3.1B-lora-1"

# results file
RESULTS_FILE="./../../../multimodal-reasoning/results/files/results_$(date +%Y%m%d_%H%M%S).json"

python -m eval_mathverse \
    --image-folder $IMAGE_FOLDER \
    --data-file $DATA_FILE \
    --model-base $MODEL_BASE \
    --model-path $MODEL_PATH 
    --results-file $RESULTS_FILE \
    --temperature 0 \
    --conv-mode phi