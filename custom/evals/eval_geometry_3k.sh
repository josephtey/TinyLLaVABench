#!/bin/bash

# Check if model name argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <model_name>"
  exit 1
fi

MODEL_NAME=$1

# data
DATA_FILE="./../../../multimodal-reasoning/dataset/eval/geometry_3k/geometry_3k.json"
IMAGE_FOLDER="./../../../multimodal-reasoning/dataset/eval/geometry_3k/images"

# pretrained model
MODEL_BASE="./../checkpoints/pre-trained/TinyLLaVA-3.1B"

# finetuned model
MODEL_PATH="./../checkpoints/fine-tuned/geometry_3k/${MODEL_NAME}"

# results file
RESULTS_FILE="./../../../multimodal-reasoning/results/files/results_geometry_3k_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).json"

python -m eval_geometry_3k \
    --image-folder $IMAGE_FOLDER \
    --data-file $DATA_FILE \
    --model-base $MODEL_BASE \
    --model-path $MODEL_PATH \
    --results-file $RESULTS_FILE \
    --temperature 0 \
    --conv-mode phi
