#!/bin/bash

# data
DATA_FILE="./../../../multimodal-reasoning/dataset/eval/geometry_3k/geometry_3k.json"
IMAGE_FOLDER="./../../../multimodal-reasoning/dataset/eval/geometry_3k/images"

# original model
MODEL_PATH="gpt-o"

# results file
RESULTS_FILE="./../../../multimodal-reasoning/results/files/results_geometry_3k_baseline_cot_gpto_$(date +%Y%m%d_%H%M%S).json"

python -m eval_geometry_3k \
    --image-folder $IMAGE_FOLDER \
    --data-file $DATA_FILE \
    --model-path $MODEL_PATH \
    --results-file $RESULTS_FILE \
    --temperature 0 \
    --conv-mode phi \
    --baseline \
    --baseline-type "direct"
