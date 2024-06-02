#!/bin/bash

# data
DATA_FILE="./../../../multimodal-reasoning/dataset/eval/geometry_3k/geometry_3k.json"
IMAGE_FOLDER="./../../../multimodal-reasoning/dataset/eval/geometry_3k/images"
IDX=10

# finetuned model
MODEL_PATH="./../checkpoints/pre-trained/TinyLLaVA-3.1B"

# results file
RESULTS_FILE="./../../../multimodal-reasoning/results/files/results_geometry_3k_baseline_cot_$(date +%Y%m%d_%H%M%S).json"
ATTENTION_FILE="./../../../multimodal-reasoning/results/attention_results/results_geometry_3k_baseline_cot_$(date +%Y%m%d_%H%M%S).json"
ATTENTION_WEIGHTS_FILE="./../../../multimodal-reasoning/results/attention_results/results_geometry_3k_baseline_cot_$(date +%Y%m%d_%H%M%S).pt"

python -m eval_geometry_3k \
    --image-folder $IMAGE_FOLDER \
    --idx $IDX \
    --data-file $DATA_FILE \
    --model-path $MODEL_PATH \
    --results-file $RESULTS_FILE \
    --attention-file $ATTENTION_FILE \
    --attention-weights-file $ATTENTION_WEIGHTS_FILE \
    --temperature 0 \
    --conv-mode phi \
    --single \
    --baseline \
    --baseline-type "cot"
