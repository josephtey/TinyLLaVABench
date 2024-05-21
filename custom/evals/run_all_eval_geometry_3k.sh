#!/bin/bash

echo "Starting baseline direct evaluation..."
# Run baseline direct evaluation
./eval_geometry_3k_baseline_direct.sh
echo "Finished baseline direct evaluation."

echo "Starting baseline CoT evaluation..."
# Run baseline CoT evaluation
./eval_geometry_3k_baseline_cot.sh
echo "Finished baseline CoT evaluation."

echo "Starting finetuned evaluation..."
# Run finetuned evaluation
./eval_geometry_3k_finetuned.sh
echo "Finished finetuned evaluation."