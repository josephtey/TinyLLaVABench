#!/bin/bash

echo "Starting peano..."
# Run baseline direct evaluation
bash eval_geometry_3k_peano.sh

echo "Starting logic..."
# Run baseline CoT evaluation
bash eval_geometry_3k_logic.sh
