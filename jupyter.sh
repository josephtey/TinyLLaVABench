#!/bin/bash

# Export the XDG_RUNTIME_DIR environment variable
export XDG_RUNTIME_DIR=""

# Start Jupyter Lab without a browser, on port 8880, accessible from any IP address
jupyter lab --no-browser --port=8880 --ip='0.0.0.0'
