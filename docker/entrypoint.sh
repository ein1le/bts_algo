#!/bin/bash
set -e

# Initialize logging
echo "Starting BTS Algorithmic Trading System at $(date)"

# Check environment variables
if [ -z "$MODEL_PATH" ]; then
    echo "Warning: MODEL_PATH not set, using default"
    export MODEL_PATH="./models/saved_model"
fi

if [ -z "$LOG_LEVEL" ]; then
    export LOG_LEVEL="INFO"
fi

# Create necessary directories
mkdir -p logs reports data/processed

# Check if TensorFlow model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: TensorFlow model not found at $MODEL_PATH"
    echo "Please ensure the model is properly trained and exported"
    exit 1
fi

# Check TensorFlow C API
if ! ldconfig -p | grep -q tensorflow; then
    echo "Error: TensorFlow C API not found"
    exit 1
fi

echo "Environment check passed. Starting application..."

# Execute the command passed to the container
exec "$@" 