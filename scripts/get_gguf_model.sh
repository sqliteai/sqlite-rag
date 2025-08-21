#!/bin/bash

# Usage: ./get_gguf_model.sh <HF_MODEL_ID> <HF_GGUF_FILEname>
# or via Environment variables: HF_MODEL_ID, HF_HF_GGUF_FILE

# Example: ./get_gguf_model.sh Qwen/Qwen3-Embedding-0.6B-GGUF Qwen3-Embedding-0.6B-Q8_0.gguf

set -e

LOCAL_DIR="./models"

# Priority: CLI arguments > environment variables
if [ $# -ge 2 ]; then
    HF_MODEL_ID="$1"
    HF_GGUF_FILE="$2"
    [ $# -ge 3 ] && LOCAL_DIR="$3"
elif [ -z "$HF_MODEL_ID" ] || [ -z "$HF_GGUF_FILE" ]; then
    echo "Error: Missing required parameters"
    echo "Provide either:"
    echo "  CLI: $0 <HF_MODEL_ID> <HF_GGUF_FILE>"
    echo "  ENV: HF_MODEL_ID and HF_GGUF_FILE environment variables"
    exit 1
fi

# Check if huggingface-hub is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli not found. Install with: pip install huggingface_hub"
    exit 1
fi

mkdir -p $LOCAL_DIR

# Download specific GGUF file
echo "Downloading $HF_GGUF_FILE from $HF_MODEL_ID..."
hf download "$HF_MODEL_ID" "$HF_GGUF_FILE" --local-dir "$LOCAL_DIR/$HF_MODEL_ID"

echo "Downloaded to: $LOCAL_DIR/$HF_MODEL_ID/$HF_GGUF_FILE"
