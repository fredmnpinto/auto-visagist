#!/usr/bin/env bash
# Download the dlib 68-point shape predictor model.
#
# Usage:
#   bash scripts/download_model.sh
#
# The model file is downloaded and placed at:
#   data/shape_predictor_68_face_landmarks.dat
#
# This file is required for facial landmark detection.
# Source: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"
MODEL_FILE="$DATA_DIR/shape_predictor_68_face_landmarks.dat"
MODEL_BZ2="$MODEL_FILE.bz2"
MODEL_URL="http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

echo "=== dlib Shape Predictor Download ==="
echo "Target: $MODEL_FILE"

# Check if model already exists
if [ -f "$MODEL_FILE" ]; then
    echo "Model file already exists at: $MODEL_FILE"
    echo "To re-download, delete it first: rm $MODEL_FILE"
    exit 0
fi

# Create data directory if needed
mkdir -p "$DATA_DIR"

# Download
echo "Downloading from: $MODEL_URL"
echo "This may take a while (~100MB)..."
if command -v wget &>/dev/null; then
    wget -O "$MODEL_BZ2" "$MODEL_URL"
elif command -v curl &>/dev/null; then
    curl -o "$MODEL_BZ2" "$MODEL_URL"
else
    echo "Error: Neither wget nor curl found. Please install one of them."
    exit 1
fi

# Decompress
echo "Decompressing..."
bzip2 -d "$MODEL_BZ2"

echo "Done! Model saved to: $MODEL_FILE"
