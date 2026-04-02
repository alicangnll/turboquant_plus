#!/bin/bash

# To prevent errors when running on Mac
set -e

# Starting Directory (Assuming you run this in the project root)
ROOT_DIR=$(pwd)

echo "=============== TURBOQUANT DEMO ==============="

# Part 1: Python Prototype
echo ">>> [1/4] Setting up Python environment and installing dependencies..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e "."

echo ">>> Starting theoretical compression demo via Python..."
python3 benchmarks/demo.py
echo "-----------------------------------------------"

# Part 2: Downloading and compiling llama.cpp fork (Apple Silicon - Metal)
echo ">>> [2/4] Downloading llama.cpp TurboQuant version for practical use..."
if [ ! -d "llama-cpp-turboquant" ]; then
    git clone https://github.com/TheTom/llama-cpp-turboquant.git
fi

cd llama-cpp-turboquant
echo ">>> Switching to the relevant working branch..."
git checkout feature/turboquant-kv-cache

echo ">>> Compiling C++ engine for Mac (Metal) (this may take a while)..."
cmake -B build -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Part 3: Downloading an Example Model
if [ -n "$1" ]; then
    model_choice="$1"
    echo ">>> [3/4] Model selected via argument: $model_choice"
else
    echo ">>> [3/4] Select the model you want to run:"
    echo "1) Llama 3.1 8B Instruct (~5 GB - Fast, General Purpose)"
    echo "2) Qwen 2.5 32B Instruct (~20 GB - Balanced, Good Performance)"
    echo "3) Command R+ 104B (~43 GB - Highest Quality, 100B+ Class)"
    echo "4) Qwen 2.5 0.5B Instruct (~400 MB - For Quick Testing Only)"
    echo "5) Llama-3-405B / 500B Class (~250 GB - Extreme Memory / NVMe SWAP Test)"
    read -p "Your choice (1/2/3/4/5) [Default: 4]: " model_choice
fi

mkdir -p models

case "$model_choice" in
  1|"8B"|"8b")
    MODEL_NAME="Llama 3.1 8B"
    MODEL_URL="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    MODEL_FILE="models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    ;;
  2|"32B"|"32b")
    MODEL_NAME="Qwen 2.5 32B"
    MODEL_URL="https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q4_K_M.gguf"
    MODEL_FILE="models/Qwen2.5-32B-Instruct-Q4_K_M.gguf"
    ;;
  3|"100B"|"100b")
    MODEL_NAME="Command R+ 104B"
    MODEL_URL="https://huggingface.co/mradermacher/c4ai-command-r-plus-08-2024-GGUF/resolve/main/c4ai-command-r-plus-08-2024.Q2_K.gguf"
    MODEL_FILE="models/c4ai-command-r-plus-08-2024.Q2_K.gguf"
    ;;
  5|"500B"|"500b")
    MODEL_NAME="Llama 3.1 405B (500B+ Class)"
    MODEL_URL="https://huggingface.co/mradermacher/Meta-Llama-3.1-405B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-405B-Instruct.Q2_K.gguf"
    MODEL_FILE="models/Meta-Llama-3.1-405B-Instruct.Q2_K.gguf"
    ;;
  *)
    MODEL_NAME="Qwen 2.5 0.5B"
    MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    MODEL_FILE="models/qwen2.5-0.5b-q4_k_m.gguf"
    ;;
esac

echo ">>> $MODEL_NAME model selected, preparing..."

if [ ! -f "$MODEL_FILE" ]; then
    echo ">>> Downloading model... ($MODEL_FILE)"
    curl -L -o "$MODEL_FILE" "$MODEL_URL"
else
    echo ">>> Model is already downloaded: $MODEL_FILE"
fi

# Part 4: Running the Model with TurboQuant Settings
echo ">>> [4/4] Starting the model with TurboQuant memory compression..."

if [[ "$model_choice" == "5" || "$model_choice" == "500"*"b" || "$model_choice" == "500"*"B" ]]; then
    echo ">>> 500B+ Class Model Detected: Activating EXTREME memory savings and NVMe swap optimizations..."
    # -b 128: Low batch to prevent memory spikes on swap
    # turbo2: 6.4x extreme compression to squeeze cache into remaining unified memory
    EXTRA_ARGS="-c 512 -b 128 -ub 64 -t 8 --no-mmap"
    CACHE_TYPE="turbo2"
    echo "    Extra parameters (NVMe Swap Safe, Low Memory Footprint): $EXTRA_ARGS with $CACHE_TYPE"
elif [[ "$model_choice" == "3" || "$model_choice" == "100"*"b" || "$model_choice" == "100"*"B" ]]; then
    echo ">>> 100B Class Model Detected: Activating maximum performance and stability settings for Apple Silicon M Series..."
    EXTRA_ARGS="-c 1024 -b 2048 -ub 512 -t 12 --no-mmap"
    CACHE_TYPE="turbo4"
    echo "    Extra parameters (Wired-memory freeze prevention, Wide Batch): $EXTRA_ARGS"
else
    EXTRA_ARGS="-c 4096"
    CACHE_TYPE="turbo4"
fi

# README recommends turbo4 to overcome M1/M2/M3 L2 Cache wall and accelerate dequantization
# (For 500B models, turbo2 is dynamically mandated to avoid out-of-memory)
echo "Used parameters: -ngl 99 $EXTRA_ARGS --cache-type-k $CACHE_TYPE --cache-type-v $CACHE_TYPE"
echo "-----------------------------------------------"

env TURBO_LAYER_ADAPTIVE=7 ./build/bin/llama-cli \
  -m "$MODEL_FILE" \
  -ngl 99 \
  $EXTRA_ARGS \
  -fa on \
  --cache-type-k $CACHE_TYPE \
  --cache-type-v $CACHE_TYPE \
  -p "Can you explain how we can compress the memory of an artificial intelligence model with a very simple story like a children's fairy tale?" \
  -n 300

echo "-----------------------------------------------"
echo ">>> Demo completed! You've run an LLM on your device with TurboQuant."
