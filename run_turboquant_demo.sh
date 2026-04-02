#!/bin/bash

# To prevent errors when running on Mac
set -e

# Starting Directory (Assuming you run this in the project root)
ROOT_DIR=$(pwd)

echo "=============== TURBOQUANT DEMO ==============="

# Helper: Detect and Install libomp (OpenMP) for high-speed CPU inference
SYSTEM_PROMPT="You are a helpful, creative, and professional AI assistant. You provide concise and accurate answers."
install_libomp() {
    echo ">>> Checking for libomp (OpenMP Support)..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if ! brew list libomp &>/dev/null; then
            echo ">>> macOS detected: Installing libomp via Homebrew..."
            brew install libomp
        fi
        LIBOMP_PREFIX=$(brew --prefix libomp)
        OMP_ENABLED="ON"
        OMP_C_FLAGS="-Xpreprocessor -fopenmp -I$LIBOMP_PREFIX/include"
        OMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$LIBOMP_PREFIX/include"
        OMP_LIB_LDFLAGS="-L$LIBOMP_PREFIX/lib"
        OMP_INC_CPPFLAGS="-I$LIBOMP_PREFIX/include"
        OMP_DYLD_PATH="$LIBOMP_PREFIX/lib"
        OMP_LIB_PATH="$LIBOMP_PREFIX/lib/libomp.dylib"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &>/dev/null; then
            if ! dpkg -s libomp-dev &>/dev/null; then
                echo ">>> Linux (Debian/Ubuntu) detected: Installing libomp via apt..."
                sudo apt-get update && sudo apt-get install -y libomp-dev
            fi
        elif command -v pacman &>/dev/null; then
            if ! pacman -Qs libomp &>/dev/null; then
                echo ">>> Linux (Arch) detected: Installing libomp via pacman..."
                sudo pacman -S --noconfirm libomp
            fi
        fi
        OMP_ENABLED="ON"
    elif [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "cygwin"* ]]; then
        echo ">>> Windows (MSYS2/Cygwin) detected: Checking for mingw-w64-libomp..."
        if command -v pacman &>/dev/null && ! pacman -Qs mingw-w64-x86_64-libomp &>/dev/null; then
            pacman -S --noconfirm mingw-w64-x86_64-libomp
        fi
        OMP_ENABLED="ON"
    else
        echo ">>> Unrecognized OS. Please ensure libomp is installed manually for OpenMP support."
        OMP_ENABLED="OFF"
    fi
}

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
install_libomp

if [ ! -d "llama-cpp-turboquant" ]; then
    git clone https://github.com/TheTom/llama-cpp-turboquant.git
fi

cd llama-cpp-turboquant
echo ">>> Switching to the relevant working branch..."
git checkout feature/turboquant-kv-cache

echo ">>> Compiling C++ engine with Dual Acceleration (Metal + OpenMP)..."
cmake -B build \
    -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OPENMP="${OMP_ENABLED:-OFF}" \
    -DOpenMP_C_FLAGS="${OMP_C_FLAGS:-}" \
    -DOpenMP_C_LIB_NAMES="omp" \
    -DOpenMP_CXX_FLAGS="${OMP_CXX_FLAGS:-}" \
    -DOpenMP_CXX_LIB_NAMES="omp" \
    -DOpenMP_omp_LIBRARY="${OMP_LIB_PATH:-}"

cmake --build build -j --target llama-cli

# Part 3: Memory Optimization Level
echo ">>> [3/5] Select Memory Optimization Level:"
echo "1) Performance (High GPU, fast, needs 32GB+ RAM)"
echo "2) Balanced   (Moderate GPU, 24GB RAM safe)"
echo "3) Ultra-Eco   (Minimal RAM, stable on 16GB systems)"
read -p "Your choice (1/2/3) [Default: 2]: " mem_choice
mem_choice=${mem_choice:-2}

case "$mem_choice" in
  1) MEM_BUDGET_PCT=80; MEM_LABEL="Performance" ;;
  2) MEM_BUDGET_PCT=30; MEM_LABEL="Balanced" ;;
  3) MEM_BUDGET_PCT=5; MEM_LABEL="Ultra-Eco" ;;
  *) MEM_BUDGET_PCT=30; MEM_LABEL="Balanced" ;;
esac

# Detect Performance Cores (P-Cores) for Apple Silicon
THREADS=$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo ">>> Memory Mode: $MEM_LABEL (Targeting $MEM_BUDGET_PCT% GPU weight budget)"
echo ">>> CPU Optimization: Using $THREADS performance cores for inference."
echo ""

# Part 4: Downloading an Example Model
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

# Check both possible model locations (legacy location and current)
MODEL_BASENAME=$(basename "$MODEL_FILE")
if [ -f "llama-cpp-turboquant/models/$MODEL_BASENAME" ]; then
    MODEL_FILE="llama-cpp-turboquant/models/$MODEL_BASENAME"
    echo ">>> Model found (existing): $MODEL_FILE"
elif [ ! -f "$MODEL_FILE" ]; then
    echo ">>> Downloading model... ($MODEL_FILE)"
    mkdir -p models
    curl -L -o "$MODEL_FILE" "$MODEL_URL"
else
    echo ">>> Model is already downloaded: $MODEL_FILE"
fi


# Part 4: Running the Model with TurboQuant Settings
echo ">>> [4/4] Starting the model with TurboQuant memory compression..."

if [[ "$model_choice" == "5" || "$model_choice" == "500"*"b" || "$model_choice" == "500"*"B" ]]; then
    echo ">>> 500B+ Class Model Detected: Activating EXTREME swap-safe settings..."
    EXTRA_ARGS="-c 512 -b 128 -ub 64 -t 8"
    CACHE_TYPE_K="turbo2"
    CACHE_TYPE_V="turbo2"
    echo "    Extra parameters (Extreme Swap): $EXTRA_ARGS with turbo2+2"
elif [[ "$model_choice" == "3" || "$model_choice" == "100"*"b" || "$model_choice" == "100"*"B" ]]; then
    EXTRA_ARGS="-c $CTX -b 512 -ub 256 -t 12 --repeat-penalty 1.1 --top-p 0.9"
    CACHE_TYPE_K="turbo4"
    CHAT_TEMPLATE="command-r"
    if [ "$mem_choice" -eq 3 ]; then CACHE_TYPE_V="turbo2"; else CACHE_TYPE_V="turbo4"; fi
    echo "    Extra parameters: $EXTRA_ARGS"
elif [[ "$model_choice" == "2" || "$model_choice" == "32B" || "$model_choice" == "32b" ]]; then
    echo ">>> 32B Class Model Detected: Tuning for $MEM_LABEL mode (Target 16GB RSS)..."
    # Drop context and batch to save peak memory
    CTX=512
    [ "$mem_choice" -eq 1 ] && CTX=1024
    [ "$mem_choice" -eq 3 ] && CTX=256
    EXTRA_ARGS="-c $CTX -b 32 -ub 32 -sm none --repeat-penalty 1.1 --top-p 0.9"
    # To hit 16GB, we use aggressive turbo2 for both K and V in Balanced/Eco
    CACHE_TYPE_K="turbo4"
    CACHE_TYPE_V="turbo2"
    CHAT_TEMPLATE="qwen2"
else
    # 8B and smaller: Use high precision (f16) to ensure maximum intelligence
    MODEL_NAME="Llama 3.1 8B"
    EXTRA_ARGS="-c 4096 -b 512 -ub 256 --repeat-penalty 1.1 --top-p 0.9"
    CACHE_TYPE_K="f16"
    CACHE_TYPE_V="f16"
    CHAT_TEMPLATE="llama3"
fi

# ---------------------------------------------------------------------------
# Auto-detect safe GPU layer count (NGL) to prevent Metal OOM
#
# Strategy (3-component budget model):
#   metal_budget = recommendedMaxWorkingSetSize from Metal driver
#   kv_budget    = estimated KV cache size (turbo4 compressed, current ctx)
#   graph_budget = ~1500 MB for ggml graph buffers, activation mem, overhead
#   weight_budget = metal_budget - kv_budget - graph_budget
#   safe_ngl = floor(weight_budget * 0.95 / bytes_per_layer)
#
# Using system_profiler first (exact driver value), sysctl as fallback.
# system_profiler SPHardwareDataType is faster than SPDisplaysDataType.
# ---------------------------------------------------------------------------
get_metal_budget_mb() {
    local budget=0

    # Primary: parse ioreg for IOAccelerator recommendedMaxWorkingSetSize
    # This is the exact same value llama.cpp reads internally.
    if command -v ioreg &>/dev/null; then
        budget=$(ioreg -r -d 1 -c AGXAccelerator 2>/dev/null \
            | grep -i 'recommendedMaxWorkingSetSize' \
            | awk '{gsub(/[^0-9]/,"",$NF); print $NF+0}' \
            | head -1)
        # Convert bytes → MB if needed (value > 1,000,000 means bytes)
        if [ -n "$budget" ] && [ "$budget" -gt 1000000 ] 2>/dev/null; then
            budget=$(( budget / 1024 / 1024 ))
        fi
    fi

    # Fallback: 75% of physical RAM (conservative, matches M-series observed ratio)
    if [ -z "$budget" ] || [ "$budget" -le 0 ] 2>/dev/null; then
        local total_bytes
        total_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
        budget=$(( total_bytes / 1024 / 1024 * 75 / 100 ))
    fi

    echo "${budget:-16384}"  # hard fallback: 16 GB
}

estimate_kv_mb() {
    # Estimate turbo4-compressed KV cache size in MB
    # Formula: ctx * layers * heads * head_dim * 2bytes * 2(K+V) / turbo4_ratio / 1024^2
    # turbo4 = 4.25 bits/value → ratio vs fp16 = 16/4.25 ≈ 3.76x
    local ctx_len="$1"
    local n_layers="$2"
    local n_heads="$3"
    local head_dim="$4"
    local cache_type="$5"

    local ratio=376  # turbo4: 3.76x (× 100 for integer math)
    [ "$cache_type" = "turbo2" ] && ratio=640   # turbo2: 6.4x
    [ "$cache_type" = "turbo3" ] && ratio=490   # turbo3: 4.9x
    [ "$cache_type" = "q8_0" ]   && ratio=200   # q8_0: 2x

    # raw = ctx * layers * heads * head_dim * 2 bytes * 2 (K+V) in MB
    local raw_mb=$(( ctx_len * n_layers * n_heads * head_dim * 4 / 1024 / 1024 ))
    local compressed_mb=$(( raw_mb * 100 / ratio ))
    echo "$compressed_mb"
}

calculate_ngl() {
    local model_file="$1"
    local total_layers="$2"
    local n_heads="$3"
    local head_dim="$4"
    local ctx_len="$5"
    local cache_type="$6"

    local metal_budget_mb
    metal_budget_mb=$(get_metal_budget_mb)

    local model_size_mb=0
    if [ -f "$model_file" ]; then
        model_size_mb=$(( $(stat -f%z "$model_file" 2>/dev/null || echo 0) / 1024 / 1024 ))
    fi

    if [ "$model_size_mb" -eq 0 ] || [ "$total_layers" -eq 0 ]; then
        echo 99
        return
    fi

    local bytes_per_layer_mb=$(( model_size_mb / total_layers ))

    # Estimate KV cache size at given context
    local kv_mb
    kv_mb=$(estimate_kv_mb "$ctx_len" "$total_layers" "$n_heads" "$head_dim" "$cache_type")

    # ggml compute graph + activation buffers + Metal overhead
    # Increased to 2500MB as 32B models have larger intermediate activation tensors
    local graph_overhead_mb=2500

    # Weight budget = total budget − KV − graph overhead
    local weight_budget_mb=$(( metal_budget_mb - kv_mb - graph_overhead_mb ))

    [ "$weight_budget_mb" -le 0 ] && weight_budget_mb=$(( metal_budget_mb / 3 ))

    # weight_budget_mb * MEM_BUDGET_PCT / 100 — User selected headroom
    local available_mb=$(( weight_budget_mb * MEM_BUDGET_PCT / 100 ))

    local safe_ngl=$(( available_mb / bytes_per_layer_mb ))

    [ "$safe_ngl" -gt "$total_layers" ] && safe_ngl=$total_layers
    [ "$safe_ngl" -lt 1 ] && safe_ngl=1

    echo "$safe_ngl"
}

# Per-model architecture parameters (heads, head_dim, context already set in EXTRA_ARGS)
# Parse context from EXTRA_ARGS for KV estimation
CTX_LEN=$(echo "$EXTRA_ARGS" | grep -o '\-c [0-9]*' | awk '{print $2}')
[ -z "$CTX_LEN" ] && CTX_LEN=2048

case "$model_choice" in
  1|"8B"|"8b")     NUM_LAYERS=32; N_HEADS=32; HEAD_DIM=128 ;;
  2|"32B"|"32b")   NUM_LAYERS=64; N_HEADS=40; HEAD_DIM=128 ;;
  3|"100B"|"100b") NUM_LAYERS=96; N_HEADS=128; HEAD_DIM=128 ;;
  5|"500B"|"500b") NUM_LAYERS=126; N_HEADS=128; HEAD_DIM=128 ;;
  *)               NUM_LAYERS=24; N_HEADS=8; HEAD_DIM=64 ;;  # 0.5B
esac

METAL_BUDGET=$(get_metal_budget_mb)
KV_EST=$(estimate_kv_mb "$CTX_LEN" "$NUM_LAYERS" "$N_HEADS" "$HEAD_DIM" "$CACHE_TYPE_K")
NGL=$(calculate_ngl "$MODEL_FILE" "$NUM_LAYERS" "$N_HEADS" "$HEAD_DIM" "$CTX_LEN" "$CACHE_TYPE_K")

echo ">>> Memory budget breakdown:"
echo "    Metal GPU budget:  ~${METAL_BUDGET} MB"
echo "    KV cache (${CACHE_TYPE}, ctx=${CTX_LEN}): ~${KV_EST} MB"
echo "    Safe GPU layers:   ${NGL}/${NUM_LAYERS}"

# README recommends turbo4 to overcome M1/M2/M3 L2 Cache wall and accelerate dequantization
# Export library paths for OpenMP runtime
if [[ "$OSTYPE" == "darwin"* ]]; then
    export LDFLAGS="${OMP_LIB_LDFLAGS:-}"
    export CPPFLAGS="${OMP_INC_CPPFLAGS:-}"
    export DYLD_LIBRARY_PATH="${OMP_DYLD_PATH:-}:$DYLD_LIBRARY_PATH"
fi

env TURBO_LAYER_ADAPTIVE=7 ./build/bin/llama-cli \
  -m "$MODEL_FILE" \
  -ngl "$NGL" \
  -t "$THREADS" \
  $EXTRA_ARGS \
  -fa on \
  -cnv \
  -sp "$SYSTEM_PROMPT" \
  --chat-template "$CHAT_TEMPLATE" \
  --cache-type-k "$CACHE_TYPE_K" \
  --cache-type-v "$CACHE_TYPE_V"
  -p "Can you explain how we can compress the memory of an artificial intelligence model with a very simple story like a children's fairy tale?" \
  -n 300

echo "-----------------------------------------------"
echo ">>> Demo completed! You've run an LLM on your device with TurboQuant."
