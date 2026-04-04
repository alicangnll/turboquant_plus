#!/bin/bash

# To prevent errors when running on Mac
set -e

# Starting Directory (Assuming you run this in the project root)
ROOT_DIR=$(pwd)
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

echo "=============== TURBOQUANT+ DEMO ==============="
# RAM goals and RSS measurement: see docs/memory-rss-targets.md and scripts/measure_rss_macos.sh

# Helper: Detect and Install libomp (OpenMP) for high-speed CPU inference
# [TURBO 2.1] Turkish Technical System Prompt
SYSTEM_PROMPT="Sen TurboQuant+ (2026 Edition) yardımcısısın. Apple Silicon üzerinde LLMTuning optimizasyonu ile en yüksek performansta çalışıyorsun. Sorulara teknik, kesin ve yardımsever yanıtlar vermelisin. Yanıtların her zaman Türkçe olmalıdır (kullanıcı aksini istemedikçe)."
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
    else
        OMP_ENABLED="OFF"
    fi
}

# Part 1: Downloading and compiling llama.cpp fork
echo ">>> [1/4] Preparing C++ engine..."
install_libomp

cd llama-cpp-turboquant

# User's choice: Fresh build for maximum stability
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo ">>> Cleaning build for fresh optimization..."
    rm -rf build
fi

if [ ! -d "build" ]; then
    echo ">>> Compiling C++ engine with Dual Acceleration (Metal + OpenMP)..."
    cmake -B build \
        -DGGML_METAL=ON \
        -DGGML_METAL_EMBED_LIBRARY=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CPU_REPACK=ON \
        -DGGML_OPENMP="${OMP_ENABLED:-OFF}" \
        -DOpenMP_C_FLAGS="${OMP_C_FLAGS:-}" \
        -DOpenMP_C_LIB_NAMES="omp" \
        -DOpenMP_CXX_FLAGS="${OMP_CXX_FLAGS:-}" \
        -DOpenMP_CXX_LIB_NAMES="omp" \
        -DOpenMP_omp_LIBRARY="${OMP_LIB_PATH:-}"
    cmake --build build -j --target llama-cli
fi

# Part 2: Memory Optimization Level
echo ">>> [2/4] Select Memory Optimization Level (primary tuning: peak RSS — see docs/memory-rss-targets.md):"
echo "1) Performance (High GPU, fast, needs 32GB+ RAM)"
echo "2) Balanced    (Moderate GPU, 24GB RAM safe)"
echo "3) Ultra-Eco    (Minimal RAM, stable on 16GB systems)"
read -p "Your choice (1/2/3) [Default: 2]: " mem_choice
mem_choice=${mem_choice:-2}

case "$mem_choice" in
  1) MEM_BUDGET_PCT=90; MEM_LABEL="Performance" ;;
  2) MEM_BUDGET_PCT=40; MEM_LABEL="Balanced" ;;
  3) MEM_BUDGET_PCT=10; MEM_LABEL="Ultra-Eco" ;;
  *) MEM_BUDGET_PCT=40; MEM_LABEL="Balanced" ;;
esac

THREADS=$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Part 3: Model Selection
if [ -z "$model_choice" ]; then
    echo ">>> [3/4] Select the model you want to run:"
    echo "1) Llama 3.1 8B Instruct (~5 GB)"
    echo "2) Qwen 2.5 32B Instruct (~20 GB)"
    echo "3) Command R+ 104B (~43 GB)"
    echo "4) Qwen 2.5 0.5B Instruct (~400 MB)"
    echo "6) GPT 20B (OpenAI OSS-20B Class - ~12 GB)"
    echo "7) Gemma 4 31B (Google - ~18 GB)"
    read -p "Your choice (1/2/3/4/6/7) [Default: 1]: " model_choice
    model_choice=${model_choice:-1}
fi

# Use LLMTuning Policy Bridge (Python)
CONFIG_JSON="$ROOT_DIR/tq_cli_config_${model_choice}_${mem_choice}.json"
echo ">>> Calling LLMTuning Bridge..."
if python3 -m turboquant.cli_config_export --model-choice "$model_choice" --mem-choice "$mem_choice" > "$CONFIG_JSON"; then
    echo ">>> LLMTuning: Policy exported."
else
    echo ">>> Error: Python bridge failed. Check PYTHONPATH or script."
    exit 1
fi

# Parse Config
CTX=$(python3 -c "import json; print(json.load(open('$CONFIG_JSON'))['ctx_len'])")
CACHE_V=$(python3 -c "import json; print(json.load(open('$CONFIG_JSON'))['cache_type_v'])")
CACHE_K=$(python3 -c "import json; print(json.load(open('$CONFIG_JSON'))['cache_type_k'])")
LAYERS=$(python3 -c "import json; print(json.load(open('$CONFIG_JSON'))['num_layers'])")
HEADS=$(python3 -c "import json; print(json.load(open('$CONFIG_JSON'))['num_heads'])")
DIM=$(python3 -c "import json; print(json.load(open('$CONFIG_JSON'))['head_dim'])")
TEMPLATE=$(python3 -c "import json; print(json.load(open('$CONFIG_JSON'))['chat_template'])")
EXTRA_POLICY=$(python3 -c "import json; print(json.load(open('$CONFIG_JSON')).get('extra_args', ''))")
BATCH_SIZE=$(python3 -c "import json; print(json.load(open('$CONFIG_JSON'))['batch_size'])")
UBATCH_SIZE=$(python3 -c "import json; print(json.load(open('$CONFIG_JSON'))['ubatch_size'])")
RSS_NOTE=$(python3 -c "import json; print(json.load(open('$CONFIG_JSON')).get('rss_target_note', ''))")
LOSSLESS_NOTE=$(python3 -c "import json; print(json.load(open('$CONFIG_JSON')).get('lossless_definition', ''))")

# Model Mapping
case "$model_choice" in
  1) MODEL_NAME="Llama 3.1 8B"; MODEL_URL="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"; MODEL_FILE="models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" ;;
  2) MODEL_NAME="Qwen 2.5 32B"; MODEL_URL="https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q4_K_M.gguf"; MODEL_FILE="models/Qwen2.5-32B-Instruct-Q4_K_M.gguf" ;;
  3) MODEL_NAME="Command R+ 104B"; MODEL_URL="https://huggingface.co/mradermacher/c4ai-command-r-plus-08-2024-GGUF/resolve/main/c4ai-command-r-plus-08-2024.Q2_K.gguf"; MODEL_FILE="models/c4ai-command-r-plus-08-2024.Q2_K.gguf" ;;
  6) MODEL_NAME="GPT 20B OSS"; MODEL_URL="https://huggingface.co/bartowski/openai_gpt-oss-20b-GGUF/resolve/main/openai_gpt-oss-20b-Q4_K_M.gguf"; MODEL_FILE="models/openai_gpt-oss-20b-Q4_K_M.gguf" ;;
  7) MODEL_NAME="Gemma 4 31B"; MODEL_URL="https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/resolve/main/gemma-4-31B-it-Q4_K_M.gguf"; MODEL_FILE="models/gemma-4-31B-it-Q4_K_M.gguf" ;;
  *) MODEL_NAME="Qwen 0.5B"; MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"; MODEL_FILE="models/qwen2.5-0.5b-q4_k_m.gguf" ;;
esac

if [ ! -f "../$MODEL_FILE" ]; then
    echo ">>> Downloading $MODEL_NAME..."
    mkdir -p ../models
    curl -L -o "../$MODEL_FILE" "$MODEL_URL"
    MODEL_FILE="../$MODEL_FILE"
else
    MODEL_FILE="../$MODEL_FILE"
fi

# Advanced Memory Budgeting (NGL Calculation)

estimate_kv() {
    local c=$1; local l=$2; local h=$3; local d=$4; local t=$5
    local r=376 # turbo4
    [[ "$t" == "turbo2" ]] && r=640
    [[ "$t" == "q8_0" ]] && r=200
    echo $(( c * l * h * d * 4 * 100 / r / 1024 / 1024 ))
}

METAL_MB=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 17179869184) * 75 / 100 / 1024 / 1024 ))
KV_MB=$(estimate_kv "$CTX" "$LAYERS" "$HEADS" "$DIM" "$CACHE_K")
MODEL_MB=$(( $(stat -f%z "$MODEL_FILE") / 1024 / 1024 ))

# [TURBO 2.1] Dynamic Graph Budget (Scales with model size)
# For small models, 100-200MB is enough. For giant models, we need 5GB+.
GRAPH_MB=$(( MODEL_MB / 2 )) 
[[ $GRAPH_MB -gt 5000 ]] && GRAPH_MB=5000
[[ $GRAPH_MB -lt 100 ]] && GRAPH_MB=100

# 8B + Ultra-Eco: 10%% weight budget + hybrid CPU/Metal with aggressive turbo KV produced garbage tokens.
# Policy JSON uses q8_0 KV; raise GPU layer budget here (model_choice is known; MEM_BUDGET_PCT was set before model pick).
if [[ "$mem_choice" == "3" && "$model_choice" == "1" ]]; then
  MEM_BUDGET_PCT=55
  echo ">>> 8B Ultra-Eco: weight budget ${MEM_BUDGET_PCT}% + q8_0 KV (policy) for stable full-layer Metal offload."
fi

AVAIL_MB=$(python3 -c "print(int(($METAL_MB - $KV_MB - $GRAPH_MB) * $MEM_BUDGET_PCT / 100))")
BPL=$(( MODEL_MB / LAYERS ))
NGL=$(( AVAIL_MB / BPL ))
[[ $NGL -gt $LAYERS ]] && NGL=$LAYERS
[[ $NGL -lt 1 ]] && NGL=1

echo ">>> TurboQuant+ Standard Sharding:"
echo "    Budget: ${METAL_MB} MB | Target: ${AVAIL_MB} MB for Weights"
echo "    Policy: Cache-K=$CACHE_K, Cache-V=$CACHE_V, Context=$CTX"
echo "    Batch/Ubatch: $BATCH_SIZE / $UBATCH_SIZE (from LLMTuning policy JSON)"
[[ -n "$RSS_NOTE" ]] && echo "    RSS goal: $RSS_NOTE"
[[ -n "$LOSSLESS_NOTE" ]] && echo "    Quality note: $LOSSLESS_NOTE"

CLI_CMD="./build/bin/llama-cli -m \"$MODEL_FILE\" -ngl $NGL -t $THREADS -c $CTX --batch-size $BATCH_SIZE --ubatch-size $UBATCH_SIZE $EXTRA_POLICY"
if [[ "$mem_choice" == "3" ]]; then
    echo "    Status: Ultra-Eco — lowest peak-RSS tier; if quality suffers, use option 2 (Balanced)."
else
    echo "    Status: LLMTuning Active Sharding (tier $mem_choice)."
fi

# Special Chat Handling for 20B MoE
if [[ "$TEMPLATE" == "none" ]]; then
    CLI_CMD="$CLI_CMD -p \"$SYSTEM_PROMPT\nUser: Can you explain model compression?\nAI:\""
else
    CLI_CMD="$CLI_CMD -cnv --chat-template \"$TEMPLATE\" -sys \"$SYSTEM_PROMPT\""
fi

# TurboQuant+ Core Flags (Sparse-V: off by default; export TURBO_SPARSE_V=1 to opt into GPU skip)
CLI_CMD="$CLI_CMD --turbo-async --sparse-v-threshold -1 --cache-type-k $CACHE_K --cache-type-v $CACHE_V"

echo ">>> Launching Engine..."
export TURBO_ASYNC_PIPELINE=1
export TURBO_LAYER_ADAPTIVE=7

eval "$CLI_CMD -n 256"