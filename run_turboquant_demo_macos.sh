#!/bin/bash

# macOS üzerindeki çökmeleri engellemek için
set -e

# Başlangıç Dizinini Kaydet
ROOT_DIR=$(pwd)

echo "=============== TURBOQUANT DEMO ==============="

# Sistem Komutu (Türkçe Etkileşim)
SYSTEM_PROMPT="Sen Türkçe konuşan, oldukça zeki, kibar ve her konuda yardımcı olan bir yapay zeka asistanısın."

install_libomp() {
    echo ">>> libomp (OpenMP Desteği) kontrol ediliyor..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if ! brew list libomp &>/dev/null; then
            echo ">>> macOS algılandı: libomp Homebrew üzerinden kuruluyor..."
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
        echo ">>> Linux/Windows algılandı, varsayılan OpenMP ayarları kullanılıyor."
        OMP_ENABLED="ON"
    fi
}

# 1. Python Ortamı Hazırlığı
echo ">>> [1/4] Python ortamı ve bağımlılıklar hazırlanıyor..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip >/dev/null 2>&1
pip install -e "." >/dev/null 2>&1

# 2. C++ Motorunun İndirilmesi ve Derlenmesi
echo ">>> [2/4] llama.cpp TurboQuant motoru hazırlanıyor..."
install_libomp

if [ ! -d "llama-cpp-turboquant" ]; then
    git clone https://github.com/TheTom/llama-cpp-turboquant.git
fi

cd llama-cpp-turboquant
echo ">>> İlgili branch'e geçiliyor..."
git checkout feature/turboquant-kv-cache >/dev/null 2>&1 || true

echo ">>> C++ motoru Dual Acceleration (Metal + OpenMP) ile derleniyor..."
cmake -B build \
    -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OPENMP="${OMP_ENABLED:-OFF}" \
    -DOpenMP_C_FLAGS="${OMP_C_FLAGS:-}" \
    -DOpenMP_C_LIB_NAMES="omp" \
    -DOpenMP_CXX_FLAGS="${OMP_CXX_FLAGS:-}" \
    -DOpenMP_CXX_LIB_NAMES="omp" \
    -DOpenMP_omp_LIBRARY="${OMP_LIB_PATH:-}" >/dev/null 2>&1

cmake --build build -j --target llama-cli >/dev/null 2>&1
cd ..

# 3. Bellek Profili Seçimi
echo ">>> [3/5] Bellek Optimizasyon Seviyesini Seçin:"
echo "1) Performance (Yüksek GPU kullanımı, +32GB RAM gerekli)"
echo "2) Balanced   (Dengeli, 24GB RAM sistemler için güvenli)"
echo "3) Ultra-Eco  (AirLLM Tetikleyici! - 16GB ve altı için minimum RAM)"
read -p "Seçiminiz (1/2/3) [Varsayılan: 2]: " mem_choice
mem_choice=${mem_choice:-2}

case "$mem_choice" in
  1) MEM_BUDGET_PCT=80; MEM_LABEL="Performance" ;;
  2) MEM_BUDGET_PCT=30; MEM_LABEL="Balanced" ;;
  3) MEM_BUDGET_PCT=5;  MEM_LABEL="Ultra-Eco" ;;
  *) MEM_BUDGET_PCT=30; MEM_LABEL="Balanced" ;;
esac

THREADS=$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo ">>> Seçilen Mod: $MEM_LABEL (Hedeflenen GPU Yükü: %$MEM_BUDGET_PCT)"
echo ">>> CPU Optimizasyonu: Çıkarım için $THREADS performans çekirdeği kullanılacak."
echo ""

# 4. Model Seçimi ve İndirilmesi
if [ -n "$1" ]; then
    model_choice="$1"
    echo ">>> [3/4] Argüman ile model seçildi: $model_choice"
else
    echo ">>> [3/4] Çalıştırmak istediğiniz modeli seçin:"
    echo "1) Llama 3.1 8B Instruct (~5 GB - Hızlı)"
    echo "2) Qwen 2.5 32B Instruct (~20 GB - ÖNERİLEN)"
    echo "3) Command R+ 104B (~43 GB - Yüksek Kalite)"
    echo "4) Qwen 2.5 0.5B Instruct (~400 MB - Test İçin)"
    echo "6) GPT 20B (OpenAI OSS-20B MoE - ~12 GB)"
    read -p "Seçiminiz (1/2/3/4/6) [Varsayılan: 2]: " model_choice
    model_choice=${model_choice:-2}
fi

mkdir -p llama-cpp-turboquant/models

case "$model_choice" in
  1|"8B"|"8b")
    MODEL_NAME="Llama 3.1 8B"
    MODEL_URL="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    MODEL_FILE="llama-cpp-turboquant/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    ;;
  2|"32B"|"32b")
    MODEL_NAME="Qwen 2.5 32B"
    MODEL_URL="https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q4_K_M.gguf"
    MODEL_FILE="llama-cpp-turboquant/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf"
    ;;
  3|"100B"|"100b")
    MODEL_NAME="Command R+ 104B"
    MODEL_URL="https://huggingface.co/mradermacher/c4ai-command-r-plus-08-2024-GGUF/resolve/main/c4ai-command-r-plus-08-2024.Q2_K.gguf"
    MODEL_FILE="llama-cpp-turboquant/models/c4ai-command-r-plus-08-2024.Q2_K.gguf"
    ;;
  6|"20B"|"20b")
    MODEL_NAME="GPT 20B (OSS)"
    MODEL_URL="https://huggingface.co/bartowski/openai_gpt-oss-20b-GGUF/resolve/main/openai_gpt-oss-20b-Q4_K_M.gguf"
    MODEL_FILE="llama-cpp-turboquant/models/openai_gpt-oss-20b-Q4_K_M.gguf"
    ;;
  *)
    MODEL_NAME="Qwen 2.5 0.5B"
    MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    MODEL_FILE="llama-cpp-turboquant/models/qwen2.5-0.5b-q4_k_m.gguf"
    ;;
esac

echo ">>> $MODEL_NAME modeli seçildi, kontrol ediliyor..."
if [ ! -f "$MODEL_FILE" ]; then
    echo ">>> Model indiriliyor... ($MODEL_FILE)"
    curl -L -o "$MODEL_FILE" "$MODEL_URL"
else
    echo ">>> Model diskte mevcut: $MODEL_FILE"
fi

# 5. Model Parametreleri ve Çıkarım Konfigürasyonu
echo ">>> [4/4] TurboQuant Hafıza Sıkıştırma ayarları yapılandırılıyor..."

# 5.1 Modellerin Bağlam (Context) Ayarları
if [[ "$model_choice" == "3" || "$model_choice" == "100"*"b" || "$model_choice" == "100"*"B" ]]; then
    EXTRA_ARGS="-c 1024 -b 512 -ub 256 -t 12 --repeat-penalty 1.1 --top-p 0.9"
    CHAT_TEMPLATE="command-r"
elif [[ "$model_choice" == "2" || "$model_choice" == "32B" || "$model_choice" == "32b" ]]; then
    CTX=512; [ "$mem_choice" -eq 1 ] && CTX=1024; [ "$mem_choice" -eq 3 ] && CTX=256
    EXTRA_ARGS="-c $CTX -b 32 -ub 32 -sm none --repeat-penalty 1.1 --top-p 0.9"
    CHAT_TEMPLATE="qwen2"
elif [[ "$model_choice" == "6" || "$model_choice" == "20B" || "$model_choice" == "20b" ]]; then
    EXTRA_ARGS="-c 2048 -b 512 -ub 256 --repeat-penalty 1.1 --top-p 0.9 -fa on"
    CHAT_TEMPLATE=""
else
    # 8B ve altı modeller
    EXTRA_ARGS="-c 1024 -b 512 -ub 256 --repeat-penalty 1.1 --top-p 0.9"
    CHAT_TEMPLATE=""
fi

# 5.2 BELLEK MODUNA GÖRE SIKIŞTIRMA (AirLLM ZORLAMASI)
if [ "$mem_choice" -eq 3 ]; then
    # ULTRA-ECO: Model ne olursa olsun AirLLM'i tetiklemek için K=4, V=2 yapılır!
    CACHE_TYPE_K="turbo4"
    CACHE_TYPE_V="turbo2"
elif [ "$mem_choice" -eq 2 ]; then
    # BALANCED: Dengeli turbo4 sıkıştırması
    CACHE_TYPE_K="turbo4"
    CACHE_TYPE_V="turbo4"
else
    # PERFORMANCE: Küçük modellere q8_0 yüksek kalite, büyüklere turbo4
    if [[ "$model_choice" == "1" || "$model_choice" == "4" ]]; then
        CACHE_TYPE_K="q8_0"
        CACHE_TYPE_V="q8_0"
    else
        CACHE_TYPE_K="turbo4"
        CACHE_TYPE_V="turbo4"
    fi
fi

# Katman ve Donanım Hesaplamaları
CTX_LEN=$(echo "$EXTRA_ARGS" | grep -o '\-c [0-9]*' | awk '{print $2}')
[ -z "$CTX_LEN" ] && CTX_LEN=2048

case "$model_choice" in
  1|"8B"|"8b")     NUM_LAYERS=32; N_HEADS=32; HEAD_DIM=128 ;;
  2|"32B"|"32b")   NUM_LAYERS=64; N_HEADS=40; HEAD_DIM=128 ;;
  3|"100B"|"100b") NUM_LAYERS=96; N_HEADS=128; HEAD_DIM=128 ;;
  6|"20B"|"20b")   NUM_LAYERS=24; N_HEADS=8; HEAD_DIM=64 ;;
  *)               NUM_LAYERS=24; N_HEADS=8; HEAD_DIM=64 ;;
esac

get_metal_budget_mb() {
    local budget=0
    if command -v ioreg &>/dev/null; then
        budget=$(ioreg -r -d 1 -c AGXAccelerator 2>/dev/null | grep -i 'recommendedMaxWorkingSetSize' | awk '{gsub(/[^0-9]/,"",$NF); print $NF+0}' | head -1)
        if [ -n "$budget" ] && [ "$budget" -gt 1000000 ] 2>/dev/null; then
            budget=$(( budget / 1024 / 1024 ))
        fi
    fi
    if [ -z "$budget" ] || [ "$budget" -le 0 ] 2>/dev/null; then
        local total_bytes
        total_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
        budget=$(( total_bytes / 1024 / 1024 * 75 / 100 ))
    fi
    echo "${budget:-16384}"
}

METAL_BUDGET=$(get_metal_budget_mb)
NGL=$(( METAL_BUDGET / 300 )) # Basit bir katman limiti koruması
[ "$NGL" -gt "$NUM_LAYERS" ] && NGL=$NUM_LAYERS
[ "$NGL" -lt 1 ] && NGL=1

echo ">>> Bellek Dağılımı:"
echo "    Metal GPU Bütçesi:  ~${METAL_BUDGET} MB"
echo "    KV Cache Sıkıştırma: K=${CACHE_TYPE_K}, V=${CACHE_TYPE_V}"

if [[ "$OSTYPE" == "darwin"* ]]; then
    export LDFLAGS="${OMP_LIB_LDFLAGS:-}"
    export CPPFLAGS="${OMP_INC_CPPFLAGS:-}"
    export DYLD_LIBRARY_PATH="${OMP_DYLD_PATH:-}:$DYLD_LIBRARY_PATH"
fi

# 6. SOHBET BAŞLATMA (C++ ve Python Otomatik Geçişi)
USE_LLMTUNING=0
if [[ "$CACHE_TYPE_K" == "turbo4" && "$CACHE_TYPE_V" == "turbo2" ]]; then
    USE_LLMTUNING=1
fi

# C++ Motoru Argümanları (Hatalı -i silindi, -cnv ile interaktif sohbet aktif)
CLI_CMD="./llama-cpp-turboquant/build/bin/llama-cli -m \"$MODEL_FILE\" -ngl \"$NGL\" -t \"$THREADS\" $EXTRA_ARGS -fa on -cnv --cache-type-k \"$CACHE_TYPE_K\" --cache-type-v \"$CACHE_TYPE_V\""
CLI_CMD="$CLI_CMD -sys \"$SYSTEM_PROMPT\""
[ -n "$CHAT_TEMPLATE" ] && CLI_CMD="$CLI_CMD --chat-template \"$CHAT_TEMPLATE\""

if [ "$USE_LLMTUNING" -eq 1 ]; then
    echo "==========================================================="
    echo "🚀 [LLMTuning (AirLLM) Mantığı Devrede] <<<"
    echo ">>> Mod: Bellek Optimize Edilmiş Katman Parçalama"
    echo ">>> Fayda: 32B+ modelleri ~2 GB RAM ile tam istikrarlı çalıştırma"
    echo "==========================================================="
    
    # Python motoru ile İNTERAKTİF SOHBETİ başlat (-p ve -n parametreleri SİLİNDİ)
    PYTHONPATH="." python3 -m turboquant.streamed_inference -m "$MODEL_FILE" --model-size "$NUM_LAYERS" --cache-type-k "$CACHE_TYPE_K" --cache-type-v "$CACHE_TYPE_V"
else
    echo "==========================================================="
    echo "🚀 [Standart llama.cpp + TurboQuant Devrede] <<<"
    echo "==========================================================="
    
    # C++ motoru ile İNTERAKTİF SOHBETİ başlat
    eval "env TURBO_LAYER_ADAPTIVE=7 $CLI_CMD"
fi

echo "-----------------------------------------------"
echo ">>> Sohbet Sonlandırıldı."