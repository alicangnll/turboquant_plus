@echo off
setlocal enabledelayedexpansion

echo =============== TURBOQUANT WINDOWS DEMO ===============

set "SYSTEM_PROMPT=You are a Technical Research AI. Respond in the user's language when they write in that language."

:: Part 1: Python environment setup
echo ^>>> [1/4] Setting up Python environment...
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found. Please install Python and add it to your PATH.
    pause
    exit /b 1
)

if not exist ".venv" (
    python -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e "."

echo ^>>> Starting theoretical compression demo via Python...
python benchmarks\demo.py
echo -----------------------------------------------

:: Part 2: llama.cpp TurboQuant
echo ^>>> [2/4] Downloading and compiling llama.cpp TurboQuant...

if not exist "llama-cpp-turboquant" (
    git clone https://github.com/TheTom/llama-cpp-turboquant.git
)

cd llama-cpp-turboquant
echo ^>>> Switching to the relevant working branch...
git checkout feature/turboquant-kv-cache

echo ^>>> Compiling C++ engine (Windows Architecture)...
if not exist "build" mkdir build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=ON
cmake --build build --config Release -j --target llama-cli

cd ..

:: Part 3: Memory Optimization Level
echo ^>>> [3/5] Select Memory Optimization Level:
echo 1^) Performance ^(High GPU, Needs 32GB+ RAM^)
echo 2^) Balanced    ^(Moderate GPU, 24GB RAM safe^)
echo 3^) Ultra-Eco    ^(Minimal RAM, stable on 16GB systems^)

set /p mem_choice="Your choice (1/2/3) [Default: 2]: "
if "%mem_choice%"=="" set mem_choice=2

if "%mem_choice%"=="1" (
    set MEM_BUDGET_PCT=80
    set MEM_LABEL=Performance
) else if "%mem_choice%"=="3" (
    set MEM_BUDGET_PCT=5
    set MEM_LABEL=Ultra-Eco
) else (
    set MEM_BUDGET_PCT=30
    set MEM_LABEL=Balanced
)

:: Part 4: Model Selection
echo ^>>> [3/4] Select the model you want to run:
echo 1^) Llama 3.1 8B Instruct ^(~5 GB^)
echo 2^) Qwen 2.5 32B Instruct ^(~20 GB^)
echo 3^) Command R+ 104B ^(~43 GB^)
echo 4^) Qwen 2.5 0.5B Instruct ^(~400 MB^)
echo 5^) Llama-3-405B / 500B Class ^(~250 GB^)
echo 6^) GPT 20B ^(MoE - ~12 GB^)

set /p model_choice="Your choice (1/2/3/4/5/6) [Default: 4]: "
if "%model_choice%"=="" set model_choice=4

if not exist "models" mkdir models

if "%model_choice%"=="1" (
    set MODEL_NAME=Llama 3.1 8B
    set MODEL_URL=https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
    set MODEL_FILE=models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
) else if "%model_choice%"=="2" (
    set MODEL_NAME=Qwen 2.5 32B
    set MODEL_URL=https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q4_K_M.gguf
    set MODEL_FILE=models\Qwen2.5-32B-Instruct-Q4_K_M.gguf
) else if "%model_choice%"=="3" (
    set MODEL_NAME=Command R+ 104B
    set MODEL_URL=https://huggingface.co/mradermacher/c4ai-command-r-plus-08-2024-GGUF/resolve/main/c4ai-command-r-plus-08-2024.Q2_K.gguf
    set MODEL_FILE=models\c4ai-command-r-plus-08-2024.Q2_K.gguf
) else if "%model_choice%"=="5" (
    set MODEL_NAME=Llama 3.1 405B
    set MODEL_URL=https://huggingface.co/mradermacher/Meta-Llama-3.1-405B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-405B-Instruct.Q2_K.gguf
    set MODEL_FILE=models\Meta-Llama-3.1-405B-Instruct.Q2_K.gguf
) else if "%model_choice%"=="6" (
    set MODEL_NAME=GPT 20B (OSS)
    set MODEL_URL=https://huggingface.co/bartowski/openai_gpt-oss-20b-GGUF/resolve/main/openai_gpt-oss-20b-Q4_K_M.gguf
    set MODEL_FILE=models\openai_gpt-oss-20b-Q4_K_M.gguf
) else (
    set MODEL_NAME=Qwen 2.5 0.5B
    set MODEL_URL=https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
    set MODEL_FILE=models\qwen2.5-0.5b-q4_k_m.gguf
)

echo ^>>> %MODEL_NAME% model selected, preparing...

if not exist "%MODEL_FILE%" (
    echo ^>>> Downloading model...
    curl -L -o "%MODEL_FILE%" "%MODEL_URL%"
) else (
    echo ^>>> Model is already downloaded: %MODEL_FILE%
)

:: Part 4: LLMTuning policy (turbo KV only) + run
echo ^>>> [4/4] Starting the model with TurboQuant...

set PYTHONPATH=%CD%
echo ^>>> Calling LLMTuning Bridge...
python -m turboquant.cli_config_export --model-choice %model_choice% --mem-choice %mem_choice% --emit bat > "%TEMP%\tq_turboquant_env.bat"
if errorlevel 1 (
    echo [ERROR] LLMTuning bridge failed. Check Python path and turboquant package.
    pause
    exit /b 1
)
call "%TEMP%\tq_turboquant_env.bat"

:: NGL heuristics (Windows); ctx / cache / batch come from policy JSON above.
set NGL=32
if "%model_choice%"=="2" set NGL=24
if "%model_choice%"=="3" set NGL=15
if "%model_choice%"=="5" set NGL=5
if "%model_choice%"=="6" set NGL=12
if "%mem_choice%"=="3" if "%model_choice%"=="1" set NGL=32

echo ^>>> Memory Mode: %MEM_LABEL%
echo ^>>> Running !MODEL_NAME! ctx=!CTX! cache=!CACHE_TYPE_K!/!CACHE_TYPE_V! NGL=!NGL!

set TURBO_LAYER_ADAPTIVE=7
set TURBO_ASYNC_PIPELINE=1

set CMD_RUN=llama-cpp-turboquant\build\bin\Release\llama-cli.exe -m "!MODEL_FILE!" -ngl !NGL! -c !CTX! -b !BATCH_SIZE! --ubatch-size !UBATCH_SIZE! --temp 0.4
if not "!EXTRA_POLICY!"=="" set CMD_RUN=!CMD_RUN! !EXTRA_POLICY!
set CMD_RUN=!CMD_RUN! --cache-type-k !CACHE_TYPE_K! --cache-type-v !CACHE_TYPE_V! --turbo-async --sparse-v-threshold -1

if "%model_choice%"=="6" (
    set CMD_RUN=!CMD_RUN! -p "Can you explain how we can compress the memory of an artificial intelligence model with a very simple story like a children's fairy tale?"
) else (
    if not "!CHAT_TEMPLATE!"=="" (
        set CMD_RUN=!CMD_RUN! -cnv --chat-template !CHAT_TEMPLATE! -sys "!SYSTEM_PROMPT!"
    ) else (
        set CMD_RUN=!CMD_RUN! -cnv -sys "!SYSTEM_PROMPT!"
    )
)

!CMD_RUN! -n 300

echo -----------------------------------------------
echo ^>>> Demo completed!
pause
