#include "llama-llmtuning.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-kv-cache.h"
#include "../ggml/src/ggml-quants.h"

//
// llama_layer_prefetcher
//

llama_layer_prefetcher::llama_layer_prefetcher(const struct llama_model & model) : model(model) {
    worker = std::thread(&llama_layer_prefetcher::worker_loop, this);
}

llama_layer_prefetcher::~llama_layer_prefetcher() {
    {
        std::unique_lock<std::mutex> lock(mtx);
        should_exit = true;
        cv.notify_one();
    }
    if (worker.joinable()) {
        worker.join();
    }
}

void llama_layer_prefetcher::prefetch(int il) {
    if (il < 0 || il >= (int)model.layers.size()) return;

    // Skip prefetch if layer is already in GPU memory (Metal)
    // Full GPU offload means we don't need CPU-based prefetching
    if (model.dev_layer(il) && strcmp(ggml_backend_dev_name(model.dev_layer(il)), "CPU") != 0) {
        return;
    }

    std::unique_lock<std::mutex> lock(mtx);
    next_layer = il;
    cv.notify_one();
}

void llama_layer_prefetcher::wait(int il) {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [this, il] { return ready_layer == il || should_exit; });
}

void llama_layer_prefetcher::worker_loop() {
    while (true) {
        int il = -1;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [this] { return next_layer != -1 || should_exit; });
            if (should_exit) {
                break;
            }
            il = next_layer;
            next_layer = -1;
        }

        // simulating prefetch for now by accessing some weights
        if (il < 0 || il >= (int)model.layers.size()) {
            LLAMA_LOG_ERROR("%s: layer index %d out of bounds (n_layers = %zu)\n", __func__, il, model.layers.size());
            continue;
        }

        const llama_layer & layer = model.layers[il];
        
        // Touch weights to page them in (Thread 1)
        LLAMA_LOG_DEBUG("%s: prefetching layer %d\n", __func__, il);
        if (layer.wq) { (void)layer.wq->data; }
        if (layer.wk) { (void)layer.wk->data; }
        if (layer.wv) { (void)layer.wv->data; }
        if (layer.wo) { (void)layer.wo->data; }

        {
            std::unique_lock<std::mutex> lock(mtx);
            ready_layer = il;
            cv.notify_all();
        }
    }
}

//
// llama_kv_compress_worker
//

llama_kv_compress_worker::llama_kv_compress_worker(const struct llama_context & ctx) : ctx(ctx) {
    worker = std::thread(&llama_kv_compress_worker::worker_loop, this);
}

llama_kv_compress_worker::~llama_kv_compress_worker() {
    {
        std::unique_lock<std::mutex> lock(mtx);
        should_exit = true;
        cv.notify_one();
    }
    if (worker.joinable()) {
        worker.join();
    }
}

void llama_kv_compress_worker::compress_async(int il, struct ggml_tensor * k, struct ggml_tensor * v) {
    std::unique_lock<std::mutex> lock(mtx);
    jobs.push({il, k, v});
    cv.notify_one();
}

void llama_kv_compress_worker::wait(int il) {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [this, il] { return completed_layer == il || should_exit; });
}

void llama_kv_compress_worker::wait_all() {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [this] { return (jobs.empty() && !is_working) || should_exit; });
}

void llama_kv_compress_worker::worker_loop() {
    while (true) {
        job j;
        {
            std::unique_lock<std::mutex> lock(mtx);
            is_working = false;
            cv.notify_all();
            cv.wait(lock, [this] { return !jobs.empty() || should_exit; });
            if (should_exit) {
                break;
            }
            j = jobs.front();
            jobs.pop();
            is_working = true;
        }

        // Stage 3: TurboQuant CPU compression (async, Thread 3)
        // This is where we run the quantization logic from turbo-quant.c
        
        // Skip if already quantized by backend (Metal handles TQ natively)
        if (j.k->type == GGML_TYPE_TURBO2_0 || j.k->type == GGML_TYPE_TURBO3_0 || 
            j.k->type == GGML_TYPE_TURBO4_0 || j.k->type == GGML_TYPE_Q8_0) {
            std::unique_lock<std::mutex> lock(mtx);
            completed_layer = j.il;
            cv.notify_all();
            continue;
        }

        // STUBBED: Do NOT modify KV cache from CPU to avoid corruption with Metal
        // Metal backend handles TQ natively. We keep this thread for orchestration.
        // We simply skip the quantization work and signal completion.

        {
            std::unique_lock<std::mutex> lock(mtx);
            completed_layer = j.il;
            cv.notify_all();
        }
    }
}

//
// llama_tuning_session
//

llama_tuning_session::llama_tuning_session(const struct llama_context & ctx) : ctx(ctx) {
    prefetcher = std::make_unique<llama_layer_prefetcher>(ctx.get_model());
    compressor = std::make_unique<llama_kv_compress_worker>(ctx);
}

void llama_tuning_session::step(int il, struct ggml_tensor * k, struct ggml_tensor * v) const {
    // 1. Kick off prefetch for next layer N+1
    if (il + 1 < (int)ctx.get_model().layers.size()) {
        prefetcher->prefetch(il + 1);
    }
    
    // 2. Main compute (Current Thread 2) for layer N is happening in Parallel
    
    // 3. Compress KV for layer N (async, Thread 3)
    if (compressor) {
        compressor->compress_async(il, k, v);
    }
}

void llama_tuning_session::prefetch_all() const {
    const int n_layer = (int)ctx.get_model().layers.size();
    for (int il = 0; il < n_layer; ++il) {
        if (il + 1 < n_layer) {
            prefetcher->prefetch(il + 1);
        }
    }
}

void llama_tuning_session::compress_all(const struct llama_memory_i & memory) const {
    const int n_layer = (int)ctx.get_model().layers.size();
    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * k = memory.get_layer_k(il);
        struct ggml_tensor * v = memory.get_layer_v(il);
        if (k && v && compressor) {
            compressor->compress_async(il, k, v);
        }
    }
}

void llama_tuning_session::wait_all() const {
    if (compressor) {
        compressor->wait_all();
    }
}

void llama_tuning_session::shutdown() {
    prefetcher.reset();
    compressor.reset();
}
