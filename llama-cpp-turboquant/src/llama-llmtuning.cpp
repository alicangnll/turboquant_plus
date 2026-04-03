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

void llama_kv_compress_worker::worker_loop() {
    while (true) {
        job j;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [this] { return !jobs.empty() || should_exit; });
            if (should_exit) {
                break;
            }
            j = jobs.front();
            jobs.pop();
        }

        // Stage 3: TurboQuant CPU compression (async, Thread 3)
        // This is where we run the quantization logic from turbo-quant.c
        
        const int64_t ne0 = j.k->ne[0]; // head_dim
        const int64_t ne1 = j.k->ne[1]; // n_heads * n_seq
        
        // We compress row by row. Each row is one head's KV data for one token.
        // head_dim is usually 128.
        
        std::vector<float> tmp_k(ne0);
        std::vector<float> tmp_v(ne0);
        
        // Determine types and quantization functions
        // For now, we hardcode compression to TURBO2_0 as specified in the LLMTuning session
        // In a real implementation, we would check environment variables or context params.
        
        for (int64_t i = 0; i < ne1; ++i) {
            // 1. Dequantize K to float
            ggml_get_type_traits(j.k->type)->to_float(
                (const char *)j.k->data + i * ggml_row_size(j.k->type, ne0),
                tmp_k.data(),
                ne0
            );
            
            // 2. Quantize K to TURBO2 (if that's our target)
            // Note: Since we are doing this in-place or replacing the buffer, 
            // we must be careful about memory layout. 
            // The LLMTuning logic usually assumes a hybrid KV cache where the 
            // "compressed" buffer is pre-allocated or we are overwriting a 
            // higher-precision one.
            
            // For this implementation, we assume the tensor 'j.k' is the target 
            // and its 'type' might be updated or it was already set to a 
            // quantized type.
            
            quantize_row_turbo2_0_ref(tmp_k.data(), (block_turbo2_0 *)((char *)j.k->data + i * ggml_row_size(GGML_TYPE_TURBO2_0, ne0)), ne0);
            
            // Repeat for V
            ggml_get_type_traits(j.v->type)->to_float(
                (const char *)j.v->data + i * ggml_row_size(j.v->type, ne0),
                tmp_v.data(),
                ne0
            );
            quantize_row_turbo2_0_ref(tmp_v.data(), (block_turbo2_0 *)((char *)j.v->data + i * ggml_row_size(GGML_TYPE_TURBO2_0, ne0)), ne0);
        }

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
        prefetcher->prefetch(il + 1); // wait for it inside the next step
    }
    
    // 2. Main compute on GPU for layer N (called by main thread)
    // ... main thread runs the graph ...

    // 3. Compress KV for layer N (async, Thread 3)
    compressor->compress_async(il, k, v);
}

void llama_tuning_session::shutdown() {
    prefetcher.reset();
    compressor.reset();
}
