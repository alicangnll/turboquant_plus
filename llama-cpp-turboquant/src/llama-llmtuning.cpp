#include "llama-llmtuning.h"
#include "llama-repack.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-kv-cache.h"
#include <dirent.h>
#include "../ggml/src/ggml-quants.h"

#include <sys/mman.h>
#include <unistd.h>
#include <cstring>

// Helper to unload a single tensor's data from RAM
static void llama_unload_tensor(struct ggml_tensor * t) {
    if (!t || !t->data) return;

    // We use madvise(MADV_DONTNEED) to tell the OS that we don't need this memory right now.
    // The OS will reclaim the physical RAM pages, but the virtual address space remains valid.
    // Next time we access it, the OS will page it back in from disk.
    const size_t size = ggml_nbytes(t);
    const size_t page_size = sysconf(_SC_PAGESIZE);
    
    // Align to page boundaries for madvise
    void * addr = (void *)((uintptr_t)t->data & ~(page_size - 1));
    size_t length = size + ((uintptr_t)t->data & (page_size - 1));
    length = (length + page_size - 1) & ~(page_size - 1);

    if (madvise(addr, length, MADV_DONTNEED) != 0) {
        // Log error only in debug mode
    }
}

void llama_unload_address(void * addr, size_t size) {
    if (!addr || size == 0) return;

    const size_t page_size = sysconf(_SC_PAGESIZE);
    
    // Align to page boundaries for madvise
    void * aligned_addr = (void *)((uintptr_t)addr & ~(page_size - 1));
    size_t length = size + ((uintptr_t)addr & (page_size - 1));
    length = (length + page_size - 1) & ~(page_size - 1);

    if (madvise(aligned_addr, length, MADV_DONTNEED) != 0) {
        // Log error only in debug mode
    }
}

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

void llama_layer_prefetcher::unload(int il) {
    if (il < 0 || il >= (int)model.layers.size()) return;

    // Skip unload if layer is on GPU
    if (model.dev_layer(il) && strcmp(ggml_backend_dev_name(model.dev_layer(il)), "CPU") != 0) {
        return;
    }

    const llama_layer & layer = model.layers[il];
    
    LLAMA_LOG_INFO("%s: unloading layer %d from physical RAM (Active Sharding)\n", __func__, il);

    // Unload all major tensors in the layer
    llama_unload_tensor(layer.wq);
    llama_unload_tensor(layer.wk);
    llama_unload_tensor(layer.wv);
    llama_unload_tensor(layer.wo);
    llama_unload_tensor(layer.wqkv);
    
    llama_unload_tensor(layer.ffn_gate);
    llama_unload_tensor(layer.ffn_down);
    llama_unload_tensor(layer.ffn_up);

    llama_unload_tensor(layer.attn_norm);
    llama_unload_tensor(layer.ffn_norm);
}

void llama_layer_prefetcher::unload_all() const {
    const int n_layer = (int)model.layers.size();
    LLAMA_LOG_INFO("%s: Cold Boot: Evacuating %d transformer layers to minimize initial RAM footprint...\n", __func__, n_layer);
    for (int il = 0; il < n_layer; ++il) {
        // cast away const to call member function (or make unload const)
        const_cast<llama_layer_prefetcher*>(this)->unload(il);
    }
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

        if (il < 0 || il >= (int)model.layers.size()) {
            continue;
        }

        const llama_layer & layer = model.layers[il];
        
        // Touch weights to page them in (Thread 1)
        LLAMA_LOG_DEBUG("%s: prefetching layer %d\n", __func__, il);
        if (layer.wq)   { (void)*(const volatile char *)layer.wq->data; }
        if (layer.wk)   { (void)*(const volatile char *)layer.wk->data; }
        if (layer.wv)   { (void)*(const volatile char *)layer.wv->data; }
        if (layer.wo)   { (void)*(const volatile char *)layer.wo->data; }
        if (layer.wqkv) { (void)*(const volatile char *)layer.wqkv->data; }
        
        if (layer.ffn_gate) { (void)*(const volatile char *)layer.ffn_gate->data; }
        if (layer.ffn_down) { (void)*(const volatile char *)layer.ffn_down->data; }
        if (layer.ffn_up)   { (void)*(const volatile char *)layer.ffn_up->data; }

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

llama_kv_compress_worker::llama_kv_compress_worker(const struct llama_context & /*ctx*/) {
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

        if (j.k->type == GGML_TYPE_TURBO2_0 || j.k->type == GGML_TYPE_TURBO3_0 || 
            j.k->type == GGML_TYPE_TURBO4_0 || j.k->type == GGML_TYPE_Q8_0) {
            std::unique_lock<std::mutex> lock(mtx);
            completed_layer = j.il;
            cv.notify_all();
            continue;
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

    // TQR (TurboQuant Repack) Hijacking Status check
    char model_desc[256];
    llama_model_desc(&ctx.get_model(), model_desc, sizeof(model_desc));

    // Cleanup stale TQR files in the current directory
    std::string current_tqr_name = model_desc;
    for (char & c : current_tqr_name) {
        if (!isalnum(c)) c = '_';
    }
    std::string current_tqr_filename = current_tqr_name + ".tqr";

    DIR * dir = opendir(".");
    if (dir) {
        struct dirent * entry;
        while ((entry = readdir(dir)) != NULL) {
            std::string fname = entry->d_name;
            if (fname.size() > 4 && fname.substr(fname.size() - 4) == ".tqr") {
                // If it's a TQR file but not for THIS model, delete it
                if (fname != current_tqr_filename) {
                    LLAMA_LOG_INFO("%s: [TURBO] Cleaning up stale TQR: %s\n", __func__, fname.c_str());
                    unlink(fname.c_str());
                }
            }
        }
        closedir(dir);
    }

    if (ggml_cpu_repack_is_hijacked()) {
        LLAMA_LOG_INFO("%s: [TURBO] Zero-Allocation Hijacking active. Using pre-mapped weights from SSD.\n", __func__);
    } else {
        LLAMA_LOG_INFO("%s: Hijacking not active. Checking if we should cache current optimized weights...\n", __func__);
        
        // Try to save the current repack if it doesn't exist
        if (llama_model_repack_save(const_cast<struct llama_model *>(&ctx.get_model()), current_tqr_filename.c_str())) {
             LLAMA_LOG_INFO("%s: [TURBO] Optimization cached to %s for future instant boots.\n", __func__, current_tqr_filename.c_str());
        }
    }

    // Initial Footprint Minimization: Evacuate weights to SSD
    prefetcher->unload_all();
}

void llama_tuning_session::step(int il, struct ggml_tensor * k, struct ggml_tensor * v) const {
    // Stage 1: LLMTuning Sharding
    // Unload the previous layer (il-1) to free RAM immediately
    if (il > 0) {
        prefetcher->unload(il - 1);
        if (il % 8 == 0) {
            LLAMA_LOG_INFO("%s: LLMTuning Active Sharding... (Layer %d)\n", __func__, il);
        }
    }

    // Prefetch the next layer (il+1) to be ready for the next step
    if (il + 1 < (int)ctx.get_model().layers.size()) {
        prefetcher->prefetch(il + 1);
    }
    
    // Stage 3: TurboQuant KV Compression
    if (compressor) {
        compressor->compress_async(il, k, v);
    }
}

void llama_tuning_session::prefetch_all() const {
    const int n_layer = (int)ctx.get_model().layers.size();
    for (int il = 0; il < n_layer; ++il) {
        prefetcher->prefetch(il);
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
