#pragma once

#include "llama-kv-cache.h"
#include <memory>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <queue>

//
// llama_layer_prefetcher (Thread 1: Disk -> RAM)
//
// Asynchronously loads layer weights from disk into RAM, overlapping with GPU compute
// on the current layer.
//

class llama_layer_prefetcher {
public:
    llama_layer_prefetcher(const struct llama_model & model);
    ~llama_layer_prefetcher();

    // Start background prefetch of layer `il` weights.
    void prefetch(int il);

    // Block until layer `il` weights are fully loaded into RAM.
    void wait(int il);

private:
    const struct llama_model & model;
    std::thread worker;
    std::mutex mtx;
    std::condition_variable cv;
    int next_layer = -1;
    int ready_layer = -1;
    bool should_exit = false;

    void worker_loop();
};

//
// llama_kv_compress_worker (Thread 3: RAM -> Compressed RAM)
//
// Asynchronously compresses the KV cache for the previous layer while the GPU
// computes the current layer.
//

class llama_kv_compress_worker {
public:
    llama_kv_compress_worker(const struct llama_context & ctx);
    ~llama_kv_compress_worker();

    // Submit KV tensors (k, v) for background compression.
    // Shape: [n_heads, seq_len, head_dim]
    void compress_async(int il, struct ggml_tensor * k, struct ggml_tensor * v);

    // Block until compression for layer `il` is complete.
    void wait(int il);

    // Block until all pending compression jobs are finished.
    void wait_all();

private:
    const struct llama_context & ctx;
    std::thread worker;
    std::mutex mtx;
    std::condition_variable cv;
    
    struct job {
        int il;
        struct ggml_tensor * k;
        struct ggml_tensor * v;
    };
    std::queue<job> jobs;
    int completed_layer = -1;
    bool is_working = false;
    bool should_exit = false;

    void worker_loop();
};

//
// llama_tuning_session
//
// Orchestrates the 3-stage pipeline:
//   Thread 1: prefetch layer N+1
//   Thread 2: compute layer N (main thread)
//   Thread 3: compress KV layer N-1
//

struct llama_tuning_session {
    const struct llama_context & ctx;
    std::unique_ptr<llama_layer_prefetcher> prefetcher;
    std::unique_ptr<llama_kv_compress_worker> compressor;

    llama_tuning_session(const struct llama_context & ctx);
    
    // Core pipeline step for layer N
    void step(int il, struct ggml_tensor * k, struct ggml_tensor * v) const;

    // Trigger prefetching for ALL layers (safe to run in parallel with GPU)
    void prefetch_all() const;

    // Trigger KV compression for ALL layers (run AFTER GPU compute)
    void compress_all(const struct llama_memory_i & memory) const;

    // Block until all asynchronous stages are completed for the current batch.
    void wait_all() const;

    void shutdown();
};
