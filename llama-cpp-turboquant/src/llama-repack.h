#ifndef LLAMA_REPACK_H
#define LLAMA_REPACK_H

#include "ggml.h"
#include <string>
#include <vector>
#include <map>

struct llama_model;

// TQR (TurboQuant Repack) File Header
// Magic: "TQRP" (4 bytes)
// Version: 1 (4 bytes)
// Tensor Count: N (4 bytes)
// ---
// Repeated N times:
//   Name Length: L (4 bytes)
//   Name: String (L bytes)
//   Offset: uint64 (8 bytes)
//   Size: uint64 (8 bytes)
// ---
// Raw Data Blobs

struct llama_repack_tensor {
    std::string name;
    uint64_t offset;
    uint64_t size;
};

struct llama_repack_index {
    uint32_t version = 1;
    std::map<std::string, llama_repack_tensor> tensors;
};

// Export all current backend tensors to a .tqr file
bool llama_model_repack_save(struct llama_model * model, const char * filename);

// Map tensors in the model to the .tqr file (mmap)
bool llama_model_repack_load(struct llama_model * model, const char * filename);

// Map a TQR file into memory for hijacking
void * llama_mmap_file(const char * filename, size_t * size);

#endif // LLAMA_REPACK_H
