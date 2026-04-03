#include "llama-repack.h"
#include "llama-model.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "llama-impl.h"
#include "llama-llmtuning.h"
#include <fstream>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <map>
#include <vector>
#include <string>

// TQR Format:
// [Magic: 4][Version: 4][Count: 4]
// [NameLen: 4][Name: string][Offset: 8][Size: 8] ...
// [Data...]

static void write_string(std::ostream & os, const std::string & s) {
    uint32_t len = s.length();
    os.write((char*)&len, 4);
    os.write(s.c_str(), len);
}

static std::string read_string(const uint8_t *& ptr) {
    uint32_t len;
    memcpy(&len, ptr, 4);
    ptr += 4;
    std::string s((const char*)ptr, len);
    ptr += len;
    return s;
}

bool llama_model_repack_save(struct llama_model * model, const char * filename) {
    std::ofstream os(filename, std::ios::binary);
    if (!os) return false;

    os.write("TQRP", 4);
    uint32_t version = 1;
    os.write((char*)&version, 4);

    std::vector<std::pair<std::string, ggml_tensor*>> targets;

    for (int i = 0; i < (int)model->layers.size(); ++i) {
        auto & l = model->layers[i];
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "blk.%d.", i);

        auto add_if_repacked = [&](const char * suffix, ggml_tensor * t) {
            if (!t || !t->data) return;
            targets.push_back({std::string(prefix) + suffix, t});
        };

        add_if_repacked("attn_q.weight", l.wq);
        add_if_repacked("attn_k.weight", l.wk);
        add_if_repacked("attn_v.weight", l.wv);
        add_if_repacked("attn_output.weight", l.wo);
        add_if_repacked("ffn_gate.weight", l.ffn_gate);
        add_if_repacked("ffn_up.weight", l.ffn_up);
        add_if_repacked("ffn_down.weight", l.ffn_down);
    }

    uint32_t count = targets.size();
    os.write((char*)&count, 4);

    long header_end_pos = os.tellp();
    for (auto & p : targets) {
        write_string(os, p.first);
        uint64_t placeholder = 0;
        os.write((char*)&placeholder, 8); // Offset
        os.write((char*)&placeholder, 8); // Size
    }

    std::vector<uint64_t> offsets;
    std::vector<uint64_t> sizes;
    for (auto & p : targets) {
        offsets.push_back(os.tellp());
        uint64_t sz = ggml_nbytes(p.second);
        sizes.push_back(sz);
        os.write((char*)p.second->data, sz);
    }

    os.seekp(header_end_pos);
    for (size_t i = 0; i < targets.size(); ++i) {
        write_string(os, targets[i].first);
        os.write((char*)&offsets[i], 8);
        os.write((char*)&sizes[i], 8);
    }

    return true;
}

bool llama_model_repack_load(struct llama_model * model, const char * filename) {
    int fd = open(filename, O_RDONLY);
    if (fd < 0) return false;

    struct stat st;
    fstat(fd, &st);
    uint8_t * addr = (uint8_t*)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        close(fd);
        return false;
    }

    if (memcmp(addr, "TQRP", 4) != 0) {
        munmap(addr, st.st_size);
        close(fd);
        return false;
    }

    const uint8_t * ptr = addr + 8;
    uint32_t count;
    memcpy(&count, ptr, 4);
    ptr += 4;

    std::map<std::string, std::pair<uint64_t, uint64_t>> index;
    for (uint32_t i = 0; i < count; ++i) {
        std::string name = read_string(ptr);
        uint64_t off, sz;
        memcpy(&off, ptr, 8); ptr += 8;
        memcpy(&sz, ptr, 8); ptr += 8;
        index[name] = {off, sz};
    }

    for (int i = 0; i < (int)model->layers.size(); ++i) {
        auto & l = model->layers[i];
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "blk.%d.", i);

        auto apply_repack = [&](const char * suffix, ggml_tensor * t) {
            if (!t) return;
            std::string name = std::string(prefix) + suffix;
            if (index.count(name)) {
                void * old_data = t->data;
                size_t size = ggml_nbytes(t);
                
                t->data = addr + index[name].first;
                
                if (old_data) {
                    llama_unload_address(old_data, size);
                }
            }
        };

        apply_repack("attn_q.weight", l.wq);
        apply_repack("attn_k.weight", l.wk);
        apply_repack("attn_v.weight", l.wv);
        apply_repack("attn_output.weight", l.wo);
        apply_repack("ffn_gate.weight", l.ffn_gate);
        apply_repack("ffn_up.weight", l.ffn_up);
        apply_repack("ffn_down.weight", l.ffn_down);
    }

    return true;
}

void * llama_mmap_file(const char * filename, size_t * size) {
    int fd = open(filename, O_RDONLY);
    if (fd < 0) return nullptr;

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return nullptr;
    }

    void * addr = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (addr == MAP_FAILED) return nullptr;
    if (size) *size = st.st_size;

    return addr;
}
