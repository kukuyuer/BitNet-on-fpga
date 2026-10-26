#include "loader.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <fstream>
#include <vector>
#include <cstring>
#include <iostream>
#include <cmath>
#include "ggml.h"

static inline uint32_t read_u32(const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline uint64_t read_u64(const uint8_t* p) { uint64_t v; memcpy(&v, p, 8); return v; }

V37Loader::V37Loader(v37_engine_t* e) : engine(e), mmap_addr(nullptr), mmap_size(0), is_v37p(false), vocab_gguf_offset(0), vocab_gguf_size(0) {}
V37Loader::~V37Loader() { if(mmap_addr) munmap(mmap_addr, mmap_size); }

bool V37Loader::load_model(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat st; fstat(fd, &st);
    mmap_size = st.st_size;
    mmap_addr = mmap(NULL, mmap_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (mmap_addr == MAP_FAILED || !mmap_addr) {
        mmap_addr = nullptr;
        return false;
    }
    madvise(mmap_addr, mmap_size, MADV_WILLNEED);

    uint8_t* ptr = (uint8_t*)mmap_addr;
    is_v37p = false;
    vocab_gguf_offset = 0;
    vocab_gguf_size = 0;

    if (memcmp(ptr, "V37P", 4) == 0) {
        is_v37p = true;
        uint32_t v37p_ver = read_u32(ptr + 4);
        uint32_t v37p_ntensors = read_u32(ptr + 8);
        if (v37p_ver >= 4) {
            vocab_gguf_offset = read_u64(ptr + 28);
            vocab_gguf_size = read_u64(ptr + 36);
        }

        // Parse V37P TOC (starts at offset 64 for v3/v4)
        uint8_t* toc = (v37p_ver >= 3) ? (ptr + 64) : (ptr + 20);
        for (uint32_t i = 0; i < v37p_ntensors; i++) {
            char name_buf[65] = {0};
            memcpy(name_buf, toc + i * 80, 64);
            name_buf[64] = '\0';
            std::string t_name(name_buf);
            uint64_t offset = read_u64(toc + i * 80 + 64);
            uint64_t sz = read_u64(toc + i * 80 + 72);

            V37Tensor t;
            t.offset = offset;
            t.size = sz;
            t.elements = sz; // default
            t.type = 1; // default F16 for aux
            t.n_dims = 2;
            t.ne[0] = 1; t.ne[1] = 1; t.ne[2] = 1; t.ne[3] = 1;
            tensors[t_name] = t;
        }

        // If embedded GGUF vocab metadata exists, parse model hyper-parameters and vocab from it
        if (vocab_gguf_offset > 0 && vocab_gguf_size > 0 && vocab_gguf_offset + vocab_gguf_size <= mmap_size) {
            printf("[V37-LOADER] V37P container loaded (%u tensors, embedded GGUF vocab at offset 0x%lx, size %lu KB)\n",
                   v37p_ntensors, (unsigned long)vocab_gguf_offset, (unsigned long)(vocab_gguf_size / 1024));
            ptr = (uint8_t*)mmap_addr + vocab_gguf_offset;
        } else {
            printf("[V37-LOADER] V37P container loaded (%u tensors), no embedded GGUF metadata.\n", v37p_ntensors);
            return true;
        }
    }

    if (memcmp(ptr, "GGUF", 4) != 0) {
        if (is_v37p) return true;
        return false;
    }
    ptr += 4;
    uint32_t version = read_u32(ptr); ptr += 4;
    uint64_t n_tensors = read_u64(ptr); ptr += 8;
    uint64_t n_kv = read_u64(ptr); ptr += 8;

    auto read_string = [&](uint8_t*& p) -> std::string {
        uint64_t len = read_u64(p); p += 8;
        std::string s((char*)p, len);
        p += len;
        return s;
    };

    auto skip_value = [&](uint8_t*& p, int type, auto& self) -> void {
        switch (type) {
            case 0: case 1: case 7: p += 1; break;
            case 2: case 3: p += 2; break;
            case 4: case 5: case 6: p += 4; break;
            case 10: case 11: case 12: p += 8; break;
            case 8: {
                uint64_t len = read_u64(p); p += 8 + len;
                break;
            }
            case 9: {
                uint32_t atype = read_u32(p); p += 4;
                uint64_t alen = read_u64(p); p += 8;
                for (uint64_t i = 0; i < alen; i++) self(p, atype, self);
                break;
            }
        }
    };

    size_t alignment = 32;
    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = read_string(ptr);
        uint32_t type = read_u32(ptr); ptr += 4;
        if (key == "general.alignment") {
            alignment = read_u32(ptr);
            ptr += 4;
            continue;
        }
        auto read_int_val = [&](uint8_t*& p, uint32_t val_type) -> int64_t {
            if (val_type == 4) { int32_t v; memcpy(&v, p, 4); p += 4; return v; }
            if (val_type == 5) { uint32_t v; memcpy(&v, p, 4); p += 4; return v; }
            if (val_type == 10) { int64_t v; memcpy(&v, p, 8); p += 8; return v; }
            if (val_type == 11) { uint64_t v; memcpy(&v, p, 8); p += 8; return v; }
            return -1;
        };
        if (key.find(".embedding_length") != std::string::npos) {
            int64_t v = read_int_val(ptr, type);
            if (v > 0) config.n_embd = (int)v;
            else skip_value(ptr, type, skip_value);
        } else if (key.find(".block_count") != std::string::npos) {
            int64_t v = read_int_val(ptr, type);
            if (v > 0) config.n_layers = (int)v;
            else skip_value(ptr, type, skip_value);
        } else if (key.find(".feed_forward_length") != std::string::npos) {
            int64_t v = read_int_val(ptr, type);
            if (v > 0) config.n_ffn = (int)v;
            else skip_value(ptr, type, skip_value);
        } else if (key.find(".attention.head_count_kv") != std::string::npos) {
            int64_t v = read_int_val(ptr, type);
            if (v > 0) config.n_kv_heads = (int)v;
            else skip_value(ptr, type, skip_value);
        } else if (key.find(".attention.head_count") != std::string::npos) {
            int64_t v = read_int_val(ptr, type);
            if (v > 0) config.n_heads = (int)v;
            else skip_value(ptr, type, skip_value);
        } else if (key.find(".attention.key_length") != std::string::npos || key.find(".rope.dimension_count") != std::string::npos) {
            int64_t v = read_int_val(ptr, type);
            if (v > 0) config.head_dim = (int)v;
            else skip_value(ptr, type, skip_value);
        } else if (key.find(".context_length") != std::string::npos) {
            int64_t v = read_int_val(ptr, type);
            if (v > 0) config.max_seq_len = (int)std::min((int64_t)4096, v);
            else skip_value(ptr, type, skip_value);
        } else if (key.find(".rope.freq_base") != std::string::npos) {
            if (type == 6) {
                float fv; memcpy(&fv, ptr, 4); ptr += 4;
                config.rope_freq_base = fv;
            } else {
                skip_value(ptr, type, skip_value);
            }
        } else if (key.find(".attention.layer_norm_rms_epsilon") != std::string::npos) {
            if (type == 6) {
                float fv; memcpy(&fv, ptr, 4); ptr += 4;
                config.rms_norm_eps = fv;
            } else {
                skip_value(ptr, type, skip_value);
            }
        } else {
            skip_value(ptr, type, skip_value);
        }
    }

    if (n_tensors > 0) {
        for (uint64_t i = 0; i < n_tensors; i++) {
            std::string name = read_string(ptr);
            uint32_t n_dims = read_u32(ptr); ptr += 4;
            uint64_t elements = 1;
            for (uint32_t d = 0; d < n_dims; d++) {
                elements *= read_u64(ptr); ptr += 8;
            }
            uint32_t type = read_u32(ptr); ptr += 4;
            uint64_t offset = read_u64(ptr); ptr += 8;
            
            V37Tensor t;
            t.offset = offset;
            t.elements = elements;
            t.type = type;
            t.n_dims = n_dims;
            t.ne[0] = 1; t.ne[1] = 1; t.ne[2] = 1; t.ne[3] = 1;
            
            uint8_t* dim_ptr = ptr - (8 * n_dims + 12);
            for (uint32_t d = 0; d < n_dims; d++) {
                t.ne[d] = read_u64(dim_ptr); dim_ptr += 8;
            }

            if (type == 36 || type == 14) { // BitNet I2_S or GGML I2_S (2 bits/weight)
                t.size = elements / 4 + 32;
            } else if (type == 10 || type == 11 || type == 12 || type == 13 || type == 15 || type == 16 || type == 17 || type == 18) {
                t.size = elements / 2;
            } else if (type == 1) { 
                t.size = elements * 2;
            } else if (type == 0) {
                t.size = elements * 4;
            } else {
                t.size = elements; 
            }
            tensors[name] = t;
        }

        size_t data_offset = ((ptr - (uint8_t*)mmap_addr) + alignment - 1) & ~(alignment - 1);
        for (auto& pair : tensors) {
            pair.second.offset += data_offset;
        }
    }

    if (is_v37p) {
        for (auto& pair : tensors) {
            const std::string& name = pair.first;
            V37Tensor& t = pair.second;
            if (name.find("norm") != std::string::npos) {
                if (t.size % 4 == 0) {
                    t.type = 0; // F32
                    t.elements = t.size / 4;
                } else if (t.size % 2 == 0) {
                    t.type = 1; // F16
                    t.elements = t.size / 2;
                }
            } else if (name == "token_embd.weight" || name == "output.weight") {
                uint64_t expected_elems = (uint64_t)config.n_embd * config.vocab_size;
                if (t.size == expected_elems * 2) {
                    t.type = 1; // F16
                    t.elements = expected_elems;
                } else if (t.size == expected_elems * 4) {
                    t.type = 0; // F32
                    t.elements = expected_elems;
                }
                t.ne[0] = config.n_embd;
                t.ne[1] = config.vocab_size;
            } else if (name.find("hw.") == std::string::npos && name.find("fused_") == std::string::npos && name.find(".weight") != std::string::npos) {
                t.type = 36; // GGML_TYPE_I2_S
                if (name.find(".attn_q.weight") != std::string::npos) {
                    t.ne[0] = config.n_embd;
                    t.ne[1] = config.n_heads * config.head_dim;
                } else if (name.find(".attn_k.weight") != std::string::npos || name.find(".attn_v.weight") != std::string::npos) {
                    t.ne[0] = config.n_embd;
                    t.ne[1] = config.n_kv_heads * config.head_dim;
                } else if (name.find(".ffn_gate.weight") != std::string::npos || name.find(".ffn_up.weight") != std::string::npos) {
                    t.ne[0] = config.n_embd;
                    t.ne[1] = config.n_ffn;
                } else if (name.find(".ffn_down.weight") != std::string::npos) {
                    t.ne[0] = config.n_ffn;
                    t.ne[1] = config.n_embd;
                } else if (name.find(".attn_output.weight") != std::string::npos) {
                    t.ne[0] = config.n_embd;
                    t.ne[1] = config.n_embd;
                }
                t.elements = t.ne[0] * t.ne[1];
            }
        }
    }

    printf("[V37-LOADER] 自定义物理扫描器完成。发现 %lu 个张量。\n", tensors.size());
    infer_config_from_metadata_and_tensors();
    return true;
}

void V37Loader::infer_config_from_metadata_and_tensors() {
    int max_layer = -1;
    for (const auto& pair : tensors) {
        const std::string& name = pair.first;
        const V37Tensor& t = pair.second;
        if (name.rfind("blk.", 0) == 0) {
            size_t dot_pos = name.find('.', 4);
            if (dot_pos != std::string::npos) {
                int layer_idx = std::atoi(name.substr(4, dot_pos - 4).c_str());
                if (layer_idx > max_layer) max_layer = layer_idx;
            }
        }
        if (name == "token_embd.weight" || name == "output.weight") {
            if (config.n_embd <= 0 && t.ne[0] > 1) config.n_embd = (int)t.ne[0];
            if (config.vocab_size <= 0 && t.ne[1] > 1) config.vocab_size = (int)t.ne[1];
            if (config.n_embd > 0 && config.vocab_size <= 0 && t.size > 0) {
                int bytes_per_elem = (t.type == 0) ? 4 : 2;
                int inferred_vocab = (int)(t.size / (config.n_embd * bytes_per_elem));
                if (inferred_vocab > 1000) config.vocab_size = inferred_vocab;
            }
        } else if (name == "blk.0.attn_q.weight") {
            if (config.n_embd <= 0 && t.ne[0] > 1) config.n_embd = (int)t.ne[0];
            if (config.n_heads > 0 && config.head_dim <= 0 && t.ne[1] > 1) {
                config.head_dim = (int)(t.ne[1] / config.n_heads);
            }
        } else if (name == "blk.0.attn_k.weight") {
            if (config.head_dim > 0 && config.n_kv_heads <= 0 && t.ne[1] > 1) {
                config.n_kv_heads = (int)(t.ne[1] / config.head_dim);
            }
        } else if (name == "blk.0.ffn_gate.weight" || name == "blk.0.ffn_up.weight") {
            if (config.n_ffn <= 0 && t.ne[1] > 1) config.n_ffn = (int)t.ne[1];
        }
    }
    if (config.n_layers <= 0 && max_layer >= 0) {
        config.n_layers = max_layer + 1;
    }

    auto it_q = tensors.find("blk.0.attn_q.weight");
    auto it_k = tensors.find("blk.0.attn_k.weight");
    config.has_sub_norm = (tensors.find("blk.0.attn_sub_norm.weight") != tensors.end());

    // If head_count / head_dim missing from GGUF metadata, deduce from physical tensor shapes
    if (it_q != tensors.end() && it_q->second.ne[1] > 1) {
        int q_ne1 = (int)it_q->second.ne[1];
        if (config.n_heads <= 0) {
            if (q_ne1 == 2560 && config.has_sub_norm) { // BitNet-2B
                config.n_heads = 20;
                config.head_dim = 128;
                config.n_kv_heads = 5;
            } else if (q_ne1 == 2048 && config.n_embd == 2048) { // Falcon3-1B
                config.n_heads = 8;
                config.head_dim = 256;
                config.n_kv_heads = 4;
            } else if (q_ne1 == 3072 && config.n_embd == 3072) { // Falcon3-3B / 7B
                config.n_heads = 12;
                config.head_dim = 256;
                config.n_kv_heads = 4;
            } else if (q_ne1 == 4096 && config.n_embd == 4096) { // Llama3-8B
                config.n_heads = 32;
                config.head_dim = 128;
                config.n_kv_heads = 8;
            } else if (q_ne1 == 1024 && config.n_embd == 640) { // Gemma3-270M
                config.n_heads = 4;
                config.head_dim = 256;
                config.n_kv_heads = 1;
            } else if (q_ne1 == 2048 && config.n_embd == 1024) { // Qwen3-0.6B
                config.n_heads = 16;
                config.head_dim = 128;
                config.n_kv_heads = 8;
            } else {
                config.head_dim = (q_ne1 % 256 == 0 && q_ne1 >= 2048) ? 256 : 128;
                config.n_heads = q_ne1 / config.head_dim;
            }
        }
    }

    if (config.n_heads <= 0 && config.head_dim > 0 && config.n_embd > 0) {
        config.n_heads = config.n_embd / config.head_dim;
    }
    if (config.head_dim <= 0 && config.n_heads > 0 && it_q != tensors.end() && it_q->second.ne[1] > 1) {
        config.head_dim = (int)(it_q->second.ne[1] / config.n_heads);
    }
    if (config.head_dim <= 0 && config.n_heads > 0 && config.n_embd > 0) {
        config.head_dim = config.n_embd / config.n_heads;
    }
    if (config.n_kv_heads <= 0 && it_k != tensors.end() && config.head_dim > 0 && it_k->second.ne[1] > 1) {
        config.n_kv_heads = (int)(it_k->second.ne[1] / config.head_dim);
    }
    if (config.n_kv_heads <= 0 && config.n_heads > 0) {
        config.n_kv_heads = config.n_heads;
    }
    if (config.max_seq_len <= 0) {
        config.max_seq_len = 4096;
    }
    if (config.vocab_size <= 0) {
        config.vocab_size = 128256;
    }
    config.has_sub_norm = (tensors.find("blk.0.attn_sub_norm.weight") != tensors.end());
    config.use_silu = !config.has_sub_norm;
    if (getenv("USE_SILU")) {
        config.use_silu = (atoi(getenv("USE_SILU")) == 1);
    }
    if (config.rms_norm_eps <= 0.0f) {
        config.rms_norm_eps = config.has_sub_norm ? 1e-5f : 1e-6f;
    }
    if (config.rope_freq_base <= 0.0f) {
        config.rope_freq_base = config.has_sub_norm ? 500000.0f : 1000042.0f;
    }
    config.print();
}

const V37Tensor* V37Loader::get_tensor_info(const std::string& name) const {
    auto it = tensors.find(name);
    if (it != tensors.end()) return &it->second;
    return nullptr;
}

void* V37Loader::get_tensor_data(const std::string& name) {
    if (tensors.find(name) == tensors.end()) return nullptr;
    return (uint8_t*)mmap_addr + tensors[name].offset;
}

uint32_t V37Loader::get_tensor_type(const std::string& name) {
    if (tensors.find(name) == tensors.end()) return 0xFFFFFFFF;
    return tensors[name].type;
}

const float* V37Loader::get_i2_scales(const std::string& name) {
    if (tensors.find(name) != tensors.end()) {
        V37Tensor& t = tensors[name];
        if (t.type == 36) { // GGML_TYPE_I2_S format embeds scale at the end
            if (scale_cache.find(name) != scale_cache.end()) {
                return scale_cache[name].data();
            }
            uint32_t rows = (t.n_dims > 1) ? t.ne[1] : 1;
            uint64_t packed_bytes = t.elements / 4;
            
            std::vector<float> fp32_scales(rows, 1.0f);
            float* raw_scales = (float*)((uint8_t*)mmap_addr + t.offset + packed_bytes);
            float single_scale = raw_scales[0];
            for(uint32_t i=0; i<rows; i++) {
                fp32_scales[i] = single_scale;
            }
            scale_cache[name] = fp32_scales;
            return scale_cache[name].data();
        }
    }
    // Fallback for older split format or clean Format v4 scale tensors
    std::string scale_name = name + "_scale";
    if (tensors.find(scale_name) != tensors.end()) {
        if (scale_cache.find(name) != scale_cache.end()) {
            return scale_cache[name].data();
        }
        V37Tensor& t = tensors[scale_name];
        float single_scale = *(const float*)((uint8_t*)mmap_addr + t.offset);

        uint32_t rows = 65536;
        if (tensors.find(name) != tensors.end() && tensors[name].n_dims > 1) {
            rows = tensors[name].ne[1];
        } else if (name.find("ffn_gate") != std::string::npos || name.find("ffn_up") != std::string::npos) {
            rows = config.n_ffn;
        } else if (name.find("ffn_down") != std::string::npos || name.find("attn_output") != std::string::npos) {
            rows = config.n_embd;
        } else if (name.find("attn_q") != std::string::npos) {
            rows = (config.n_heads > 0 && config.head_dim > 0) ? (config.n_heads * config.head_dim) : config.n_embd;
        } else if (name.find("attn_k") != std::string::npos || name.find("attn_v") != std::string::npos) {
            rows = (config.n_kv_heads > 0 && config.head_dim > 0) ? (config.n_kv_heads * config.head_dim) : config.n_embd;
        }
        if (rows < 65536) rows = 65536;
        std::vector<float> fp32_scales(rows, single_scale);
        scale_cache[name] = fp32_scales;
        return scale_cache[name].data();
    }
    return nullptr;
}

float* V37Loader::get_embd(int token) {
    if (tensors.find("token_embd.weight") == tensors.end()) return nullptr;
    int n_embd = config.n_embd;
    embd_cache.resize(n_embd);
    V37Tensor& t = tensors["token_embd.weight"];
    
    if (t.type == 0) { // GGML_TYPE_F32
        float* f32_data = (float*)((uint8_t*)mmap_addr + t.offset + (uint64_t)token * n_embd * 4);
        for (int i = 0; i < n_embd; i++) {
            embd_cache[i] = f32_data[i];
        }
    } else if (t.type == 1) { // GGML_TYPE_F16
        uint16_t* fp16_data = (uint16_t*)((uint8_t*)mmap_addr + t.offset + (uint64_t)token * n_embd * 2);
        ggml_fp16_to_fp32_row(fp16_data, embd_cache.data(), n_embd);
    } else if (t.type == 11) { // GGML_TYPE_Q3_K
        const struct ggml_type_traits * traits = ggml_get_type_traits((enum ggml_type)11);
        if (traits && traits->to_float) {
            int block_size_bytes = 110; // Q3_K block size
            uint8_t* raw_data = (uint8_t*)mmap_addr + t.offset + (uint64_t)token * (n_embd / 256) * block_size_bytes;
            traits->to_float(raw_data, embd_cache.data(), n_embd);
        } else {
            printf("[ERROR] to_float for GGML_TYPE_Q3_K not found!\n");
        }
    } else {
        printf("[ERROR] Unsupported token_embd.weight type: %d\n", t.type);
    }
    return embd_cache.data();
}
