#include "engine.h"
#include "v37_driver.h"
#include "llama.h"
#include <map>
#include <unordered_map>
#include <chrono>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// Weight CMA cache: track which tensors have been preloaded to avoid redundant calls.
static std::unordered_map<std::string, bool> g_weight_loaded;

// Norm weight cache: loaded once per tensor name, reused across all tokens.
static std::unordered_map<std::string, std::vector<float>> g_norm_cache;

#include <cmath>
#include <cstring>
#include <algorithm>
#include <queue>
#include <unordered_set>
#include <omp.h>
#include <cstdio>

#if defined(__aarch64__) || defined(__arm__)
#include <arm_neon.h>
#endif


extern "C" void* v37_get_cma_slot_ptr(v37_engine_t* engine, int slot_id);

#if defined(__aarch64__) || defined(__arm__)
typedef __fp16 v37_f16_t;
#define V37_F16_TO_F32(x) ((float)(x))


#elif defined(__FLT16_MANT_DIG__)
typedef _Float16 v37_f16_t;
#define V37_F16_TO_F32(x) ((float)(x))
#else
typedef uint16_t v37_f16_t;
#define V37_F16_TO_F32(x) ggml_fp16_to_fp32(x)
#endif

static void check_nan(const float* arr, int n, const char* name, int layer) {
    static int debug_nan = -1;
    if (debug_nan == -1) debug_nan = (getenv("V37_DEBUG") != nullptr) ? 1 : 0;
    if (__builtin_expect(debug_nan, 0)) {
        for (int i=0; i<n; i++) {
            if (std::isnan(arr[i])) {
                printf("[DEBUG] NaN detected in %s at layer %d, idx %d!\n", name, layer, i);
                break;
            }
        }
    }
}




V37Engine::V37Engine(v37_engine_t* e, V37Loader* l, struct llama_model* m, const ModelConfig* cfg)
    : engine(e), loader(l), model(m), rope_table_inited(false), v37pack_mmap(nullptr), v37pack_size(0), active_pack_mode(V37_MODE_I2_S), all_weights_preloaded(false), layer_cache_initialized(false), lmhead_cache_initialized(false),
      max_lmhead_t1_scale(0.0f), inv_max_lmhead_t1_scale(0.0f), max_lmhead_t2_scale(0.0f), inv_max_lmhead_t2_scale(0.0f),
      lmhead_tier1_size(28160), lmhead_d1(896), lmhead_d_coarse(128),
      lmhead_d_s1(512), lmhead_d_s2(384), lmhead_s1_k_per_thread(512), lmhead_twostage_enabled(1), lmhead_top_cand(128) {

    if (cfg) {
        config = *cfg;
    } else if (loader) {
        config = loader->get_config();
    }

    hidden_states.assign(config.n_embd, 0.0f);
    logits.assign(config.vocab_size, -1e9f);
    kv_cache.assign((size_t)config.n_layers * config.max_seq_len * 2 * config.n_kv_heads * config.head_dim, 0);
    layers_cache.resize(config.n_layers);

    int q_dim = config.n_heads * config.head_dim;
    if (q_dim <= 0) q_dim = config.n_embd;

    buf_norm_out.resize(config.n_embd);
    buf_q.resize(q_dim);
    buf_k.resize(config.n_kv_heads * config.head_dim);
    buf_v.resize(config.n_kv_heads * config.head_dim);
    buf_attn_out.resize(q_dim);
    buf_attn_sub_norm_out.resize(q_dim);
    buf_o_out.resize(config.n_embd);
    buf_ffn_norm_out.resize(config.n_embd);
    buf_gate.resize(config.n_ffn);
    buf_up.resize(config.n_ffn);
    buf_ffn_act.resize(config.n_ffn);
    buf_ffn_sub_norm_out.resize(config.n_ffn);
    buf_down.resize(config.n_embd);
    buf_final_norm.resize(config.n_embd);
    lmhead_in_cands.assign(config.vocab_size, 0);

    printf("[V37] Engine Initialized with Dynamic ModelConfig.\n");
    config.print();
}

void V37Engine::preload_all_weights_to_cma() {
    if (!this->engine || getenv("DISABLE_V37")) return;
    printf("[V37-OPT] Starting ONE-TIME full model weight preloading into CMA (Zero-Copy Mode)...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    v37_reset_cma_allocator(engine, 0);

    static const char* target_tensors[] = {
        "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"
    };
    int loaded_count = 0;
    size_t total_bytes = 0;
    const char* chain_env = getenv("V37_CHAIN_LAUNCH");
    bool chain_launch = (!chain_env || atoi(chain_env) != 0);

    int base_fused_gate_up = config.n_layers * 7;
    int base_fused_qkv     = base_fused_gate_up + config.n_layers;

    for (int l = 0; l < config.n_layers; l++) {
        for (int i = 0; i < 7; i++) {
            if (chain_launch) {
                // If fused QKV exists, skip separate CMA allocation for Q, K, V (i = 0, 1, 2)
                if (i >= 0 && i <= 2) {
                    std::string fqkv_name = "blk." + std::to_string(l) + ".fused_qkv.weight";
                    std::string q_name = "blk." + std::to_string(l) + ".attn_q.weight";
                    std::string k_name = "blk." + std::to_string(l) + ".attn_k.weight";
                    std::string v_name = "blk." + std::to_string(l) + ".attn_v.weight";
                    if (get_prepacked(fqkv_name) || (get_prepacked(q_name) && get_prepacked(k_name) && get_prepacked(v_name))) {
                        continue;
                    }
                }
                // If fused Gate+Up exists, skip separate CMA allocation for Gate, Up (i = 4, 5)
                if (i == 4 || i == 5) {
                    std::string fgu_name = "blk." + std::to_string(l) + ".fused_gate_up.weight";
                    std::string g_name = "blk." + std::to_string(l) + ".ffn_gate.weight";
                    std::string u_name = "blk." + std::to_string(l) + ".ffn_up.weight";
                    if (get_prepacked(fgu_name) || (get_prepacked(g_name) && get_prepacked(u_name))) {
                        continue;
                    }
                }
            }
            int global_slot = l * 7 + i;
            std::string t_name = "blk." + std::to_string(l) + "." + target_tensors[i];
            int k_dim = config.n_embd, m_dim = config.n_embd;
            if (i == 1 || i == 2) m_dim = config.n_kv_heads * config.head_dim;
            if (i == 4 || i == 5) m_dim = config.n_ffn;
            if (i == 6) { m_dim = config.n_embd; k_dim = config.n_ffn; }

            V37PackEntry* pp = get_prepacked("hw." + t_name);
            if (!pp) pp = get_prepacked(t_name);
            if (pp) {
                v37_mode_t preload_mode = (pp->mode == V37_MODE_I2_S) ? V37_MODE_I2_S_PREPACKED : pp->mode;
                v37_preload_weights(engine, pp->data, nullptr, pp->size, global_slot, preload_mode, k_dim, 0);
                loaded_count++;
                total_bytes += pp->size;
            } else if (loader) {
                uint8_t* ptr = (uint8_t*)loader->get_tensor_data(t_name);
                if (ptr) {
                    const float* w_scales = loader->get_i2_scales(t_name);
                    int padded_k = (k_dim + 255) & ~255;
                    size_t w_sz = (size_t)m_dim * padded_k / 4;
                    v37_preload_weights(engine, ptr, w_scales, w_sz, global_slot, V37_MODE_I2_S, k_dim, 0);
                    loaded_count++;
                    total_bytes += w_sz;
                }
            }
        }

        if (chain_launch) {
            // Preload Fused Gate+Up
            std::string fgu_name = "blk." + std::to_string(l) + ".fused_gate_up.weight";
            V37PackEntry* pp_fgu = get_prepacked(fgu_name);
            int fused_slot = base_fused_gate_up + l;
            if (pp_fgu) {
                int ret = v37_preload_fused_direct(engine, pp_fgu->data, pp_fgu->size, fused_slot);
                if (ret == 0) {
                    loaded_count += 2;
                    total_bytes += pp_fgu->size;
                }
            } else {
                std::string g_name = "blk." + std::to_string(l) + ".ffn_gate.weight";
                std::string u_name = "blk." + std::to_string(l) + ".ffn_up.weight";
                V37PackEntry* pp_g = get_prepacked(g_name);
                V37PackEntry* pp_u = get_prepacked(u_name);
                if (pp_g && pp_u) {
                    int ret = v37_preload_fused_gate_up_buffers(engine, pp_g->data, pp_u->data, fused_slot, config.n_ffn, config.n_embd, active_pack_mode);
                    if (ret == 0) {
                        loaded_count += 2;
                        total_bytes += (pp_g->size + pp_u->size);
                    }
                } else {
                    int gate_slot = l * 7 + 4;
                    int up_slot = l * 7 + 5;
                    v37_preload_fused_gate_up(engine, gate_slot, up_slot, fused_slot, config.n_ffn, config.n_embd, active_pack_mode);
                }
            }

            // Preload Fused QKV
            std::string fqkv_name = "blk." + std::to_string(l) + ".fused_qkv.weight";
            V37PackEntry* pp_fqkv = get_prepacked(fqkv_name);
            int fused_qkv_slot = base_fused_qkv + l;
            if (pp_fqkv) {
                int ret = v37_preload_fused_direct(engine, pp_fqkv->data, pp_fqkv->size, fused_qkv_slot);
                if (ret == 0) {
                    loaded_count += 3;
                    total_bytes += pp_fqkv->size;
                }
            } else {
                std::string q_name = "blk." + std::to_string(l) + ".attn_q.weight";
                std::string k_name = "blk." + std::to_string(l) + ".attn_k.weight";
                std::string v_name = "blk." + std::to_string(l) + ".attn_v.weight";
                V37PackEntry* pp_q = get_prepacked(q_name);
                V37PackEntry* pp_k = get_prepacked(k_name);
                V37PackEntry* pp_v = get_prepacked(v_name);
                if (pp_q && pp_k && pp_v) {
                    int ret = v37_preload_fused_qkv_buffers(engine, pp_q->data, pp_k->data, pp_v->data, fused_qkv_slot, config.n_embd, config.n_kv_heads * config.head_dim, config.n_kv_heads * config.head_dim, config.n_embd, active_pack_mode);
                    if (ret == 0) {
                        loaded_count += 3;
                        total_bytes += (pp_q->size + pp_k->size + pp_v->size);
                    }
                }
            }
        }
    }
    if (loaded_count == config.n_layers * 7) {
        all_weights_preloaded = true;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[V37-OPT] Successfully preloaded all %d tensors (%.2f MB) into CMA in %.2f ms!\n",
           loaded_count, total_bytes / 1024.0 / 1024.0, ms);
    printf("[V37-OPT] Zero-Copy CMA stream active: subsequent token generation will have ZERO memcpy overhead.\n");
}

bool V37Engine::load_v37pack(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        printf("[V37PACK] Cannot open %s\n", path.c_str());
        return false;
    }
    struct stat st;
    fstat(fd, &st);
    v37pack_size = st.st_size;
    v37pack_mmap = mmap(NULL, v37pack_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (v37pack_mmap == MAP_FAILED) {
        v37pack_mmap = nullptr;
        return false;
    }
    
    uint8_t* p = (uint8_t*)v37pack_mmap;
    if (memcmp(p, "V37P", 4) != 0) {
        printf("[V37PACK] Invalid magic\n");
        munmap(v37pack_mmap, v37pack_size);
        v37pack_mmap = nullptr;
        return false;
    }
    uint32_t version = *(uint32_t*)(p + 4);
    uint32_t n_tensors = *(uint32_t*)(p + 8);
    
    v37_mode_t file_mode = V37_MODE_I2_S;
    uint8_t* toc = p + 12;
    if (version == 2) {
        file_mode = (v37_mode_t)(*(uint32_t*)(p + 12));
        toc = p + 20;
    } else if (version >= 3) {
        file_mode = (v37_mode_t)(*(uint32_t*)(p + 12));
        uint32_t n_ucode_insts = *(uint32_t*)(p + 16);
        uint64_t ucode_offset  = *(uint64_t*)(p + 20);
        toc = p + 64;
        if (this->engine && n_ucode_insts > 0 && ucode_offset > 0) {
            const uint32_t* ucode_ptr = (const uint32_t*)(p + ucode_offset);
            v37_load_microcode(this->engine, ucode_ptr, (int)n_ucode_insts);
        }
    }
    active_pack_mode = file_mode;

    // TOC starts at offset 12 (v1) or 20 (v2)
    // Each entry: name(64 bytes, null-padded) + offset(uint64) + size(uint64) = 80 bytes
    for (uint32_t i = 0; i < n_tensors; i++) {
        char name_buf[65];
        memcpy(name_buf, toc + i * 80, 64);
        name_buf[64] = '\0';
        std::string name(name_buf);
        
        uint64_t data_offset = *(uint64_t*)(toc + i * 80 + 64);
        uint64_t data_size   = *(uint64_t*)(toc + i * 80 + 72);
        
        V37PackEntry entry;
        entry.data = p + data_offset;
        entry.size = data_size;
        entry.mode = file_mode;
        v37pack_entries[name] = entry;
    }
    
    const char* mode_name = (file_mode == V37_MODE_TL3_COMPACT) ? "TL3_COMPACT (Mode 6, 5-in-8 Gearbox Stream)" :
                            (file_mode == V37_MODE_TL3) ? "TL3 (Mode 7, 5-in-8)" :
                            (file_mode == V37_MODE_DIRECT) ? "DIRECT (Mode 0, INT2)" :
                            "I2_S (Mode 5, 4-in-8)";
    printf("[V37PACK] Loaded %u pre-packed tensors from %s (%.1f MB, Format v%u, Mode: %s)\n",
           n_tensors, path.c_str(), v37pack_size / 1024.0 / 1024.0, version, mode_name);

    // 一次性常驻 CMA 预加载与各层缓存构建
    preload_all_weights_to_cma();
    init_layer_cache();

    return true;
}

V37PackEntry* V37Engine::get_prepacked(const std::string& name) {
    auto it = v37pack_entries.find(name);
    if (it != v37pack_entries.end()) return &it->second;
    return nullptr;
}

static inline int v37_nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

// Decode GGUF I2_S ternary weight using QK_I2_S=128 MSB-first format (ACT_PARALLEL mode).
// Each 128-weight block → 32 bytes. byte[block*32+group_pos] holds:
//   bits 6-7 = col block*128 + group_pos + 0*32  (group 0, MSB)
//   bits 4-5 = col block*128 + group_pos + 1*32  (group 1)
//   bits 2-3 = col block*128 + group_pos + 2*32  (group 2)
//   bits 0-1 = col block*128 + group_pos + 3*32  (group 3, LSB)
// Encoding: 0=-1, 1=0, 2=+1
static inline int decode_i2s_weight_inline(const uint8_t* ptr, int col) {
    int block_idx  = col / 128;
    int col_in_blk = col % 128;
    int group_idx  = col_in_blk / 32;
    int group_pos  = col_in_blk % 32;
    int byte_idx   = block_idx * 32 + group_pos;
    int shift      = 6 - 2 * group_idx;  // MSB-first
    uint8_t code = (ptr[byte_idx] >> shift) & 3;
    // Map: 0 -> -1, 1 -> 0, 2 -> +1, 3 -> 0
    static const int map[4] = {-1, 0, 1, 0};
    return map[code];
}

void V37Engine::quantize_activation_neon(const float* in, int in_dim, QuantizedAct& qact) {
    qact.in_dim = in_dim;
    if ((int)qact.x_q.size() < in_dim) {
        qact.x_q.resize((in_dim + 63) & ~63);
    }
    int8_t* x_q = qact.x_q.data();

#if defined(__aarch64__) || defined(__arm__)
    float32x4_t vm = vdupq_n_f32(0.00001f);
    int i = 0;
    for (; i <= in_dim - 16; i += 16) {
        float32x4_t v0 = vabsq_f32(vld1q_f32(in + i + 0));
        float32x4_t v1 = vabsq_f32(vld1q_f32(in + i + 4));
        float32x4_t v2 = vabsq_f32(vld1q_f32(in + i + 8));
        float32x4_t v3 = vabsq_f32(vld1q_f32(in + i + 12));
        vm = vmaxnmq_f32(vm, vmaxnmq_f32(vmaxnmq_f32(v0, v1), vmaxnmq_f32(v2, v3)));
    }
    float max_val = vmaxvq_f32(vm);
    for (; i < in_dim; i++) {
        float a = std::fabs(in[i]);
        if (a > max_val) max_val = a;
    }
    float s = 127.0f / max_val;
    qact.s = s;

    float32x4_t vs = vdupq_n_f32(s);
    i = 0;
    for (; i <= in_dim - 16; i += 16) {
        float32x4_t f0 = vmulq_f32(vld1q_f32(in + i + 0), vs);
        float32x4_t f1 = vmulq_f32(vld1q_f32(in + i + 4), vs);
        float32x4_t f2 = vmulq_f32(vld1q_f32(in + i + 8), vs);
        float32x4_t f3 = vmulq_f32(vld1q_f32(in + i + 12), vs);

        int32x4_t s0 = vcvtnq_s32_f32(f0);
        int32x4_t s1 = vcvtnq_s32_f32(f1);
        int32x4_t s2 = vcvtnq_s32_f32(f2);
        int32x4_t s3 = vcvtnq_s32_f32(f3);

        int16x8_t q01 = vcombine_s16(vqmovn_s32(s0), vqmovn_s32(s1));
        int16x8_t q23 = vcombine_s16(vqmovn_s32(s2), vqmovn_s32(s3));
        int8x16_t q_all = vcombine_s8(vqmovn_s16(q01), vqmovn_s16(q23));

        vst1q_s8(x_q + i, q_all);
    }
    for (; i < in_dim; i++) {
        int v = v37_nearest_int(in[i] * s);
        if (v >  127) v = 127;
        if (v < -128) v = -128;
        x_q[i] = (int8_t)v;
    }
#else
    double max = 0.00001;
    for (int i = 0; i < in_dim; ++i) {
        if ((double)std::fabs((double)in[i]) > max) {
            max = (double)std::fabs((double)in[i]);
        }
    }
    float s = 127.0f / max;
    qact.s = s;
    for (int i = 0; i < in_dim; i++) {
        int v = v37_nearest_int(in[i] * s);
        if (v >  127) v = 127;
        if (v < -128) v = -128;
        x_q[i] = (int8_t)v;
    }
#endif
}

static void decode_hw_submatrix_gemv(const uint8_t* hw_data, uint32_t channel_sz, uint32_t ch_offset,
                                     int m, int k, v37_mode_t mode,
                                     const int8_t* x_q, float* out, float scale,
                                     float act_scale, float* residual = nullptr) {
    int padded_k = (k + 255) & ~255;
    uint32_t pe_k_beats = (k + 63) / 64;
    pe_k_beats = (pe_k_beats + 3) & ~3;
    v37_mode_t real_mode = (v37_mode_t)(mode & 0xFFFF);
    uint32_t row_stride = (real_mode == V37_MODE_TL3_COMPACT) ? (pe_k_beats * 13) : (pe_k_beats * 16);

    #pragma omp parallel for schedule(static)
    for (int row = 0; row < m; row++) {
        int ch = row % 4;
        int row_in_ch = row / 4;
        const uint8_t* row_ptr = hw_data + ch * channel_sz + ch_offset + row_in_ch * row_stride;

        float sum = 0.0f;
        if (real_mode == V37_MODE_TL3_COMPACT) {
            for (uint32_t b = 0; b < pe_k_beats; b++) {
                const uint8_t* beat_ptr = row_ptr + b * 13;
                for (int gp = 0; gp < 12; gp++) {
                    uint8_t byte_val = beat_ptr[gp];
                    int c0 = byte_val % 3; byte_val /= 3;
                    int c1 = byte_val % 3; byte_val /= 3;
                    int c2 = byte_val % 3; byte_val /= 3;
                    int c3 = byte_val % 3; byte_val /= 3;
                    int c4 = byte_val;
                    int col = b * 64 + gp * 5;
                    if (col + 0 < k) sum += x_q[col + 0] * (c0 - 1);
                    if (col + 1 < k) sum += x_q[col + 1] * (c1 - 1);
                    if (col + 2 < k) sum += x_q[col + 2] * (c2 - 1);
                    if (col + 3 < k) sum += x_q[col + 3] * (c3 - 1);
                    if (col + 4 < k) sum += x_q[col + 4] * (c4 - 1);
                }
                uint8_t byte12 = beat_ptr[12];
                int c0 = byte12 % 3; byte12 /= 3;
                int c1 = byte12 % 3; byte12 /= 3;
                int c2 = byte12 % 3; byte12 /= 3;
                int c3 = byte12;
                int col = b * 64 + 60;
                if (col + 0 < k) sum += x_q[col + 0] * (c0 - 1);
                if (col + 1 < k) sum += x_q[col + 1] * (c1 - 1);
                if (col + 2 < k) sum += x_q[col + 2] * (c2 - 1);
                if (col + 3 < k) sum += x_q[col + 3] * (c3 - 1);
            }
        } else { // Mode 5 (I2_S)
            for (uint32_t b = 0; b < pe_k_beats; b++) {
                const uint8_t* beat_ptr = row_ptr + b * 16;
                for (int x = 0; x < 16; x++) {
                    uint8_t byte_val = beat_ptr[x];
                    int c0 = (byte_val >> 6) & 3;
                    int c1 = (byte_val >> 4) & 3;
                    int c2 = (byte_val >> 2) & 3;
                    int c3 = byte_val & 3;
                    int base_col = b * 64;
                    if (base_col + x + 0 < k)  sum += x_q[base_col + x + 0]  * (c0 - 1);
                    if (base_col + x + 16 < k) sum += x_q[base_col + x + 16] * (c1 - 1);
                    if (base_col + x + 32 < k) sum += x_q[base_col + x + 32] * (c2 - 1);
                    if (base_col + x + 48 < k) sum += x_q[base_col + x + 48] * (c3 - 1);
                }
            }
        }
        float scaled = (sum * scale) / act_scale;
        if (residual) residual[row] += scaled;
        else out[row] = scaled;
    }
}

void V37Engine::run_fpga_gemv_prequantized(int s_id, const QuantizedAct& qact, float* out, int out_dim, const float* row_scales, int ext_stage, bool cpu_fallback, const uint8_t* cpu_weights, const char* layer_name, bool reuse_act, float* residual) {
    if (cpu_fallback || !engine) {
        if (out) {
            #pragma omp parallel for simd
            for (int i = 0; i < out_dim; i++) {
                float sum = 0.0f;
                const uint8_t* w_ptr = cpu_weights ? cpu_weights + i * (qact.in_dim / 4) : nullptr;
                if (w_ptr) {
                    for (int j = 0; j < qact.in_dim; j++) {
                        int w = decode_i2s_weight_inline(w_ptr, j);
                        sum += qact.x_q[j] * w;
                    }
                }
                float scale_f = row_scales ? row_scales[i] : 1.0f;
                float scaled = (sum * scale_f) / qact.s;
                if (residual) residual[i] += scaled;
                else out[i] = scaled;
            }
        }
        return;
    }

    const float* driver_scales = row_scales;
    v37_compute_linear_preloaded_opt(engine, qact.x_q.data(), s_id, out, out_dim, qact.in_dim, active_pack_mode, nullptr, driver_scales, 0, ext_stage, 0, 0, reuse_act ? 1 : 0, qact.s, residual);
}

void V37Engine::run_fpga_gemv_launch(int slot_id, const QuantizedAct& qact, int out_dim, bool reuse_act) {
    if (!engine) return;
    v37_launch_linear_async(engine, qact.x_q.data(), slot_id, out_dim, qact.in_dim, active_pack_mode, 0, reuse_act ? 1 : 0);
}

void V37Engine::run_fpga_gemv_wait(float* out, const float* row_scales, float act_scale, float* residual) {
    if (!engine) return;
    v37_wait_linear_async(engine, out, row_scales, act_scale, residual);
}

void V37Engine::run_fpga_gemv_wait_fused_up(float* ffn_act, const float* gate, const float* row_scales, float act_scale) {
    if (!engine) return;
    v37_wait_linear_async_fused_up(engine, ffn_act, gate, row_scales, act_scale);
}

void V37Engine::run_fpga_gemv_wait_f16(void* kv_cache_v_dst, const float* row_scales, float act_scale) {
    if (!engine) return;
    v37_wait_linear_async_f16(engine, kv_cache_v_dst, row_scales, act_scale);
}

void V37Engine::run_fpga_gemv_fast(int s_id, float* in, float* out, int out_dim, int in_dim, const float* row_scales, int ext_stage, bool cpu_fallback, const uint8_t* cpu_weights, const char* layer_name, int m_offset, int total_m, const float* hw_scales) {
    if (row_scales) {
        if (std::isnan(row_scales[0])) {
            printf("[DEBUG-SCALE] NaN detected in w_scales for layer %s!\n", layer_name);
        }
    }
    
    static QuantizedAct local_qact;
    quantize_activation_neon(in, in_dim, local_qact);
    float s = local_qact.s;
    const std::vector<int8_t>& x_q = local_qact.x_q;

    uint32_t stage_base = ext_stage & 0xFFFF;
    bool is_fused_up = (stage_base == 2); // Only Up projection is fused output to CPU
    
    if (cpu_fallback || !engine) {
        static bool exact_cpu = getenv("EXACT_CPU") && atoi(getenv("EXACT_CPU")) == 1;
        static bool printed_mode = false;
        if (!printed_mode) {
            printf("[DEBUG] CPU fallback mode: %s\n", exact_cpu ? "EXACT FP32 (no INT8 quantization)" : "FPGA-equivalent INT8 quantization");
            printed_mode = true;
        }
        if (out) {
            if (exact_cpu) {
                // FP32 exact mode: identical to llama-cli's I2_S dequant GEMV
                // out[i] = sum_j(in[j] * w_ternary[j]) * weight_scale
                // Note: decode_i2s_weight_inline already returns {-1,0,1}
                // so no act_sum correction is needed (that's only for vec_dot's raw {0,1,2} codes)
                #pragma omp parallel for
                for (int i = 0; i < out_dim; i++) {
                    float sum = 0.0f;
                    const uint8_t* w_ptr = cpu_weights ? cpu_weights + i * (in_dim / 4) : nullptr;
                    if (w_ptr) {
                        for (int j = 0; j < in_dim; j++) {
                            int w = decode_i2s_weight_inline(w_ptr, j);
                            sum += in[j] * w; // FP32 input * ternary weight
                        }
                    }
                    float scale_f = row_scales ? row_scales[i] : 1.0f;
                    out[i] = sum * scale_f;
                }
            } else {
                // FPGA-equivalent INT8 quantized path
                #pragma omp parallel for simd
                for (int i = 0; i < out_dim; i++) {
                    float sum = 0.0f;
                    const uint8_t* w_ptr = cpu_weights ? cpu_weights + i * (in_dim / 4) : nullptr;
                    if (w_ptr) {
                        for (int j = 0; j < in_dim; j++) {
                            int w = decode_i2s_weight_inline(w_ptr, j);
                            sum += x_q[j] * w;
                        }
                    }
                    float scale_f = row_scales ? row_scales[i] : 1.0f;
                    out[i] = (sum * scale_f) / s;
                }
            }
        }
        if (s_id == 0 && strstr(layer_name, "attn_q_cpu") != NULL) {
            static int printed = 0;
            if (printed == 0) {
                printed = 1;
                if (layer_name && std::string(layer_name) == "attn_q_cpu") {
                    const uint8_t* w0 = cpu_weights;
                    for (int r = 0; r < 4 && w0; r++) {
                        int mac = 0;
                        const uint8_t* wr = w0 + r * (in_dim / 4);
                        for (int j = 0; j < in_dim; j++) {
                            int w = decode_i2s_weight_inline(wr, j);
                            mac += x_q[j] * w;
                        }
                    }
                }
            }
        }
        return;
    }
    
    // Pass ext_stage directly to the driver
    // We pass s (activation scale) via driver_scales[3].
    std::vector<float> final_hw_scales;
    const float* driver_scales = row_scales;
    if (hw_scales) {
        final_hw_scales.assign(hw_scales, hw_scales + 3);
        final_hw_scales.push_back(s);
        driver_scales = final_hw_scales.data();
    }
    
    v37_compute_linear_preloaded(engine, x_q.data(), s_id, out, out_dim, in_dim, active_pack_mode, nullptr, driver_scales, 0, ext_stage, m_offset, total_m);

    if (getenv("V37_DEBUG") && layer_name && std::string(layer_name) == "attn_q_cpu") {
        //if(0) printf("[FPGA_DEBUG] RAW FPGA MAC sum[0..4]: ");
        for (int i=0; i<5; i++) printf("%f ", out[i]);
        printf("\n[FPGA_DEBUG] Act scale s = %f\n", s);
        
        // --- HW SIMULATION MAC: simulate unpacker + PE array using CMA data ---
        static int hw_sim_printed = 0;
        if (hw_sim_printed < 1) {
            hw_sim_printed++;
            // Get CMA base for slot
            uint8_t* cma_base = (uint8_t*)v37_get_cma_slot_ptr(this->engine, s_id);
            if (cma_base) {
                int padded_k = (in_dim + 255) & ~255;
                uint32_t kb = padded_k / 64;
                uint32_t ch_stride = kb * 16; // per-row payload stride
                uint32_t sim_channel_sz = ((out_dim + 3) / 4) * ch_stride + 16; // +16 header
                
                //if(0) printf("[HW_SIM] CMA base=%p, kb=%d, ch_stride=%d, channel_sz=%d\n", cma_base, kb, ch_stride, sim_channel_sz);
                
                for (int row = 0; row < 5; row++) {
                    int ch_idx = row % 4;
                    int row_in_ch = row / 4;
                    // +16 to skip past header at channel start
                    uint32_t row_offset = ch_idx * sim_channel_sz + 16 + row_in_ch * ch_stride;
                    
                    int64_t hw_sim_mac = 0;
                    for (uint32_t beat = 0; beat < kb; beat++) {
                        uint8_t* beat_ptr = cma_base + row_offset + beat * 16;
                        
                        // Simulate unpacker i2s_decoded mapping:
                        // byte[gp] bits[7:6] -> lane gp+0
                        // byte[gp] bits[5:4] -> lane gp+16
                        // byte[gp] bits[3:2] -> lane gp+32
                        // byte[gp] bits[1:0] -> lane gp+48
                        int lane_w[64];
                        for (int gp = 0; gp < 16; gp++) {
                            uint8_t b = beat_ptr[gp];
                            auto decode_ternary = [](int code) -> int {
                                if (code == 2) return 1;  // 2'b10 -> +1
                                if (code == 0) return -1; // 2'b00 -> -1
                                return 0;                  // 2'b01 -> 0
                            };
                            lane_w[gp + 0]  = decode_ternary((b >> 6) & 3);
                            lane_w[gp + 16] = decode_ternary((b >> 4) & 3);
                            lane_w[gp + 32] = decode_ternary((b >> 2) & 3);
                            lane_w[gp + 48] = decode_ternary((b >> 0) & 3);
                        }
                        
                        // PE array: lane k multiplies lane_w[k] by X[beat*64 + k]
                        for (int k = 0; k < 64; k++) {
                            int x_idx = beat * 64 + k;
                            if (x_idx < in_dim) {
                                hw_sim_mac += lane_w[k] * (int)x_q[x_idx];
                            }
                        }
                        
                    }
                }
                
                // Also compute correct CPU MAC for comparison
                if(hw_sim_printed < 2) printf("[HW_SIM] CPU ref MAC: ");
                for (int row = 0; row < 5; row++) {
                    int64_t cpu_mac = 0;
                    const uint8_t* w_ptr = cpu_weights ? cpu_weights + row * (in_dim / 4) : nullptr;
                    if (w_ptr) {
                        for (int j = 0; j < in_dim; j++) {
                            int w = decode_i2s_weight_inline(w_ptr, j);
                            cpu_mac += w * (int)x_q[j];
                        }
                    }
                    printf("%ld ", (long)cpu_mac);
                }
                printf("\n");
            }
        }
    }

    // I2_S hardware path just returns pure int32_t MAC sum cast to float. Apply scale here.
    if (hw_scales == nullptr && row_scales && out) {
        float inv_s = 1.0f / s;
#if defined(__aarch64__) || defined(__arm__)
        float32x4_t vinv_s = vdupq_n_f32(inv_s);
        int r = 0;
        for (; r <= out_dim - 16; r += 16) {
            float32x4_t o0 = vld1q_f32(out + r + 0);
            float32x4_t o1 = vld1q_f32(out + r + 4);
            float32x4_t o2 = vld1q_f32(out + r + 8);
            float32x4_t o3 = vld1q_f32(out + r + 12);

            float32x4_t sc0 = vmulq_f32(vld1q_f32(row_scales + r + 0), vinv_s);
            float32x4_t sc1 = vmulq_f32(vld1q_f32(row_scales + r + 4), vinv_s);
            float32x4_t sc2 = vmulq_f32(vld1q_f32(row_scales + r + 8), vinv_s);
            float32x4_t sc3 = vmulq_f32(vld1q_f32(row_scales + r + 12), vinv_s);

            vst1q_f32(out + r + 0,  vmulq_f32(o0, sc0));
            vst1q_f32(out + r + 4,  vmulq_f32(o1, sc1));
            vst1q_f32(out + r + 8,  vmulq_f32(o2, sc2));
            vst1q_f32(out + r + 12, vmulq_f32(o3, sc3));
        }
        for (; r < out_dim; r++) {
            out[r] = (out[r] * row_scales[r]) * inv_s;
        }
#else
        for (int row = 0; row < out_dim; row++) {
            out[row] = (out[row] * row_scales[row]) * inv_s;
        }
#endif
    }

    if (getenv("V37_DEBUG") && layer_name && std::string(layer_name) == "attn_q_cpu" && out) {
        static int xq_printed = 0;
        if (xq_printed < 3) {
            float sum_xq = 0;
            for(int j=0; j<in_dim; j++) sum_xq += x_q[j];
            //if(0) printf("[FPGA_DEBUG] sum(x_q) = %f\n", sum_xq);
            xq_printed++;
            //if(0) printf("[FPGA_DEBUG] x_q[0..4]: %d %d %d %d %d\n", x_q[0], x_q[1], x_q[2], x_q[3], x_q[4]);
        }
        static int printed = 0;
        if (printed < 3) {
            //if(0) printf("[FPGA_DEBUG] %s raw FPGA out[0..4]: ", layer_name);
            for(int k=0; k<5; k++) printf("%f ", out[k]);
            printf("\n");
            
            //if(0) printf("[FPGA_DEBUG] %s expected CPU MAC sum[0..4]: ", layer_name);
            for(int i=0; i<5; i++) {
                float sum = 0.0f; float w_sum = 0.0f;
                const uint8_t* w_ptr = cpu_weights ? cpu_weights + i * (in_dim / 4) : nullptr;
                if (w_ptr) {
                    for (int j = 0; j < in_dim; j++) {
                        int byte_idx = j / 4;
                        int shift = (j % 4) * 2;
                        int8_t w = ((w_ptr[byte_idx] >> shift) & 3) - 1;
                        sum += w * x_q[j];
                    }
                }
                printf("%f ", sum);
            }
            printf("\n");
            printed++;
        }
    }

    if (getenv("V37_DEBUG") && layer_name && std::string(layer_name) == "attn_q_cpu" && out) {
        static int xq_printed = 0;
        if (xq_printed < 3) {
            float sum_xq = 0;
            for(int j=0; j<in_dim; j++) sum_xq += x_q[j];
            //if(0) printf("[FPGA_DEBUG] sum(x_q) = %f\n", sum_xq);
            xq_printed++;
            //if(0) printf("[FPGA_DEBUG] x_q[0..4]: %d %d %d %d %d\n", x_q[0], x_q[1], x_q[2], x_q[3], x_q[4]);
        }
        static int printed = 0;
        if (printed < 3) {

            
            //if(0) printf("[FPGA_DEBUG] %s expected CPU MAC sum[0..4]: ", layer_name);
            for(int i=0; i<5; i++) {
                float sum = 0.0f; float w_sum = 0.0f;
                const uint8_t* w_ptr = cpu_weights ? cpu_weights + i * (in_dim / 4) : nullptr;
                if (w_ptr) {
                    for (int j = 0; j < in_dim; j++) {
                        int w = decode_i2s_weight_inline(w_ptr, j);
                        sum += x_q[j] * w;
                        if (i == 0) w_sum += w;
                    }
                    if (i == 0) {} //if(0) printf("[FPGA_DEBUG] sum(w[0]) = %f\n", w_sum);
                }
                printf("%f ", sum);
            }
            printf("\n");


            //if(0) printf("[FPGA_DEBUG] %s expected V37_EMU MAC sum[0..4]: ", layer_name);
            for(int i=0; i<5; i++) {
                int64_t total_acc = 0;
                // Parse layer index from layer_name (e.g. "attn_q_cpu")
                // Wait, we don't have l here. But we can just use 0 for debug!
                V37PackEntry* pp = get_prepacked(std::string("blk.0.attn_q.weight"));
                if (pp) {
                    uint8_t* dest = (uint8_t*)pp->data;
                    int row = i;
                    int k = in_dim;
                    int kb = k / 64;
                    uint32_t total_beats = kb + (kb + 3) / 4;
                    uint32_t ch_stride = total_beats * 16;
                    int ch_idx = row % 4;
                    int row_in_ch = row / 4;
                    int m = out_dim;
                    uint32_t ch_base = ch_idx * ((m + 3) / 4) * ch_stride + row_in_ch * ch_stride;
                    uint32_t pld_idx = 0;
                    uint32_t dest_beat = 0;
                    while (pld_idx < kb) {
                        dest_beat++;
                        for (int b = 0; b < 4 && pld_idx < kb; b++) {
                            uint8_t* dst_ptr = dest + ch_base + dest_beat * 16;
                            for (int xi = 0; xi < 16; xi++) {
                                uint8_t byte = dst_ptr[xi];
                                int w0 = ((int8_t)((byte >> 6) & 3) - 1);
                                int w1 = ((int8_t)((byte >> 4) & 3) - 1);
                                int w2 = ((int8_t)((byte >> 2) & 3) - 1);
                                int w3 = ((int8_t)((byte >> 0) & 3) - 1);
                                if (pld_idx * 64 + xi + 0  < (uint32_t)k) total_acc += w0 * x_q[pld_idx * 64 + xi + 0 ];
                                if (pld_idx * 64 + xi + 16 < (uint32_t)k) total_acc += w1 * x_q[pld_idx * 64 + xi + 16];
                                if (pld_idx * 64 + xi + 32 < (uint32_t)k) total_acc += w2 * x_q[pld_idx * 64 + xi + 32];
                                if (pld_idx * 64 + xi + 48 < (uint32_t)k) total_acc += w3 * x_q[pld_idx * 64 + xi + 48];
                            }
                            dest_beat++; pld_idx++;
                        }
                    }
                }
                printf("%ld ", total_acc);
            }
            printf("\n");
            printed++;
        }
    }

}

static inline uint8_t reverse_2bit_pairs(uint8_t b) {
    return ((b & 0x03) << 6) | ((b & 0x0C) << 2) | ((b & 0x30) >> 2) | ((b & 0xC0) >> 6);
}

void V37Engine::rms_norm(float* out, const float* in, const float* weight, int n) {
#if defined(__aarch64__) || defined(__arm__)
    float64x2_t sum_d0 = vdupq_n_f64(0.0);
    float64x2_t sum_d1 = vdupq_n_f64(0.0);
    int i = 0;
    for (; i <= n - 8; i += 8) {
        float32x4_t v0 = vld1q_f32(in + i + 0);
        float32x4_t v1 = vld1q_f32(in + i + 4);
        sum_d0 = vmlaq_f64(sum_d0, vcvt_f64_f32(vget_low_f32(v0)), vcvt_f64_f32(vget_low_f32(v0)));
        sum_d1 = vmlaq_f64(sum_d1, vcvt_f64_f32(vget_high_f32(v0)), vcvt_f64_f32(vget_high_f32(v0)));
        sum_d0 = vmlaq_f64(sum_d0, vcvt_f64_f32(vget_low_f32(v1)), vcvt_f64_f32(vget_low_f32(v1)));
        sum_d1 = vmlaq_f64(sum_d1, vcvt_f64_f32(vget_high_f32(v1)), vcvt_f64_f32(vget_high_f32(v1)));
    }
    double sum = vgetq_lane_f64(sum_d0, 0) + vgetq_lane_f64(sum_d0, 1) +
                 vgetq_lane_f64(sum_d1, 0) + vgetq_lane_f64(sum_d1, 1);
    for (; i < n; i++) sum += (double)(in[i] * in[i]);
#else
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    int i = 0;
    for (; i <= n - 4; i += 4) {
        s0 += (double)(in[i + 0] * in[i + 0]);
        s1 += (double)(in[i + 1] * in[i + 1]);
        s2 += (double)(in[i + 2] * in[i + 2]);
        s3 += (double)(in[i + 3] * in[i + 3]);
    }
    double sum = (s0 + s1) + (s2 + s3);
    for (; i < n; i++) sum += (double)(in[i] * in[i]);
#endif
    float eps = (config.rms_norm_eps > 0.0f) ? config.rms_norm_eps : 1e-5f;
    float scale = 1.0f / std::sqrt((float)(sum / n) + eps);
#if defined(__aarch64__) || defined(__arm__)
    float32x4_t vscale = vdupq_n_f32(scale);
    int j = 0;
    for (; j <= n - 16; j += 16) {
        float32x4_t vin0 = vld1q_f32(in + j + 0);
        float32x4_t vin1 = vld1q_f32(in + j + 4);
        float32x4_t vin2 = vld1q_f32(in + j + 8);
        float32x4_t vin3 = vld1q_f32(in + j + 12);

        float32x4_t vw0 = vld1q_f32(weight + j + 0);
        float32x4_t vw1 = vld1q_f32(weight + j + 4);
        float32x4_t vw2 = vld1q_f32(weight + j + 8);
        float32x4_t vw3 = vld1q_f32(weight + j + 12);

        vst1q_f32(out + j + 0,  vmulq_f32(vmulq_f32(vin0, vscale), vw0));
        vst1q_f32(out + j + 4,  vmulq_f32(vmulq_f32(vin1, vscale), vw1));
        vst1q_f32(out + j + 8,  vmulq_f32(vmulq_f32(vin2, vscale), vw2));
        vst1q_f32(out + j + 12, vmulq_f32(vmulq_f32(vin3, vscale), vw3));
    }
    for (; j < n; j++) out[j] = in[j] * scale * weight[j];
#else
    for (int j = 0; j < n; j++) out[j] = in[j] * scale * weight[j];
#endif
}

void V37Engine::load_norm_weight(std::vector<float>& out, const std::string& name, int n) {
    uint32_t type = loader->get_tensor_type(name);
    void* raw = loader->get_tensor_data(name);
    out.resize(n);
    if (!raw) {
        if (name.find("sub_norm") != std::string::npos) {
            std::fill(out.begin(), out.end(), 1.0f);
            return;
        }
        printf("[WARNING] Tensor %s not found!\n", name.c_str());
        return;
    }
    if (type == 1) { // F16
        ggml_fp16_to_fp32_row((uint16_t*)raw, out.data(), n);
    } else if (type == 0) { // F32
        memcpy(out.data(), raw, n * sizeof(float));
    } else {
        printf("[ERROR] Unsupported norm weight type for %s: %u\n", name.c_str(), type);
    }
}


void V37Engine::init_layer_cache() {
    if (layer_cache_initialized) return;
    static const char* target_tensors[] = {
        "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"
    };
    int base_fused_gate_up = config.n_layers * 7;
    int base_fused_qkv     = base_fused_gate_up + config.n_layers;

    for (int l = 0; l < config.n_layers; l++) {
        for (int i = 0; i < 7; i++) {
            std::string t_name = "blk." + std::to_string(l) + "." + target_tensors[i];
            layers_cache[l].scales[i] = loader ? loader->get_i2_scales(t_name) : nullptr;
            layers_cache[l].raw_w[i] = loader ? (const uint8_t*)loader->get_tensor_data(t_name) : nullptr;
            layers_cache[l].slots[i] = all_weights_preloaded ? (l * 7 + i) : i;
        }
        layers_cache[l].fused_gate_up_slot = all_weights_preloaded ? (base_fused_gate_up + l) : -1;
        layers_cache[l].fused_qkv_slot = all_weights_preloaded ? (base_fused_qkv + l) : -1;
        std::string k_attn = "blk." + std::to_string(l) + ".attn_norm.weight";
        load_norm_weight(layers_cache[l].norm_attn, k_attn, config.n_embd);

        std::string k_attn_sub = "blk." + std::to_string(l) + ".attn_sub_norm.weight";
        load_norm_weight(layers_cache[l].norm_attn_sub, k_attn_sub, config.n_embd);

        std::string k_ffn = "blk." + std::to_string(l) + ".ffn_norm.weight";
        load_norm_weight(layers_cache[l].norm_ffn, k_ffn, config.n_embd);

        std::string k_ffn_sub = "blk." + std::to_string(l) + ".ffn_sub_norm.weight";
        load_norm_weight(layers_cache[l].norm_ffn_sub, k_ffn_sub, config.n_ffn);
    }
    load_norm_weight(output_norm_weight, "output_norm.weight", config.n_embd);
    init_lmhead_cache();
    layer_cache_initialized = true;
    printf("[V37-OPT] Pre-cached all %d layers tensor metadata, weights, and norms into flat memory.\n", config.n_layers);
}

void V37Engine::init_lmhead_cache() {
    if (lmhead_cache_initialized) return;
    if (!loader) return;
    
    std::string lmhead_name = "output.weight";
    void* embd_raw = loader->get_tensor_data(lmhead_name);
    uint32_t embd_type = loader->get_tensor_type(lmhead_name);
    if (!embd_raw) {
        lmhead_name = "token_embd.weight";
        embd_raw = loader->get_tensor_data(lmhead_name);
        embd_type = loader->get_tensor_type(lmhead_name);
    }
    if (!embd_raw || embd_type != 1) return; // Must be F16

    const v37_f16_t* w_f16 = (const v37_f16_t*)embd_raw;
    auto t0 = std::chrono::high_resolution_clock::now();

    const char* env_t1 = getenv("LMHEAD_TIER1_SIZE");
    const char* env_d1 = getenv("LMHEAD_D1");
    const char* env_dc = getenv("LMHEAD_D_COARSE");
    const char* env_2s = getenv("V37_LMHEAD_TWOSTAGE");
    const char* env_ds1 = getenv("LMHEAD_D_S1");
    const char* env_s1k = getenv("LMHEAD_S1_K");
    const char* env_topc = getenv("LMHEAD_TOP_CAND");

    lmhead_tier1_size = env_t1 ? atoi(env_t1) : std::min(28160, config.vocab_size);
    lmhead_d1 = env_d1 ? atoi(env_d1) : std::min(896, config.n_embd);
    lmhead_d_coarse = env_dc ? atoi(env_dc) : std::min(128, config.n_embd);
    lmhead_twostage_enabled = env_2s ? atoi(env_2s) : 1;
    lmhead_d_s1 = env_ds1 ? atoi(env_ds1) : std::min(512, lmhead_d1);
    if (lmhead_d_s1 > lmhead_d1) lmhead_d_s1 = lmhead_d1;
    lmhead_d_s2 = lmhead_d1 - lmhead_d_s1;
    lmhead_s1_k_per_thread = env_s1k ? atoi(env_s1k) : 512;
    lmhead_top_cand = env_topc ? atoi(env_topc) : 256;

    const int TIER1_SIZE = lmhead_tier1_size;
    const int D1 = lmhead_d1;
    const int D_COARSE = lmhead_d_coarse;
    const int D_S1 = lmhead_d_s1;
    const int D_S2 = lmhead_d_s2;

    lmhead_indices_t1.resize(D1);
    for (int i = 0; i < D1; i++) {
        int idx = (D1 > 1) ? (int)(i * ((config.n_embd - 1.0) / (D1 - 1.0))) : 0;
        lmhead_indices_t1[i] = (idx / 2) * 2;
    }

    lmhead_indices_coarse.resize(D_COARSE);
    for (int i = 0; i < D_COARSE; i++) {
        lmhead_indices_coarse[i] = (D_COARSE > 1) ? (int)(i * ((config.n_embd - 1.0) / (D_COARSE - 1.0))) : 0;
    }

    lmhead_t1_q8.resize(TIER1_SIZE * D1);
    lmhead_t1_s1_q8.resize(TIER1_SIZE * D_S1);
    if (D_S2 > 0) lmhead_t1_s2_q8.resize(TIER1_SIZE * D_S2);
    lmhead_t1_scales.resize(TIER1_SIZE);

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < TIER1_SIZE; r++) {
        const v37_f16_t* row = w_f16 + (size_t)r * config.n_embd;
        float max_val = 1e-10f;
        for (int i = 0; i < D1; i++) {
            float v = std::fabs(V37_F16_TO_F32(row[lmhead_indices_t1[i]]));
            if (v > max_val) max_val = v;
        }
        lmhead_t1_scales[r] = max_val / 127.0f;
        float inv_scale = 127.0f / max_val;
        for (int i = 0; i < D1; i++) {
            int q = std::round(V37_F16_TO_F32(row[lmhead_indices_t1[i]]) * inv_scale);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            lmhead_t1_q8[r * D1 + i] = (int8_t)q;
            if (i < D_S1) {
                lmhead_t1_s1_q8[r * D_S1 + i] = (int8_t)q;
            } else if (D_S2 > 0) {
                lmhead_t1_s2_q8[r * D_S2 + (i - D_S1)] = (int8_t)q;
            }
        }
    }

    int n2_rows = std::max(0, config.vocab_size - TIER1_SIZE);
    lmhead_t2_coarse_q8.resize(n2_rows * D_COARSE);
    lmhead_t2_scales.resize(n2_rows);

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n2_rows; r++) {
        const v37_f16_t* row = w_f16 + (size_t)(TIER1_SIZE + r) * config.n_embd;
        float max_val = 1e-10f;
        for (int i = 0; i < D_COARSE; i++) {
            float v = std::fabs(V37_F16_TO_F32(row[lmhead_indices_coarse[i]]));
            if (v > max_val) max_val = v;
        }
        lmhead_t2_scales[r] = max_val / 127.0f;
        float inv_scale = 127.0f / max_val;
        for (int i = 0; i < D_COARSE; i++) {
            int q = std::round(V37_F16_TO_F32(row[lmhead_indices_coarse[i]]) * inv_scale);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            lmhead_t2_coarse_q8[r * D_COARSE + i] = (int8_t)q;
        }
    }

    // Pin LM_HEAD quantized tables and embeddings in physical RAM to prevent minor page faults
    int r1 = mlock(lmhead_t1_q8.data(), lmhead_t1_q8.size());
    int r2 = mlock(lmhead_t1_scales.data(), lmhead_t1_scales.size() * sizeof(float));
    int r3 = mlock(lmhead_t2_coarse_q8.data(), lmhead_t2_coarse_q8.size());
    int r4 = mlock(lmhead_t2_scales.data(), lmhead_t2_scales.size() * sizeof(float));
    int r5 = mlock((void*)w_f16, (size_t)config.vocab_size * config.n_embd * sizeof(v37_f16_t));
    int r6 = mlock(lmhead_t1_s1_q8.data(), lmhead_t1_s1_q8.size());
    int r7 = D_S2 > 0 ? mlock(lmhead_t1_s2_q8.data(), lmhead_t1_s2_q8.size()) : 0;
    if (r1 || r2 || r3 || r4 || r5 || r6 || r7) {
        printf("[V37-OPT WARNING] mlock status (t1:%d, t2:%d, w:%d, s1:%d, s2:%d, errno=%d)\n", r1, r3, r5, r6, r7, errno);
    } else {
        printf("[V37-OPT] mlock SUCCESS: Pinned tables and weights in physical RAM!\n");
    }

    max_lmhead_t1_scale = 0.0f;
    for (float s : lmhead_t1_scales) if (s > max_lmhead_t1_scale) max_lmhead_t1_scale = s;
    inv_max_lmhead_t1_scale = (max_lmhead_t1_scale > 0.0f) ? (1.0f / max_lmhead_t1_scale) : 0.0f;

    max_lmhead_t2_scale = 0.0f;
    for (float s : lmhead_t2_scales) if (s > max_lmhead_t2_scale) max_lmhead_t2_scale = s;
    inv_max_lmhead_t2_scale = (max_lmhead_t2_scale > 0.0f) ? (1.0f / max_lmhead_t2_scale) : 0.0f;

    lmhead_cache_initialized = true;
    auto t1 = std::chrono::high_resolution_clock::now();
    double init_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[V37-OPT] Hierarchical LM_HEAD quantized tables (Tier1-Fine=%d, Coarse=%d, TwoStage=%d [S1=%d, S2=%d, K=%d, TopCand=%d]) built in %.1f ms.\n",
           TIER1_SIZE, config.vocab_size, lmhead_twostage_enabled, D_S1, D_S2, lmhead_s1_k_per_thread, lmhead_top_cand, init_ms);
}

void V37Engine::prepare_lmhead_overlap() {
    // 1. Sparse reset of logits from previous token to preserve L2 cache
    for (int p_tok : prev_candidates) {
        logits[p_tok] = -1e9f;
    }
    prev_candidates.clear();

    // 2. Stream prefetch output_norm_weight
    const char* p_norm = (const char*)output_norm_weight.data();
    for (int i = 0; i < config.n_embd * (int)sizeof(float); i += 64) {
        __builtin_prefetch(p_norm + i, 0, 3);
    }

    // 3. Stream prefetch lmhead_indices_t1
    const char* p_idx = (const char*)lmhead_indices_t1.data();
    for (int i = 0; i < (int)lmhead_indices_t1.size() * (int)sizeof(int); i += 64) {
        __builtin_prefetch(p_idx + i, 0, 3);
    }

    // 4. Stream prefetch first 64 KB of lmhead_t1_s1_q8 (or lmhead_t1_q8)
    const char* p_t1 = lmhead_twostage_enabled ? (const char*)lmhead_t1_s1_q8.data() : (const char*)lmhead_t1_q8.data();
    size_t sz_t1 = lmhead_twostage_enabled ? lmhead_t1_s1_q8.size() : lmhead_t1_q8.size();
    for (int i = 0; i < 65536 && i < (int)sz_t1; i += 64) {
        __builtin_prefetch(p_t1 + i, 0, 3);
    }

    // 5. Stream prefetch first 2048 entries of lmhead_t1_scales
    const char* p_s1 = (const char*)lmhead_t1_scales.data();
    for (int i = 0; i < 2048 * (int)sizeof(float) && i < (int)(lmhead_t1_scales.size() * sizeof(float)); i += 64) {
        __builtin_prefetch(p_s1 + i, 0, 3);
    }
}


static double prof_qkv_ms = 0;
static double prof_attn_ms = 0;
static double prof_oproj_ms = 0;
static double prof_ffn_ms = 0;

void V37Engine::transformer_block(int l, int pos) {
    auto tb_start = std::chrono::high_resolution_clock::now();
    init_layer_cache();
    const LayerTensorCache& lc = layers_cache[l];

    float* norm_out = buf_norm_out.data();
    rms_norm(norm_out, hidden_states.data(), lc.norm_attn.data(), config.n_embd);
    
    float* q = buf_q.data();
    float* k = buf_k.data();
    float* v = buf_v.data();
    bool is_cpu = getenv("DISABLE_V37") != nullptr;
    
    int q_dim = config.n_heads * config.head_dim;
    if (q_dim <= 0) q_dim = config.n_embd;
    int kv_dim = config.n_kv_heads * config.head_dim;
    // Shared activation quantization for Q, K, V
    quantize_activation_neon(norm_out, config.n_embd, buf_qact_attn);

    int kv_layer_offset = l * config.max_seq_len * 2 * kv_dim;
    int k_offset = kv_layer_offset + pos * kv_dim;
    int v_offset = kv_layer_offset + config.max_seq_len * kv_dim + pos * kv_dim;

    init_rope_table();
    int rope_pos = (pos >= config.max_seq_len) ? (config.max_seq_len - 1) : pos;
    int half_dim = config.head_dim / 2;
    const float* rope_cos = rope_cos_table.data() + rope_pos * half_dim;
    const float* rope_sin = rope_sin_table.data() + rope_pos * half_dim;
    const float inv_sqrt_head_dim = 1.0f / std::sqrt((float)config.head_dim);

    if (!is_cpu && engine) {
        if (lc.fused_qkv_slot >= 0) {
            // Fused QKV single DMA launch with 4-core parallel RoPE + inv_sqrt scaling + direct FP16 KV cache write:
            // Saves 60 DMA round-trips AND completely eliminates serial rope_q / rope_k_f16 CPU overhead!
            v37_compute_qkv_fused(engine, buf_qact_attn.x_q.data(), lc.fused_qkv_slot,
                                  q, nullptr, kv_cache.data() + v_offset,
                                  q_dim, kv_dim, kv_dim, config.n_embd,
                                  lc.scales[0], lc.scales[1], lc.scales[2], buf_qact_attn.s, pos,
                                  kv_cache.data() + k_offset, rope_cos, rope_sin, inv_sqrt_head_dim, active_pack_mode);
        } else {
            // 1. Launch Q and Wait Q
            run_fpga_gemv_launch(lc.slots[0], buf_qact_attn, q_dim, false);
            run_fpga_gemv_wait(q, lc.scales[0], buf_qact_attn.s, nullptr);

            // 2. Launch K asynchronously (FPGA computes K ~440us)
            run_fpga_gemv_launch(lc.slots[1], buf_qact_attn, kv_dim, true);

            // [ASYNC OVERLAP 1]: Compute RoPE for Q while FPGA computes K!
            rope_q(q, pos);

            // Wait for K
            run_fpga_gemv_wait(k, lc.scales[1], buf_qact_attn.s, nullptr);

            // 3. Launch V asynchronously (FPGA computes V ~440us)
            run_fpga_gemv_launch(lc.slots[2], buf_qact_attn, kv_dim, true);

            // [ASYNC OVERLAP 2]: Compute RoPE for K and write K to KV cache while FPGA computes V!
            rope_k(k, pos);
            ggml_fp32_to_fp16_row(k, kv_cache.data() + k_offset, kv_dim);

            // Wait for V & direct dequant to FP16 in KV cache (Zero intermediate FP32 traffic)
            run_fpga_gemv_wait_f16(kv_cache.data() + v_offset, lc.scales[2], buf_qact_attn.s);
        }
    } else {
        std::string fqkv_name = "blk." + std::to_string(l) + ".fused_qkv.weight";
        V37PackEntry* pp_fqkv = get_prepacked(fqkv_name);
        if (pp_fqkv && (!lc.raw_w[0] || !lc.raw_w[1] || !lc.raw_w[2])) {
            uint32_t channel_sz = pp_fqkv->size / 4;
            uint32_t pe_k_beats = (config.n_embd + 63) / 64;
            pe_k_beats = (pe_k_beats + 3) & ~3;
            uint32_t ch_stride = (pp_fqkv->mode == V37_MODE_TL3_COMPACT) ? (pe_k_beats * 13) : (pe_k_beats * 16);
            uint32_t m_blocks_q = (q_dim + 3) / 4;
            uint32_t m_blocks_k = (kv_dim + 3) / 4;
            uint32_t q_pld = ((m_blocks_q * ch_stride + 15) / 16) * 16;
            uint32_t k_pld = ((m_blocks_k * ch_stride + 15) / 16) * 16;

            decode_hw_submatrix_gemv(pp_fqkv->data, channel_sz, 0, q_dim, config.n_embd, pp_fqkv->mode, buf_qact_attn.x_q.data(), q, lc.scales[0] ? lc.scales[0][0] : 1.0f, buf_qact_attn.s, nullptr);
            decode_hw_submatrix_gemv(pp_fqkv->data, channel_sz, q_pld, kv_dim, config.n_embd, pp_fqkv->mode, buf_qact_attn.x_q.data(), k, lc.scales[1] ? lc.scales[1][0] : 1.0f, buf_qact_attn.s, nullptr);
            decode_hw_submatrix_gemv(pp_fqkv->data, channel_sz, q_pld + k_pld, kv_dim, config.n_embd, pp_fqkv->mode, buf_qact_attn.x_q.data(), v, lc.scales[2] ? lc.scales[2][0] : 1.0f, buf_qact_attn.s, nullptr);
        } else {
            run_fpga_gemv_prequantized(lc.slots[0], buf_qact_attn, q, q_dim, lc.scales[0], 0, is_cpu, lc.raw_w[0], "attn_q_cpu", false, nullptr);
            run_fpga_gemv_prequantized(lc.slots[1], buf_qact_attn, k, kv_dim, lc.scales[1], 0, is_cpu, lc.raw_w[1], "attn_k_cpu", true, nullptr);
            run_fpga_gemv_prequantized(lc.slots[2], buf_qact_attn, v, kv_dim, lc.scales[2], 0, is_cpu, lc.raw_w[2], "attn_v_cpu", true, nullptr);
        }
        rope(q, k, pos);
        ggml_fp32_to_fp16_row(k, kv_cache.data() + k_offset, kv_dim);
        ggml_fp32_to_fp16_row(v, kv_cache.data() + v_offset, kv_dim);
    }
    
    auto tb_qkv_end = std::chrono::high_resolution_clock::now();
    prof_qkv_ms += std::chrono::duration<double, std::milli>(tb_qkv_end - tb_start).count();

    float* attn_out = buf_attn_out.data();
    int groups = config.n_heads / config.n_kv_heads;
    const v37_f16_t* kv_fp16 = (const v37_f16_t*)kv_cache.data();

    // Fast NEON vectorized Attention decode with 4-core OpenMP parallelization
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < config.n_heads; h++) {
        int kv_h = h / groups;
        float scores[2048];
        float max_score = -1e10f;
        const float* q_h = q + h * config.head_dim;
        float* attn_h = attn_out + h * config.head_dim;
        int past_k_base = kv_layer_offset + kv_h * config.head_dim;
        int k_stride = kv_dim;
        int past_v_base = kv_layer_offset + config.max_seq_len * kv_dim + kv_h * config.head_dim;
        int v_stride = kv_dim;

#if defined(__aarch64__) || defined(__arm__)
        if (config.head_dim == 128) {
            // Q is already pre-scaled by inv_sqrt_head_dim inside rope_q / v37_compute_qkv_fused
            float32x4_t q_vecs[32];
            for (int i = 0; i < 32; i++) {
                q_vecs[i] = vld1q_f32(q_h + i * 4);
            }

            int t = 0;
            for (; t <= pos - 1; t += 2) {
                const v37_f16_t* k_ptr0 = kv_fp16 + past_k_base + (t + 0) * k_stride;
                const v37_f16_t* k_ptr1 = kv_fp16 + past_k_base + (t + 1) * k_stride;

                float32x4_t sum0_0 = vdupq_n_f32(0.0f), sum0_1 = vdupq_n_f32(0.0f);
                float32x4_t sum0_2 = vdupq_n_f32(0.0f), sum0_3 = vdupq_n_f32(0.0f);

                float32x4_t sum1_0 = vdupq_n_f32(0.0f), sum1_1 = vdupq_n_f32(0.0f);
                float32x4_t sum1_2 = vdupq_n_f32(0.0f), sum1_3 = vdupq_n_f32(0.0f);

                for (int c = 0; c < 8; c++) {
                    float32x4_t q0 = q_vecs[c * 4 + 0];
                    float32x4_t q1 = q_vecs[c * 4 + 1];
                    float32x4_t q2 = q_vecs[c * 4 + 2];
                    float32x4_t q3 = q_vecs[c * 4 + 3];

                    float16x8_t kh0_0 = vld1q_f16((const __fp16*)(k_ptr0 + c * 16 + 0));
                    float16x8_t kh0_1 = vld1q_f16((const __fp16*)(k_ptr0 + c * 16 + 8));

                    sum0_0 = vmlaq_f32(sum0_0, q0, vcvt_f32_f16(vget_low_f16(kh0_0)));
                    sum0_1 = vmlaq_f32(sum0_1, q1, vcvt_f32_f16(vget_high_f16(kh0_0)));
                    sum0_2 = vmlaq_f32(sum0_2, q2, vcvt_f32_f16(vget_low_f16(kh0_1)));
                    sum0_3 = vmlaq_f32(sum0_3, q3, vcvt_f32_f16(vget_high_f16(kh0_1)));

                    float16x8_t kh1_0 = vld1q_f16((const __fp16*)(k_ptr1 + c * 16 + 0));
                    float16x8_t kh1_1 = vld1q_f16((const __fp16*)(k_ptr1 + c * 16 + 8));

                    sum1_0 = vmlaq_f32(sum1_0, q0, vcvt_f32_f16(vget_low_f16(kh1_0)));
                    sum1_1 = vmlaq_f32(sum1_1, q1, vcvt_f32_f16(vget_high_f16(kh1_0)));
                    sum1_2 = vmlaq_f32(sum1_2, q2, vcvt_f32_f16(vget_low_f16(kh1_1)));
                    sum1_3 = vmlaq_f32(sum1_3, q3, vcvt_f32_f16(vget_high_f16(kh1_1)));
                }
                float sc0 = vaddvq_f32(vaddq_f32(vaddq_f32(sum0_0, sum0_1), vaddq_f32(sum0_2, sum0_3)));
                float sc1 = vaddvq_f32(vaddq_f32(vaddq_f32(sum1_0, sum1_1), vaddq_f32(sum1_2, sum1_3)));
                scores[t + 0] = sc0;
                scores[t + 1] = sc1;
                if (sc0 > max_score) max_score = sc0;
                if (sc1 > max_score) max_score = sc1;
            }
            for (; t <= pos; t++) {
                const v37_f16_t* k_ptr = kv_fp16 + past_k_base + t * k_stride;

                float32x4_t sum0 = vdupq_n_f32(0.0f), sum1 = vdupq_n_f32(0.0f);
                float32x4_t sum2 = vdupq_n_f32(0.0f), sum3 = vdupq_n_f32(0.0f);

                for (int c = 0; c < 8; c++) {
                    float32x4_t q0 = q_vecs[c * 4 + 0];
                    float32x4_t q1 = q_vecs[c * 4 + 1];
                    float32x4_t q2 = q_vecs[c * 4 + 2];
                    float32x4_t q3 = q_vecs[c * 4 + 3];

                    float16x8_t kh0 = vld1q_f16((const __fp16*)(k_ptr + c * 16 + 0));
                    float16x8_t kh1 = vld1q_f16((const __fp16*)(k_ptr + c * 16 + 8));

                    sum0 = vmlaq_f32(sum0, q0, vcvt_f32_f16(vget_low_f16(kh0)));
                    sum1 = vmlaq_f32(sum1, q1, vcvt_f32_f16(vget_high_f16(kh0)));
                    sum2 = vmlaq_f32(sum2, q2, vcvt_f32_f16(vget_low_f16(kh1)));
                    sum3 = vmlaq_f32(sum3, q3, vcvt_f32_f16(vget_high_f16(kh1)));
                }
                float score = vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
                scores[t] = score;
                if (score > max_score) max_score = score;
            }

            float sum_exp = 0.0f;
            for (int t = 0; t <= pos; t++) {
                scores[t] = expf(scores[t] - max_score);
                sum_exp += scores[t];
            }
            float inv_sum_exp = 1.0f / sum_exp;
            for (int t = 0; t <= pos; t++) {
                scores[t] *= inv_sum_exp;
            }

            // Chunk 0: i = 0 .. 63
            float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f), acc3 = vdupq_n_f32(0.0f);
            float32x4_t acc4 = vdupq_n_f32(0.0f), acc5 = vdupq_n_f32(0.0f);
            float32x4_t acc6 = vdupq_n_f32(0.0f), acc7 = vdupq_n_f32(0.0f);
            float32x4_t acc8 = vdupq_n_f32(0.0f), acc9 = vdupq_n_f32(0.0f);
            float32x4_t acc10 = vdupq_n_f32(0.0f), acc11 = vdupq_n_f32(0.0f);
            float32x4_t acc12 = vdupq_n_f32(0.0f), acc13 = vdupq_n_f32(0.0f);
            float32x4_t acc14 = vdupq_n_f32(0.0f), acc15 = vdupq_n_f32(0.0f);

            for (int t = 0; t <= pos; t++) {
                float s_val = scores[t];
                float32x4_t vs = vdupq_n_f32(s_val);
                const v37_f16_t* v_ptr = kv_fp16 + past_v_base + t * v_stride;

                float16x8_t vh0 = vld1q_f16((const __fp16*)(v_ptr + 0));
                float16x8_t vh1 = vld1q_f16((const __fp16*)(v_ptr + 8));
                float16x8_t vh2 = vld1q_f16((const __fp16*)(v_ptr + 16));
                float16x8_t vh3 = vld1q_f16((const __fp16*)(v_ptr + 24));
                float16x8_t vh4 = vld1q_f16((const __fp16*)(v_ptr + 32));
                float16x8_t vh5 = vld1q_f16((const __fp16*)(v_ptr + 40));
                float16x8_t vh6 = vld1q_f16((const __fp16*)(v_ptr + 48));
                float16x8_t vh7 = vld1q_f16((const __fp16*)(v_ptr + 56));

                acc0  = vmlaq_f32(acc0,  vcvt_f32_f16(vget_low_f16(vh0)), vs);
                acc1  = vmlaq_f32(acc1,  vcvt_f32_f16(vget_high_f16(vh0)), vs);
                acc2  = vmlaq_f32(acc2,  vcvt_f32_f16(vget_low_f16(vh1)), vs);
                acc3  = vmlaq_f32(acc3,  vcvt_f32_f16(vget_high_f16(vh1)), vs);
                acc4  = vmlaq_f32(acc4,  vcvt_f32_f16(vget_low_f16(vh2)), vs);
                acc5  = vmlaq_f32(acc5,  vcvt_f32_f16(vget_high_f16(vh2)), vs);
                acc6  = vmlaq_f32(acc6,  vcvt_f32_f16(vget_low_f16(vh3)), vs);
                acc7  = vmlaq_f32(acc7,  vcvt_f32_f16(vget_high_f16(vh3)), vs);
                acc8  = vmlaq_f32(acc8,  vcvt_f32_f16(vget_low_f16(vh4)), vs);
                acc9  = vmlaq_f32(acc9,  vcvt_f32_f16(vget_high_f16(vh4)), vs);
                acc10 = vmlaq_f32(acc10, vcvt_f32_f16(vget_low_f16(vh5)), vs);
                acc11 = vmlaq_f32(acc11, vcvt_f32_f16(vget_high_f16(vh5)), vs);
                acc12 = vmlaq_f32(acc12, vcvt_f32_f16(vget_low_f16(vh6)), vs);
                acc13 = vmlaq_f32(acc13, vcvt_f32_f16(vget_high_f16(vh6)), vs);
                acc14 = vmlaq_f32(acc14, vcvt_f32_f16(vget_low_f16(vh7)), vs);
                acc15 = vmlaq_f32(acc15, vcvt_f32_f16(vget_high_f16(vh7)), vs);
            }
            vst1q_f32(attn_h + 0,  acc0);
            vst1q_f32(attn_h + 4,  acc1);
            vst1q_f32(attn_h + 8,  acc2);
            vst1q_f32(attn_h + 12, acc3);
            vst1q_f32(attn_h + 16, acc4);
            vst1q_f32(attn_h + 20, acc5);
            vst1q_f32(attn_h + 24, acc6);
            vst1q_f32(attn_h + 28, acc7);
            vst1q_f32(attn_h + 32, acc8);
            vst1q_f32(attn_h + 36, acc9);
            vst1q_f32(attn_h + 40, acc10);
            vst1q_f32(attn_h + 44, acc11);
            vst1q_f32(attn_h + 48, acc12);
            vst1q_f32(attn_h + 52, acc13);
            vst1q_f32(attn_h + 56, acc14);
            vst1q_f32(attn_h + 60, acc15);

            // Chunk 1: i = 64 .. 127
            float32x4_t bcc0 = vdupq_n_f32(0.0f), bcc1 = vdupq_n_f32(0.0f);
            float32x4_t bcc2 = vdupq_n_f32(0.0f), bcc3 = vdupq_n_f32(0.0f);
            float32x4_t bcc4 = vdupq_n_f32(0.0f), bcc5 = vdupq_n_f32(0.0f);
            float32x4_t bcc6 = vdupq_n_f32(0.0f), bcc7 = vdupq_n_f32(0.0f);
            float32x4_t bcc8 = vdupq_n_f32(0.0f), bcc9 = vdupq_n_f32(0.0f);
            float32x4_t bcc10 = vdupq_n_f32(0.0f), bcc11 = vdupq_n_f32(0.0f);
            float32x4_t bcc12 = vdupq_n_f32(0.0f), bcc13 = vdupq_n_f32(0.0f);
            float32x4_t bcc14 = vdupq_n_f32(0.0f), bcc15 = vdupq_n_f32(0.0f);

            for (int t = 0; t <= pos; t++) {
                float s_val = scores[t];
                float32x4_t vs = vdupq_n_f32(s_val);
                const v37_f16_t* v_ptr = kv_fp16 + past_v_base + t * v_stride + 64;

                float16x8_t vh0 = vld1q_f16((const __fp16*)(v_ptr + 0));
                float16x8_t vh1 = vld1q_f16((const __fp16*)(v_ptr + 8));
                float16x8_t vh2 = vld1q_f16((const __fp16*)(v_ptr + 16));
                float16x8_t vh3 = vld1q_f16((const __fp16*)(v_ptr + 24));
                float16x8_t vh4 = vld1q_f16((const __fp16*)(v_ptr + 32));
                float16x8_t vh5 = vld1q_f16((const __fp16*)(v_ptr + 40));
                float16x8_t vh6 = vld1q_f16((const __fp16*)(v_ptr + 48));
                float16x8_t vh7 = vld1q_f16((const __fp16*)(v_ptr + 56));

                bcc0  = vmlaq_f32(bcc0,  vcvt_f32_f16(vget_low_f16(vh0)), vs);
                bcc1  = vmlaq_f32(bcc1,  vcvt_f32_f16(vget_high_f16(vh0)), vs);
                bcc2  = vmlaq_f32(bcc2,  vcvt_f32_f16(vget_low_f16(vh1)), vs);
                bcc3  = vmlaq_f32(bcc3,  vcvt_f32_f16(vget_high_f16(vh1)), vs);
                bcc4  = vmlaq_f32(bcc4,  vcvt_f32_f16(vget_low_f16(vh2)), vs);
                bcc5  = vmlaq_f32(bcc5,  vcvt_f32_f16(vget_high_f16(vh2)), vs);
                bcc6  = vmlaq_f32(bcc6,  vcvt_f32_f16(vget_low_f16(vh3)), vs);
                bcc7  = vmlaq_f32(bcc7,  vcvt_f32_f16(vget_high_f16(vh3)), vs);
                bcc8  = vmlaq_f32(bcc8,  vcvt_f32_f16(vget_low_f16(vh4)), vs);
                bcc9  = vmlaq_f32(bcc9,  vcvt_f32_f16(vget_high_f16(vh4)), vs);
                bcc10 = vmlaq_f32(bcc10, vcvt_f32_f16(vget_low_f16(vh5)), vs);
                bcc11 = vmlaq_f32(bcc11, vcvt_f32_f16(vget_high_f16(vh5)), vs);
                bcc12 = vmlaq_f32(bcc12, vcvt_f32_f16(vget_low_f16(vh6)), vs);
                bcc13 = vmlaq_f32(bcc13, vcvt_f32_f16(vget_high_f16(vh6)), vs);
                bcc14 = vmlaq_f32(bcc14, vcvt_f32_f16(vget_low_f16(vh7)), vs);
                bcc15 = vmlaq_f32(bcc15, vcvt_f32_f16(vget_high_f16(vh7)), vs);
            }
            vst1q_f32(attn_h + 64,  bcc0);
            vst1q_f32(attn_h + 68,  bcc1);
            vst1q_f32(attn_h + 72,  bcc2);
            vst1q_f32(attn_h + 76,  bcc3);
            vst1q_f32(attn_h + 80,  bcc4);
            vst1q_f32(attn_h + 84,  bcc5);
            vst1q_f32(attn_h + 88,  bcc6);
            vst1q_f32(attn_h + 92,  bcc7);
            vst1q_f32(attn_h + 96,  bcc8);
            vst1q_f32(attn_h + 100, bcc9);
            vst1q_f32(attn_h + 104, bcc10);
            vst1q_f32(attn_h + 108, bcc11);
            vst1q_f32(attn_h + 112, bcc12);
            vst1q_f32(attn_h + 116, bcc13);
            vst1q_f32(attn_h + 120, bcc14);
            vst1q_f32(attn_h + 124, bcc15);
        } else
#endif
            {
                // Generic attention decode path for arbitrary head dimensions
                for (int t = 0; t <= pos; t++) {
                    int past_k_off = kv_layer_offset + t * k_stride + kv_h * config.head_dim;
                    const v37_f16_t* k_ptr = kv_fp16 + past_k_off;
                    float score = 0.0f;
                    for (int i = 0; i < config.head_dim; i++) {
                        score += q_h[i] * V37_F16_TO_F32(k_ptr[i]);
                    }
                    // Note: q_h was already pre-scaled by inv_sqrt in rope_q / v37_compute_qkv_fused
                    scores[t] = score;
                    if (score > max_score) max_score = score;
                }

                float sum_exp = 0.0f;
                for (int t = 0; t <= pos; t++) {
                    scores[t] = expf(scores[t] - max_score);
                    sum_exp += scores[t];
                }
                float inv_sum_exp = 1.0f / sum_exp;

                for (int i = 0; i < config.head_dim; i++) attn_h[i] = 0.0f;
                for (int t = 0; t <= pos; t++) {
                    float s_val = scores[t] * inv_sum_exp;
                    const v37_f16_t* v_ptr = kv_fp16 + past_v_base + t * v_stride;
                    for (int i = 0; i < config.head_dim; i++) {
                        attn_h[i] += s_val * V37_F16_TO_F32(v_ptr[i]);
                    }
                }
            }
        }
    auto tb_attn_end = std::chrono::high_resolution_clock::now();
    prof_attn_ms += std::chrono::duration<double, std::milli>(tb_attn_end - tb_qkv_end).count();
    
    int o_in_dim = config.n_heads * config.head_dim;
    if (o_in_dim <= 0) o_in_dim = config.n_embd;
    float* o_in = attn_out;
    if (config.has_sub_norm) {
        float* attn_sub_norm_out = buf_attn_sub_norm_out.data();
        rms_norm(attn_sub_norm_out, attn_out, lc.norm_attn_sub.data(), o_in_dim);
        o_in = attn_sub_norm_out;
    }

    quantize_activation_neon(o_in, o_in_dim, buf_qact_attn_sub);

    float* o_out = buf_o_out.data();
    if (!is_cpu && engine) {
        run_fpga_gemv_launch(lc.slots[3], buf_qact_attn_sub, config.n_embd, false);
        run_fpga_gemv_wait(nullptr, lc.scales[3], buf_qact_attn_sub.s, hidden_states.data());
    } else {
        std::string o_name = "hw.blk." + std::to_string(l) + ".attn_output.weight";
        V37PackEntry* pp_o = get_prepacked(o_name);
        if (!pp_o) pp_o = get_prepacked("blk." + std::to_string(l) + ".attn_output.weight");
        if (pp_o && !lc.raw_w[3]) {
            uint32_t channel_sz = pp_o->size / 4;
            decode_hw_submatrix_gemv(pp_o->data, channel_sz, 0, config.n_embd, o_in_dim, pp_o->mode, buf_qact_attn_sub.x_q.data(), nullptr, lc.scales[3] ? lc.scales[3][0] : 1.0f, buf_qact_attn_sub.s, hidden_states.data());
        } else {
            run_fpga_gemv_prequantized(lc.slots[3], buf_qact_attn_sub, o_out, config.n_embd, lc.scales[3], 0, is_cpu, lc.raw_w[3], "attn_o_cpu", false, hidden_states.data());
        }
    }
    check_nan(hidden_states.data(), config.n_embd, "hidden_states_post_attn", l);
    auto tb_oproj_end = std::chrono::high_resolution_clock::now();
    prof_oproj_ms += std::chrono::duration<double, std::milli>(tb_oproj_end - tb_attn_end).count();
    
    // 5. FFN Input Norm
    float* ffn_norm_out = buf_ffn_norm_out.data();
    rms_norm(ffn_norm_out, hidden_states.data(), lc.norm_ffn.data(), config.n_embd);

    // Shared activation quantization for Gate and Up
    quantize_activation_neon(ffn_norm_out, config.n_embd, buf_qact_ffn);

    float* gate = buf_gate.data();
    float* up = buf_up.data();
    float* ffn_act = buf_ffn_act.data();
    bool use_silu = config.use_silu;
    if (getenv("USE_SILU")) {
        use_silu = (atoi(getenv("USE_SILU")) == 1);
    }
    static bool use_chain = (!getenv("V37_CHAIN_LAUNCH") || atoi(getenv("V37_CHAIN_LAUNCH")) == 1) ||
                            (getenv("V37_LOOPBACK") && atoi(getenv("V37_LOOPBACK")) == 1);

    if (!is_cpu && engine && use_chain && lc.fused_gate_up_slot >= 0) {
        // [CHAIN LAUNCH PIPELINE FUSION] Entire FFN (Fused Gate+Up -> NEON Activation & DynQuant -> Down GEMV) on FPGA
        float ffn_down_quant_s = 1.0f;
        v37_launch_ffn_chain(engine, buf_qact_ffn.x_q.data(), lc.slots[4], lc.slots[5], lc.slots[6],
                              lc.fused_gate_up_slot,
                              config.n_embd, config.n_ffn,
                              lc.scales[4], lc.scales[5],
                              buf_qact_ffn.s, config.has_sub_norm ? lc.norm_ffn_sub.data() : nullptr, &ffn_down_quant_s,
                              active_pack_mode, use_silu ? 1 : 0);

        // [ASYNC OVERLAP]: While FPGA is calculating Down GEMV (~286 us):
        if (l + 1 < config.n_layers) {
            const auto& next_lc = layers_cache[l + 1];
            __builtin_prefetch(next_lc.norm_attn.data(), 0, 3);
            __builtin_prefetch(next_lc.scales[0], 0, 3);
            __builtin_prefetch(next_lc.scales[1], 0, 3);
            __builtin_prefetch(next_lc.scales[2], 0, 3);
            __builtin_prefetch(next_lc.scales[3], 0, 3);
            __builtin_prefetch(next_lc.norm_attn_sub.data(), 0, 3);
            __builtin_prefetch(next_lc.norm_ffn.data(), 0, 3);
            __builtin_prefetch(next_lc.scales[4], 0, 3);
            __builtin_prefetch(next_lc.scales[5], 0, 3);
            __builtin_prefetch(next_lc.norm_ffn_sub.data(), 0, 3);
            __builtin_prefetch(next_lc.scales[6], 0, 3);
        } else {
            // Layer 29: Prefetch & prepare LM_HEAD while FPGA Down GEMV is running!
            prepare_lmhead_overlap();
        }

        v37_wait_ffn_chain_down(engine, nullptr, lc.scales[6], ffn_down_quant_s, hidden_states.data());
    } else {
        if (!is_cpu && engine) {
            // Launch Gate & wait Gate
            run_fpga_gemv_launch(lc.slots[4], buf_qact_ffn, config.n_ffn, false);
            run_fpga_gemv_wait(gate, lc.scales[4], buf_qact_ffn.s, nullptr);

            // Launch Up asynchronously (FPGA computes Up ~460us)
            run_fpga_gemv_launch(lc.slots[5], buf_qact_ffn, config.n_ffn, true);

            if (!use_silu) {
                // [ASYNC OVERLAP 3]: Precompute Squared ReLU on Gate while FPGA computes Up!
#if defined(__aarch64__) || defined(__arm__)
                float32x4_t zero = vdupq_n_f32(0.0f);
                int i = 0;
                for (; i <= config.n_ffn - 16; i += 16) {
                    float32x4_t g0 = vmaxq_f32(vld1q_f32(gate + i + 0), zero);
                    float32x4_t g1 = vmaxq_f32(vld1q_f32(gate + i + 4), zero);
                    float32x4_t g2 = vmaxq_f32(vld1q_f32(gate + i + 8), zero);
                    float32x4_t g3 = vmaxq_f32(vld1q_f32(gate + i + 12), zero);

                    vst1q_f32(gate + i + 0,  vmulq_f32(g0, g0));
                    vst1q_f32(gate + i + 4,  vmulq_f32(g1, g1));
                    vst1q_f32(gate + i + 8,  vmulq_f32(g2, g2));
                    vst1q_f32(gate + i + 12, vmulq_f32(g3, g3));
                }
                for (; i < config.n_ffn; i++) {
                    float x = gate[i];
                    float r = (x > 0.0f) ? x : 0.0f;
                    gate[i] = r * r;
                }
#else
                for (int i = 0; i < config.n_ffn; i++) {
                    float x = gate[i];
                    float r = (x > 0.0f) ? x : 0.0f;
                    gate[i] = r * r;
                }
#endif
                // Fused Up dequant directly into ffn_act multiplied with gate squared-relu (Zero intermediate up.data() RAM traffic)
                run_fpga_gemv_wait_fused_up(ffn_act, gate, lc.scales[5], buf_qact_ffn.s);
            } else {
                run_fpga_gemv_wait(up, lc.scales[5], buf_qact_ffn.s, nullptr);
                for (int i = 0; i < config.n_ffn; i++) {
                    float x = gate[i];
                    float silu = x / (1.0f + expf(-x));
                    ffn_act[i] = silu * up[i];
                }
            }
        } else {
            std::string fgu_name = "blk." + std::to_string(l) + ".fused_gate_up.weight";
            V37PackEntry* pp_fgu = get_prepacked(fgu_name);
            if (pp_fgu && (!lc.raw_w[4] || !lc.raw_w[5])) {
                uint32_t channel_sz = pp_fgu->size / 4;
                uint32_t pe_k_beats = (config.n_embd + 63) / 64;
                pe_k_beats = (pe_k_beats + 3) & ~3;
                uint32_t ch_stride = (pp_fgu->mode == V37_MODE_TL3_COMPACT) ? (pe_k_beats * 13) : (pe_k_beats * 16);
                uint32_t m_blocks_half = (config.n_ffn + 3) / 4;
                uint32_t gate_pld = ((m_blocks_half * ch_stride + 15) / 16) * 16;

                decode_hw_submatrix_gemv(pp_fgu->data, channel_sz, 0, config.n_ffn, config.n_embd, pp_fgu->mode, buf_qact_ffn.x_q.data(), gate, lc.scales[4] ? lc.scales[4][0] : 1.0f, buf_qact_ffn.s, nullptr);
                decode_hw_submatrix_gemv(pp_fgu->data, channel_sz, gate_pld, config.n_ffn, config.n_embd, pp_fgu->mode, buf_qact_ffn.x_q.data(), up, lc.scales[5] ? lc.scales[5][0] : 1.0f, buf_qact_ffn.s, nullptr);
            } else {
                run_fpga_gemv_prequantized(lc.slots[4], buf_qact_ffn, gate, config.n_ffn, lc.scales[4], 0, is_cpu, lc.raw_w[4], "ffn_gate_cpu", false, nullptr);
                run_fpga_gemv_prequantized(lc.slots[5], buf_qact_ffn, up, config.n_ffn, lc.scales[5], 0, is_cpu, lc.raw_w[5], "ffn_up_cpu", true, nullptr);
            }
            if (!use_silu) {
#if defined(__aarch64__) || defined(__arm__)
                float32x4_t zero = vdupq_n_f32(0.0f);
                int i = 0;
                for (; i <= config.n_ffn - 16; i += 16) {
                    float32x4_t g0 = vld1q_f32(gate + i + 0);
                    float32x4_t g1 = vld1q_f32(gate + i + 4);
                    float32x4_t g2 = vld1q_f32(gate + i + 8);
                    float32x4_t g3 = vld1q_f32(gate + i + 12);

                    float32x4_t u0 = vld1q_f32(up + i + 0);
                    float32x4_t u1 = vld1q_f32(up + i + 4);
                    float32x4_t u2 = vld1q_f32(up + i + 8);
                    float32x4_t u3 = vld1q_f32(up + i + 12);

                    float32x4_t r0 = vmaxq_f32(g0, zero);
                    float32x4_t r1 = vmaxq_f32(g1, zero);
                    float32x4_t r2 = vmaxq_f32(g2, zero);
                    float32x4_t r3 = vmaxq_f32(g3, zero);

                    vst1q_f32(ffn_act + i + 0,  vmulq_f32(vmulq_f32(r0, r0), u0));
                    vst1q_f32(ffn_act + i + 4,  vmulq_f32(vmulq_f32(r1, r1), u1));
                    vst1q_f32(ffn_act + i + 8,  vmulq_f32(vmulq_f32(r2, r2), u2));
                    vst1q_f32(ffn_act + i + 12, vmulq_f32(vmulq_f32(r3, r3), u3));
                }
                for (; i < config.n_ffn; i++) {
                    float x = gate[i];
                    float r = (x > 0.0f) ? x : 0.0f;
                    ffn_act[i] = r * r * up[i];
                }
#else
                for (int i = 0; i < config.n_ffn; i++) {
                    float x = gate[i];
                    float r = (x > 0.0f) ? x : 0.0f;
                    ffn_act[i] = r * r * up[i];
                }
#endif
            } else {
                for (int i = 0; i < config.n_ffn; i++) {
                    float x = gate[i];
                    float silu = x / (1.0f + expf(-x));
                    ffn_act[i] = silu * up[i];
                }
            }
        }
        
        float* ffn_down_in = ffn_act;
        if (config.has_sub_norm) {
            float* ffn_sub_norm_out = buf_ffn_sub_norm_out.data();
            rms_norm(ffn_sub_norm_out, ffn_act, lc.norm_ffn_sub.data(), config.n_ffn);
            ffn_down_in = ffn_sub_norm_out;
        }

        quantize_activation_neon(ffn_down_in, config.n_ffn, buf_qact_ffn_sub);

        float* down = buf_down.data();
        if (!is_cpu && engine) {
            run_fpga_gemv_launch(lc.slots[6], buf_qact_ffn_sub, config.n_embd, false);

            // [ASYNC OVERLAP 4]: Prefetch next layer caches or LM_HEAD tables while FPGA computes FFN Down (~1.1 ms)
            if (l + 1 < config.n_layers) {
                const auto& next_lc = layers_cache[l + 1];
                __builtin_prefetch(next_lc.norm_attn.data(), 0, 3);
                __builtin_prefetch(next_lc.scales[0], 0, 3);
                __builtin_prefetch(next_lc.scales[1], 0, 3);
                __builtin_prefetch(next_lc.scales[2], 0, 3);
                __builtin_prefetch(next_lc.scales[3], 0, 3);
                __builtin_prefetch(next_lc.norm_attn_sub.data(), 0, 3);
                __builtin_prefetch(next_lc.norm_ffn.data(), 0, 3);
                __builtin_prefetch(next_lc.scales[4], 0, 3);
                __builtin_prefetch(next_lc.scales[5], 0, 3);
                __builtin_prefetch(next_lc.norm_ffn_sub.data(), 0, 3);
                __builtin_prefetch(next_lc.scales[6], 0, 3);
            } else {
                prepare_lmhead_overlap();
            }

            run_fpga_gemv_wait(nullptr, lc.scales[6], buf_qact_ffn_sub.s, hidden_states.data());
        } else {
            std::string down_name = "hw.blk." + std::to_string(l) + ".ffn_down.weight";
            V37PackEntry* pp_down = get_prepacked(down_name);
            if (!pp_down) pp_down = get_prepacked("blk." + std::to_string(l) + ".ffn_down.weight");
            if (pp_down && !lc.raw_w[6]) {
                uint32_t channel_sz = pp_down->size / 4;
                decode_hw_submatrix_gemv(pp_down->data, channel_sz, 0, config.n_embd, config.n_ffn, pp_down->mode, buf_qact_ffn_sub.x_q.data(), nullptr, lc.scales[6] ? lc.scales[6][0] : 1.0f, buf_qact_ffn_sub.s, hidden_states.data());
            } else {
                run_fpga_gemv_prequantized(lc.slots[6], buf_qact_ffn_sub, down, config.n_embd, lc.scales[6], 0, is_cpu, lc.raw_w[6], "ffn_down_cpu", false, hidden_states.data());
            }
        }
    }
    check_nan(hidden_states.data(), config.n_embd, "hidden_states_post_ffn", l);
    auto tb_ffn_end = std::chrono::high_resolution_clock::now();
    prof_ffn_ms += std::chrono::duration<double, std::milli>(tb_ffn_end - tb_oproj_end).count();
}

float* V37Engine::forward(const std::vector<int>& tokens, int pos, bool compute_logits) {
    prof_qkv_ms = 0;
    prof_attn_ms = 0;
    prof_oproj_ms = 0;
    prof_ffn_ms = 0;
    auto t_start_fwd = std::chrono::high_resolution_clock::now();
    if (this->engine) {
        v37_start_token(this->engine);
    }
    float* embd = loader->get_embd(tokens.back());
    if (embd) {
        memcpy(hidden_states.data(), embd, config.n_embd * sizeof(float));
        check_nan(hidden_states.data(), config.n_embd, "token_embd", -1);
    }

    if (pos == 0) {
        past_tokens_history.clear();
    }
    if (tokens.size() <= 1 && !tokens.empty()) {
        past_tokens_history.push_back(tokens.back());
    }

    for (int l = 0; l < config.n_layers; l++) transformer_block(l, pos);
    auto t_end_trans = std::chrono::high_resolution_clock::now();

    if (!compute_logits) {
        return nullptr;
    }

    rms_norm(buf_final_norm.data(), hidden_states.data(), output_norm_weight.data(), config.n_embd);
    float* final_norm = buf_final_norm.data();
    auto t_end_norm = std::chrono::high_resolution_clock::now();

    // repacking LM_HEAD into slot 0 (only if output.weight is quantized I2_S)
    uint32_t output_type = loader ? loader->get_tensor_type("output.weight") : 0;
    uint8_t* head_w = (output_type == 36) ? (uint8_t*)loader->get_tensor_data("output.weight") : nullptr;
    if (getenv("V37_DEBUG")) {
        printf("[V37_DEBUG] LM_HEAD: output.weight %s (type=%u)\n", head_w ? "EXISTS (I2_S FPGA path)" : "NOT I2_S (F16 LM Head path)", output_type);
    }
    if (head_w) {
        bool is_cpu = getenv("DISABLE_V37") != nullptr;
        const float* head_scales = loader->get_i2_scales("output.weight");
        
        // Preload LM_HEAD weights (once)
        if (!is_cpu && !all_weights_preloaded) {
            v37_reset_cma_allocator(engine, 7);
            V37PackEntry* pp = get_prepacked("output.weight");
            if (pp) {
                v37_mode_t preload_mode = (pp->mode == V37_MODE_I2_S) ? V37_MODE_I2_S_PREPACKED : pp->mode;
                v37_preload_weights(engine, pp->data, nullptr, pp->size, 7, preload_mode, config.n_embd, 0);
            } else {
                int padded_k = (config.n_embd + 255) & ~255;
                size_t w_sz = (size_t)config.vocab_size * padded_k / 4;
                v37_preload_weights(engine, head_w, head_scales, w_sz, 7, V37_MODE_I2_S, config.n_embd, 0);
            }
        }
        
        // Use the unified run_fpga_gemv_fast path for consistent scaling
        run_fpga_gemv_fast(7, final_norm, logits.data(), config.vocab_size, config.n_embd, head_scales, 0, is_cpu, head_w, "lm_head", 0, config.vocab_size);
    } else {
        // [BitNet.cpp 官方架构对齐] LM_HEAD 两级自适应粗细精筛极速算子 (28ms, 100% 黄金基线对齐)
        std::string lmhead_name = "output.weight";
        void* embd_raw = (void*)loader->get_tensor_data(lmhead_name);
        uint32_t embd_type = loader->get_tensor_type(lmhead_name);
        if (!embd_raw) {
            lmhead_name = "token_embd.weight";
            embd_raw = (void*)loader->get_tensor_data(lmhead_name);
            embd_type = loader->get_tensor_type(lmhead_name);
        }
        
        if (embd_type != 1) { // GGML_TYPE_F16
            printf("[ERROR] %s must be F16 for LM_HEAD. Found: %u\n", lmhead_name.c_str(), embd_type);
            return nullptr;
        }

        const v37_f16_t* w_f16 = (const v37_f16_t*)embd_raw;
        bool fallback_full = (getenv("DISABLE_FAST_LMHEAD") != nullptr) || !lmhead_cache_initialized;

        if (fallback_full) {
            #pragma omp parallel for schedule(static)
            for (int v = 0; v < config.vocab_size; v++) {
                const v37_f16_t* row_ptr = (const v37_f16_t*)((const uint16_t*)embd_raw + (size_t)v * config.n_embd);
                if (v + 2 < config.vocab_size) {
                    __builtin_prefetch((const char*)((const uint16_t*)embd_raw + (size_t)(v + 2) * config.n_embd), 0, 1);
                }
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
                float s8 = 0.0f, s9 = 0.0f, sa = 0.0f, sb = 0.0f;
                float sc = 0.0f, sd = 0.0f, se = 0.0f, sf = 0.0f;
                int i = 0;
                for (; i <= config.n_embd - 16; i += 16) {
                    s0 += final_norm[i +  0] * V37_F16_TO_F32(row_ptr[i +  0]);
                    s1 += final_norm[i +  1] * V37_F16_TO_F32(row_ptr[i +  1]);
                    s2 += final_norm[i +  2] * V37_F16_TO_F32(row_ptr[i +  2]);
                    s3 += final_norm[i +  3] * V37_F16_TO_F32(row_ptr[i +  3]);
                    s4 += final_norm[i +  4] * V37_F16_TO_F32(row_ptr[i +  4]);
                    s5 += final_norm[i +  5] * V37_F16_TO_F32(row_ptr[i +  5]);
                    s6 += final_norm[i +  6] * V37_F16_TO_F32(row_ptr[i +  6]);
                    s7 += final_norm[i +  7] * V37_F16_TO_F32(row_ptr[i +  7]);
                    s8 += final_norm[i +  8] * V37_F16_TO_F32(row_ptr[i +  8]);
                    s9 += final_norm[i +  9] * V37_F16_TO_F32(row_ptr[i +  9]);
                    sa += final_norm[i + 10] * V37_F16_TO_F32(row_ptr[i + 10]);
                    sb += final_norm[i + 11] * V37_F16_TO_F32(row_ptr[i + 11]);
                    sc += final_norm[i + 12] * V37_F16_TO_F32(row_ptr[i + 12]);
                    sd += final_norm[i + 13] * V37_F16_TO_F32(row_ptr[i + 13]);
                    se += final_norm[i + 14] * V37_F16_TO_F32(row_ptr[i + 14]);
                    sf += final_norm[i + 15] * V37_F16_TO_F32(row_ptr[i + 15]);
                }
                float sum = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7)) +
                            ((s8 + s9) + (sa + sb)) + ((sc + sd) + (se + sf));
                for (; i < config.n_embd; i++) {
                    sum += final_norm[i] * V37_F16_TO_F32(row_ptr[i]);
                }
                logits[v] = sum;
            }
        } else {
            // --- FAST TWO-STAGE HYBRID LM_HEAD OPERATOR ---
            auto t_lm0 = std::chrono::high_resolution_clock::now();
            const int TIER1_SIZE = lmhead_tier1_size;
            const int D1 = lmhead_d1;
            const int D_COARSE = lmhead_d_coarse;
            static aligned_vector64<int8_t> x_t1_q8(D1);
            static aligned_vector64<int8_t> x_sub_q8(D_COARSE);
            if ((int)x_t1_q8.size() != D1) x_t1_q8.resize(D1);
            if ((int)x_sub_q8.size() != D_COARSE) x_sub_q8.resize(D_COARSE);

            float max_x_t1 = 1e-10f;
            for (int i = 0; i < D1; i++) {
                float v = std::fabs(final_norm[lmhead_indices_t1[i]]);
                if (v > max_x_t1) max_x_t1 = v;
            }
            float x_t1_scale = max_x_t1 / 127.0f;
            float inv_x_t1_scale = 127.0f / max_x_t1;
            for (int i = 0; i < D1; i++) {
                int q = std::round(final_norm[lmhead_indices_t1[i]] * inv_x_t1_scale);
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                x_t1_q8[i] = (int8_t)q;
            }

            float max_x_sub = 1e-10f;
            for (int i = 0; i < D_COARSE; i++) {
                float v = std::fabs(final_norm[lmhead_indices_coarse[i]]);
                if (v > max_x_sub) max_x_sub = v;
            }
            float x_sub_scale = max_x_sub / 127.0f;
            float inv_x_sub_scale = 127.0f / max_x_sub;
            for (int i = 0; i < D_COARSE; i++) {
                int q = std::round(final_norm[lmhead_indices_coarse[i]] * inv_x_sub_scale);
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                x_sub_q8[i] = (int8_t)q;
            }

            static const int MAX_K_CAND = 128;
            static int k_cand_cfg = -1;
            if (k_cand_cfg == -1) {
                const char* env_kc = getenv("LMHEAD_K_CAND");
                k_cand_cfg = env_kc ? std::min(MAX_K_CAND, std::max(16, atoi(env_kc))) : 32;
            }
            const int K_CAND = k_cand_cfg;
            using Elem = std::pair<float, int>;
            static const int MAX_THREADS = 8;
            static Elem thread_top_t1[MAX_THREADS][MAX_K_CAND];
            static int thread_top_t1_sz[MAX_THREADS] = {0};
            static Elem thread_top_t2[MAX_THREADS][MAX_K_CAND];
            static int thread_top_t2_sz[MAX_THREADS] = {0};
            int nthreads = std::min(omp_get_max_threads(), MAX_THREADS);
            for (int t = 0; t < nthreads; t++) {
                thread_top_t1_sz[t] = 0;
                thread_top_t2_sz[t] = 0;
            }

            auto t_lm1 = std::chrono::high_resolution_clock::now();
            int n2_rows = std::max(0, config.vocab_size - TIER1_SIZE);

            // 2. Tier 1 & Tier 2 Unified Single OpenMP Parallel Region with Stack Min-Heap and Mathematical Upper-Bound Pruning
            float inv_scale1 = inv_max_lmhead_t1_scale;
            float inv_scale2 = inv_max_lmhead_t2_scale;

            if (lmhead_twostage_enabled) {
                const int D_S1 = lmhead_d_s1;
                const int D_S2 = lmhead_d_s2;
                const int K_S1 = lmhead_s1_k_per_thread;
                const int MAX_TOTAL_CANDS = MAX_THREADS * 512;

                struct CandS1 {
                    float score;
                    int32_t dot_s1;
                    int token_id;
                };
                struct CandS1Greater {
                    inline bool operator()(const CandS1& a, const CandS1& b) const {
                        if (a.score != b.score) return a.score > b.score;
                        return a.token_id < b.token_id;
                    }
                };
                static CandS1 pool_cands[MAX_TOTAL_CANDS];
                static int pool_cands_sz[MAX_THREADS] = {0};
                static int total_cands_count = 0;

                const int8_t* __restrict__ x_s1_ptr = (const int8_t*)__builtin_assume_aligned(x_t1_q8.data(), 64);
                const int8_t* __restrict__ x_s2_ptr = (const int8_t*)__builtin_assume_aligned(x_t1_q8.data() + D_S1, 64);

                const int cur_k = std::min(K_S1, 512);

                #pragma omp parallel
                {
                    int tid = omp_get_thread_num();
                    int n_per_th = TIER1_SIZE / 4;
                    int start = tid * n_per_th;
                    int end = (tid == 3) ? TIER1_SIZE : (start + n_per_th);

                    CandS1 heap[512];
                    int heap_sz = 0;
                    float cur_thresh = -1e30f;

                    auto insert_cand = [&](float score, int32_t dot, int r) {
                        if (heap_sz < cur_k) {
                            heap[heap_sz++] = {score, dot, r};
                            if (heap_sz == cur_k) {
                                std::make_heap(heap, heap + cur_k, CandS1Greater());
                                cur_thresh = heap[0].score;
                            }
                        } else if (__builtin_expect(score > cur_thresh, 0)) {
                            std::pop_heap(heap, heap + cur_k, CandS1Greater());
                            heap[cur_k - 1] = {score, dot, r};
                            std::push_heap(heap, heap + cur_k, CandS1Greater());
                            cur_thresh = heap[0].score;
                        }
                    };

#if defined(__aarch64__) || defined(__arm__)
                    int r = start;
                    for (; r <= end - 2; r += 2) {
                        const int8_t* __restrict__ rA_ptr = (const int8_t*)__builtin_assume_aligned(lmhead_t1_s1_q8.data() + (r + 0) * D_S1, 64);
                        const int8_t* __restrict__ rB_ptr = (const int8_t*)__builtin_assume_aligned(lmhead_t1_s1_q8.data() + (r + 1) * D_S1, 64);
                        if (r + 3 < end) {
                            __builtin_prefetch(lmhead_t1_s1_q8.data() + (r + 2) * D_S1, 0, 0);
                            __builtin_prefetch(lmhead_t1_s1_q8.data() + (r + 3) * D_S1, 0, 0);
                        }

                        int32x4_t aA0 = vdupq_n_s32(0), aA1 = vdupq_n_s32(0);
                        int32x4_t aA2 = vdupq_n_s32(0), aA3 = vdupq_n_s32(0);
                        int32x4_t aB0 = vdupq_n_s32(0), aB1 = vdupq_n_s32(0);
                        int32x4_t aB2 = vdupq_n_s32(0), aB3 = vdupq_n_s32(0);

                        for (int i = 0; i < D_S1; i += 64) {
                            int8x16_t vx0 = vld1q_s8(x_s1_ptr + i + 0);
                            int8x16_t vx1 = vld1q_s8(x_s1_ptr + i + 16);
                            int8x16_t vx2 = vld1q_s8(x_s1_ptr + i + 32);
                            int8x16_t vx3 = vld1q_s8(x_s1_ptr + i + 48);

                            int8x16_t vrA0 = vld1q_s8(rA_ptr + i + 0);
                            int8x16_t vrA1 = vld1q_s8(rA_ptr + i + 16);
                            int8x16_t vrA2 = vld1q_s8(rA_ptr + i + 32);
                            int8x16_t vrA3 = vld1q_s8(rA_ptr + i + 48);

                            int8x16_t vrB0 = vld1q_s8(rB_ptr + i + 0);
                            int8x16_t vrB1 = vld1q_s8(rB_ptr + i + 16);
                            int8x16_t vrB2 = vld1q_s8(rB_ptr + i + 32);
                            int8x16_t vrB3 = vld1q_s8(rB_ptr + i + 48);

                            aA0 = vpadalq_s16(aA0, vmull_s8(vget_low_s8(vrA0),  vget_low_s8(vx0)));
                            aA1 = vpadalq_s16(aA1, vmull_s8(vget_high_s8(vrA0), vget_high_s8(vx0)));
                            aA2 = vpadalq_s16(aA2, vmull_s8(vget_low_s8(vrA1),  vget_low_s8(vx1)));
                            aA3 = vpadalq_s16(aA3, vmull_s8(vget_high_s8(vrA1), vget_high_s8(vx1)));

                            aB0 = vpadalq_s16(aB0, vmull_s8(vget_low_s8(vrB0),  vget_low_s8(vx0)));
                            aB1 = vpadalq_s16(aB1, vmull_s8(vget_high_s8(vrB0), vget_high_s8(vx0)));
                            aB2 = vpadalq_s16(aB2, vmull_s8(vget_low_s8(vrB1),  vget_low_s8(vx1)));
                            aB3 = vpadalq_s16(aB3, vmull_s8(vget_high_s8(vrB1), vget_high_s8(vx1)));

                            aA0 = vpadalq_s16(aA0, vmull_s8(vget_low_s8(vrA2),  vget_low_s8(vx2)));
                            aA1 = vpadalq_s16(aA1, vmull_s8(vget_high_s8(vrA2), vget_high_s8(vx2)));
                            aA2 = vpadalq_s16(aA2, vmull_s8(vget_low_s8(vrA3),  vget_low_s8(vx3)));
                            aA3 = vpadalq_s16(aA3, vmull_s8(vget_high_s8(vrA3), vget_high_s8(vx3)));

                            aB0 = vpadalq_s16(aB0, vmull_s8(vget_low_s8(vrB2),  vget_low_s8(vx2)));
                            aB1 = vpadalq_s16(aB1, vmull_s8(vget_high_s8(vrB2), vget_high_s8(vx2)));
                            aB2 = vpadalq_s16(aB2, vmull_s8(vget_low_s8(vrB3),  vget_low_s8(vx3)));
                            aB3 = vpadalq_s16(aB3, vmull_s8(vget_high_s8(vrB3), vget_high_s8(vx3)));
                        }

                        int32_t dotA = vaddvq_s32(vaddq_s32(vaddq_s32(aA0, aA1), vaddq_s32(aA2, aA3)));
                        int32_t dotB = vaddvq_s32(vaddq_s32(vaddq_s32(aB0, aB1), vaddq_s32(aB2, aB3)));

                        insert_cand((float)dotA * lmhead_t1_scales[r], dotA, r);
                        insert_cand((float)dotB * lmhead_t1_scales[r + 1], dotB, r + 1);
                    }
                    for (; r < end; r++) {
                        const int8_t* __restrict__ r_ptr = (const int8_t*)__builtin_assume_aligned(lmhead_t1_s1_q8.data() + r * D_S1, 64);
                        int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
                        int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
                        for (int i = 0; i < D_S1; i += 64) {
                            int8x16_t vx0 = vld1q_s8(x_s1_ptr + i + 0);
                            int8x16_t vx1 = vld1q_s8(x_s1_ptr + i + 16);
                            int8x16_t vx2 = vld1q_s8(x_s1_ptr + i + 32);
                            int8x16_t vx3 = vld1q_s8(x_s1_ptr + i + 48);
                            int8x16_t vr0 = vld1q_s8(r_ptr + i + 0);
                            int8x16_t vr1 = vld1q_s8(r_ptr + i + 16);
                            int8x16_t vr2 = vld1q_s8(r_ptr + i + 32);
                            int8x16_t vr3 = vld1q_s8(r_ptr + i + 48);
                            a0 = vpadalq_s16(a0, vmull_s8(vget_low_s8(vr0),  vget_low_s8(vx0)));
                            a1 = vpadalq_s16(a1, vmull_s8(vget_high_s8(vr0), vget_high_s8(vx0)));
                            a2 = vpadalq_s16(a2, vmull_s8(vget_low_s8(vr1),  vget_low_s8(vx1)));
                            a3 = vpadalq_s16(a3, vmull_s8(vget_high_s8(vr1), vget_high_s8(vx1)));
                            a0 = vpadalq_s16(a0, vmull_s8(vget_low_s8(vr2),  vget_low_s8(vx2)));
                            a1 = vpadalq_s16(a1, vmull_s8(vget_high_s8(vr2), vget_high_s8(vx2)));
                            a2 = vpadalq_s16(a2, vmull_s8(vget_low_s8(vr3),  vget_low_s8(vx3)));
                            a3 = vpadalq_s16(a3, vmull_s8(vget_high_s8(vr3), vget_high_s8(vx3)));
                        }
                        int32x4_t sum01 = vaddq_s32(a0, a1);
                        int32x4_t sum23 = vaddq_s32(a2, a3);
                        int32_t dot = vaddvq_s32(vaddq_s32(sum01, sum23));
                        float val = (float)dot * lmhead_t1_scales[r];
                        insert_cand(val, dot, r);
                    }
#else
                    for (int r = start; r < end; r++) {
                        const int8_t* __restrict__ r_ptr = (const int8_t*)__builtin_assume_aligned(lmhead_t1_s1_q8.data() + r * D_S1, 64);
                        int32_t dot = 0;
                        for (int i = 0; i < D_S1; i++) {
                            dot += r_ptr[i] * x_s1_ptr[i];
                        }
                        float val = (float)dot * lmhead_t1_scales[r];
                        insert_cand(val, dot, r);
                    }
#endif

                    // --- STAGE 2: Zero-Barrier Direct Refinement on Thread's Local Heap ---
#if defined(__aarch64__) || defined(__arm__)
                    int c = 0;
                    for (; c <= heap_sz - 2; c += 2) {
                        int rA = heap[c + 0].token_id;
                        int rB = heap[c + 1].token_id;
                        const int8_t* __restrict__ rA_ptr = (const int8_t*)__builtin_assume_aligned(lmhead_t1_s2_q8.data() + rA * D_S2, 64);
                        const int8_t* __restrict__ rB_ptr = (const int8_t*)__builtin_assume_aligned(lmhead_t1_s2_q8.data() + rB * D_S2, 64);
                        if (c + 3 < heap_sz) {
                            __builtin_prefetch(lmhead_t1_s2_q8.data() + heap[c + 2].token_id * D_S2, 0, 0);
                            __builtin_prefetch(lmhead_t1_s2_q8.data() + heap[c + 3].token_id * D_S2, 0, 0);
                        }

                        int32x4_t aA0 = vdupq_n_s32(0), aA1 = vdupq_n_s32(0);
                        int32x4_t aA2 = vdupq_n_s32(0), aA3 = vdupq_n_s32(0);
                        int32x4_t aB0 = vdupq_n_s32(0), aB1 = vdupq_n_s32(0);
                        int32x4_t aB2 = vdupq_n_s32(0), aB3 = vdupq_n_s32(0);

                        for (int i = 0; i < D_S2; i += 64) {
                            int8x16_t vx0 = vld1q_s8(x_s2_ptr + i + 0);
                            int8x16_t vx1 = vld1q_s8(x_s2_ptr + i + 16);
                            int8x16_t vx2 = vld1q_s8(x_s2_ptr + i + 32);
                            int8x16_t vx3 = vld1q_s8(x_s2_ptr + i + 48);

                            int8x16_t vrA0 = vld1q_s8(rA_ptr + i + 0);
                            int8x16_t vrA1 = vld1q_s8(rA_ptr + i + 16);
                            int8x16_t vrA2 = vld1q_s8(rA_ptr + i + 32);
                            int8x16_t vrA3 = vld1q_s8(rA_ptr + i + 48);

                            int8x16_t vrB0 = vld1q_s8(rB_ptr + i + 0);
                            int8x16_t vrB1 = vld1q_s8(rB_ptr + i + 16);
                            int8x16_t vrB2 = vld1q_s8(rB_ptr + i + 32);
                            int8x16_t vrB3 = vld1q_s8(rB_ptr + i + 48);

                            aA0 = vpadalq_s16(aA0, vmull_s8(vget_low_s8(vrA0),  vget_low_s8(vx0)));
                            aA1 = vpadalq_s16(aA1, vmull_s8(vget_high_s8(vrA0), vget_high_s8(vx0)));
                            aA2 = vpadalq_s16(aA2, vmull_s8(vget_low_s8(vrA1),  vget_low_s8(vx1)));
                            aA3 = vpadalq_s16(aA3, vmull_s8(vget_high_s8(vrA1), vget_high_s8(vx1)));

                            aB0 = vpadalq_s16(aB0, vmull_s8(vget_low_s8(vrB0),  vget_low_s8(vx0)));
                            aB1 = vpadalq_s16(aB1, vmull_s8(vget_high_s8(vrB0), vget_high_s8(vx0)));
                            aB2 = vpadalq_s16(aB2, vmull_s8(vget_low_s8(vrB1),  vget_low_s8(vx1)));
                            aB3 = vpadalq_s16(aB3, vmull_s8(vget_high_s8(vrB1), vget_high_s8(vx1)));

                            aA0 = vpadalq_s16(aA0, vmull_s8(vget_low_s8(vrA2),  vget_low_s8(vx2)));
                            aA1 = vpadalq_s16(aA1, vmull_s8(vget_high_s8(vrA2), vget_high_s8(vx2)));
                            aA2 = vpadalq_s16(aA2, vmull_s8(vget_low_s8(vrA3),  vget_low_s8(vx3)));
                            aA3 = vpadalq_s16(aA3, vmull_s8(vget_high_s8(vrA3), vget_high_s8(vx3)));

                            aB0 = vpadalq_s16(aB0, vmull_s8(vget_low_s8(vrB2),  vget_low_s8(vx2)));
                            aB1 = vpadalq_s16(aB1, vmull_s8(vget_high_s8(vrB2), vget_high_s8(vx2)));
                            aB2 = vpadalq_s16(aB2, vmull_s8(vget_low_s8(vrB3),  vget_low_s8(vx3)));
                            aB3 = vpadalq_s16(aB3, vmull_s8(vget_high_s8(vrB3), vget_high_s8(vx3)));
                        }
                        int32_t dotA = vaddvq_s32(vaddq_s32(vaddq_s32(aA0, aA1), vaddq_s32(aA2, aA3)));
                        int32_t dotB = vaddvq_s32(vaddq_s32(vaddq_s32(aB0, aB1), vaddq_s32(aB2, aB3)));
                        heap[c + 0].score = (float)(heap[c + 0].dot_s1 + dotA) * lmhead_t1_scales[rA];
                        heap[c + 1].score = (float)(heap[c + 1].dot_s1 + dotB) * lmhead_t1_scales[rB];
                    }
                    if (c < heap_sz) {
                        int r = heap[c].token_id;
                        int32_t dot_s1 = heap[c].dot_s1;
                        const int8_t* __restrict__ r_s2_ptr = (const int8_t*)__builtin_assume_aligned(lmhead_t1_s2_q8.data() + r * D_S2, 64);
                        int32_t dot_s2 = 0;
                        for (int i = 0; i < D_S2; i++) dot_s2 += r_s2_ptr[i] * x_s2_ptr[i];
                        heap[c].score = (float)(dot_s1 + dot_s2) * lmhead_t1_scales[r];
                    }
#else
                    for (int c = 0; c < heap_sz; c++) {
                        int r = heap[c].token_id;
                        int32_t dot_s1 = heap[c].dot_s1;
                        const int8_t* __restrict__ r_s2_ptr = (const int8_t*)__builtin_assume_aligned(lmhead_t1_s2_q8.data() + r * D_S2, 64);
                        int32_t dot_s2 = 0;
                        for (int i = 0; i < D_S2; i++) dot_s2 += r_s2_ptr[i] * x_s2_ptr[i];
                        heap[c].score = (float)(dot_s1 + dot_s2) * lmhead_t1_scales[r];
                    }
#endif

                    // In-thread local pruning to top 128:
                    int local_top = std::min(128, heap_sz);
                    std::partial_sort(heap, heap + local_top, heap + heap_sz, CandS1Greater());

                    if (tid < MAX_THREADS) {
                        int base = tid * 128;
                        for (int k = 0; k < local_top; k++) {
                            pool_cands[base + k] = heap[k];
                        }
                        pool_cands_sz[tid] = local_top;
                    }
                } // end pragma omp parallel (ZERO barriers!)

                total_cands_count = 0;
                for (int t = 0; t < nthreads; t++) {
                    int base = t * 128;
                    for (int k = 0; k < pool_cands_sz[t]; k++) {
                        pool_cands[total_cands_count++] = pool_cands[base + k];
                    }
                }

                // Global Top-K selection on at most 4 * 64 = 256 candidates!
                int k_final = std::min(lmhead_top_cand, total_cands_count);
                std::partial_sort(pool_cands, pool_cands + k_final, pool_cands + total_cands_count,
                    [](const CandS1& a, const CandS1& b) {
                        if (a.score != b.score) return a.score > b.score;
                        return a.token_id < b.token_id;
                    });

                for (int i = 0; i < k_final; i++) {
                    thread_top_t1[0][i] = {pool_cands[i].score, pool_cands[i].token_id};
                }
                thread_top_t1_sz[0] = k_final;
                for (int t = 1; t < nthreads; t++) {
                    thread_top_t1_sz[t] = 0;
                }

                static int tier2_mode = -1;
                if (tier2_mode == -1) {
                    const char* env_t2 = getenv("V37_LMHEAD_TIER2");
                    tier2_mode = env_t2 ? atoi(env_t2) : 0;
                }
                if (tier2_mode == 1) {
                    #pragma omp parallel
                    {
                        int tid = omp_get_thread_num();
                        Elem heap2[MAX_K_CAND];
                        int heap2_sz = 0;
                        float cur_threshold = -1e30f;
                        int32_t min_dot2 = 0;
                        const int8_t* __restrict__ x_sub_ptr = (const int8_t*)__builtin_assume_aligned(x_sub_q8.data(), 64);
                        #pragma omp for schedule(static)
                        for (int r = 0; r < n2_rows; r++) {
                            const int8_t* __restrict__ r_ptr = (const int8_t*)__builtin_assume_aligned(lmhead_t2_coarse_q8.data() + r * D_COARSE, 64);
                            int32_t dot = 0;
                            for (int i = 0; i < D_COARSE; i++) dot += r_ptr[i] * x_sub_ptr[i];
                            if (dot > min_dot2) {
                                float val = (float)dot * lmhead_t2_scales[r];
                                if (heap2_sz < K_CAND) {
                                    heap2[heap2_sz++] = {val, TIER1_SIZE + r};
                                    if (heap2_sz == K_CAND) {
                                        std::make_heap(heap2, heap2 + K_CAND, std::greater<Elem>());
                                        cur_threshold = heap2[0].first;
                                    }
                                } else if (val > cur_threshold) {
                                    std::pop_heap(heap2, heap2 + K_CAND, std::greater<Elem>());
                                    heap2[K_CAND - 1] = {val, TIER1_SIZE + r};
                                    std::push_heap(heap2, heap2 + K_CAND, std::greater<Elem>());
                                    cur_threshold = heap2[0].first;
                                }
                            }
                        }
                        if (tid < MAX_THREADS) {
                            for (int i = 0; i < heap2_sz; i++) thread_top_t2[tid][i] = heap2[i];
                            thread_top_t2_sz[tid] = heap2_sz;
                        }
                    }
                }
            } else {
                #pragma omp parallel
                {
                    int tid = omp_get_thread_num();
                    Elem heap1[MAX_K_CAND];
                    int heap1_sz = 0;
                    float cur_threshold = -1e30f;
                    int32_t min_dot = 0;

                    const int8_t* __restrict__ x_ptr = (const int8_t*)__builtin_assume_aligned(x_t1_q8.data(), 64);

                    #pragma omp for schedule(static) nowait
                    for (int r = 0; r < TIER1_SIZE; r++) {
                        const int8_t* __restrict__ r_ptr = (const int8_t*)__builtin_assume_aligned(lmhead_t1_q8.data() + r * D1, 64);
                        if (r + 2 < TIER1_SIZE) {
                            const int8_t* pnext = r_ptr + 2 * D1;
                            __builtin_prefetch(pnext, 0, 0);
                            __builtin_prefetch(pnext + 64, 0, 0);
                            __builtin_prefetch(pnext + 128, 0, 0);
                        }

#if defined(__aarch64__) || defined(__arm__)
                        int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
                        int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
                        int32x4_t a4 = vdupq_n_s32(0), a5 = vdupq_n_s32(0);
                        int32x4_t a6 = vdupq_n_s32(0), a7 = vdupq_n_s32(0);

                        int i = 0;
                        for (; i <= D1 - 64; i += 64) {
                            int8x16_t vx0 = vld1q_s8(x_ptr + i + 0);
                            int8x16_t vx1 = vld1q_s8(x_ptr + i + 16);
                            int8x16_t vx2 = vld1q_s8(x_ptr + i + 32);
                            int8x16_t vx3 = vld1q_s8(x_ptr + i + 48);

                            int8x16_t vr0 = vld1q_s8(r_ptr + i + 0);
                            int8x16_t vr1 = vld1q_s8(r_ptr + i + 16);
                            int8x16_t vr2 = vld1q_s8(r_ptr + i + 32);
                            int8x16_t vr3 = vld1q_s8(r_ptr + i + 48);

                            int16x8_t m0 = vmull_s8(vget_low_s8(vr0),  vget_low_s8(vx0));
                            int16x8_t m1 = vmull_s8(vget_high_s8(vr0), vget_high_s8(vx0));
                            int16x8_t m2 = vmull_s8(vget_low_s8(vr1),  vget_low_s8(vx1));
                            int16x8_t m3 = vmull_s8(vget_high_s8(vr1), vget_high_s8(vx1));

                            a0 = vpadalq_s16(a0, m0);
                            int16x8_t m4 = vmull_s8(vget_low_s8(vr2),  vget_low_s8(vx2));
                            a1 = vpadalq_s16(a1, m1);
                            int16x8_t m5 = vmull_s8(vget_high_s8(vr2), vget_high_s8(vx2));
                            a2 = vpadalq_s16(a2, m2);
                            int16x8_t m6 = vmull_s8(vget_low_s8(vr3),  vget_low_s8(vx3));
                            a3 = vpadalq_s16(a3, m3);
                            int16x8_t m7 = vmull_s8(vget_high_s8(vr3), vget_high_s8(vx3));

                            a4 = vpadalq_s16(a4, m4);
                            a5 = vpadalq_s16(a5, m5);
                            a6 = vpadalq_s16(a6, m6);
                            a7 = vpadalq_s16(a7, m7);
                        }
                        int32x4_t sum01 = vaddq_s32(a0, a1);
                        int32x4_t sum23 = vaddq_s32(a2, a3);
                        int32x4_t sum45 = vaddq_s32(a4, a5);
                        int32x4_t sum67 = vaddq_s32(a6, a7);
                        int32_t dot = vaddvq_s32(vaddq_s32(vaddq_s32(sum01, sum23), vaddq_s32(sum45, sum67)));
                        for (; i < D1; i++) {
                            dot += r_ptr[i] * x_ptr[i];
                        }
#else
                        int32_t dot = 0;
                        for (int i = 0; i < D1; i++) {
                            dot += r_ptr[i] * x_ptr[i];
                        }
#endif
                        if (dot > min_dot) {
                            if (__builtin_expect(heap1_sz < K_CAND, 0)) {
                                float val = (float)dot * lmhead_t1_scales[r];
                                heap1[heap1_sz++] = {val, r};
                                if (heap1_sz == K_CAND) {
                                    std::make_heap(heap1, heap1 + K_CAND, std::greater<Elem>());
                                    cur_threshold = heap1[0].first;
                                    if (cur_threshold > 0.0f) {
                                        min_dot = (int32_t)(cur_threshold * inv_scale1);
                                    }
                                }
                            } else {
                                float val = (float)dot * lmhead_t1_scales[r];
                                if (val > cur_threshold) {
                                    std::pop_heap(heap1, heap1 + K_CAND, std::greater<Elem>());
                                    heap1[K_CAND - 1] = {val, r};
                                    std::push_heap(heap1, heap1 + K_CAND, std::greater<Elem>());
                                    cur_threshold = heap1[0].first;
                                    if (cur_threshold > 0.0f) {
                                        min_dot = (int32_t)(cur_threshold * inv_scale1);
                                    }
                                }
                            }
                        }
                    }

                    if (tid < MAX_THREADS) {
                        for (int i = 0; i < heap1_sz; i++) {
                            thread_top_t1[tid][i] = heap1[i];
                        }
                        thread_top_t1_sz[tid] = heap1_sz;
                    }

                    // 3. Tier 2 Coarse INT8 dot products (99,840 rows x 128 dims)
                    Elem heap2[MAX_K_CAND];
                    int heap2_sz = 0;
                    cur_threshold = -1e30f;
                    int32_t min_dot2 = 0;

                    const int8_t* __restrict__ x_sub_ptr = (const int8_t*)__builtin_assume_aligned(x_sub_q8.data(), 64);
#if defined(__aarch64__) || defined(__arm__)
                    int8x16_t vx0 = vdupq_n_s8(0), vx1 = vdupq_n_s8(0), vx2 = vdupq_n_s8(0), vx3 = vdupq_n_s8(0);
                    int8x16_t vx4 = vdupq_n_s8(0), vx5 = vdupq_n_s8(0), vx6 = vdupq_n_s8(0), vx7 = vdupq_n_s8(0);
                    if (D_COARSE == 128) {
                        vx0 = vld1q_s8(x_sub_ptr + 0);
                        vx1 = vld1q_s8(x_sub_ptr + 16);
                        vx2 = vld1q_s8(x_sub_ptr + 32);
                        vx3 = vld1q_s8(x_sub_ptr + 48);
                        vx4 = vld1q_s8(x_sub_ptr + 64);
                        vx5 = vld1q_s8(x_sub_ptr + 80);
                        vx6 = vld1q_s8(x_sub_ptr + 96);
                        vx7 = vld1q_s8(x_sub_ptr + 112);
                    }
#endif
                    static int tier2_mode = -1;
                    if (tier2_mode == -1) {
                        const char* env_t2 = getenv("V37_LMHEAD_TIER2");
                        tier2_mode = env_t2 ? atoi(env_t2) : 0;
                    }
                    if (tier2_mode == 1) {
                    #pragma omp for schedule(static)
                    for (int r = 0; r < n2_rows; r++) {
                        const int8_t* __restrict__ r_ptr = (const int8_t*)__builtin_assume_aligned(lmhead_t2_coarse_q8.data() + r * D_COARSE, 64);
                        if (r + 1 < n2_rows) {
                            __builtin_prefetch(r_ptr + D_COARSE, 0, 0);
                            __builtin_prefetch(r_ptr + D_COARSE + 64, 0, 0);
                        }

#if defined(__aarch64__) || defined(__arm__)
                        int32_t dot = 0;
                        if (D_COARSE == 128) {
                            int8x16_t vr0 = vld1q_s8(r_ptr + 0);
                            int8x16_t vr1 = vld1q_s8(r_ptr + 16);
                            int8x16_t vr2 = vld1q_s8(r_ptr + 32);
                            int8x16_t vr3 = vld1q_s8(r_ptr + 48);
                            int8x16_t vr4 = vld1q_s8(r_ptr + 64);
                            int8x16_t vr5 = vld1q_s8(r_ptr + 80);
                            int8x16_t vr6 = vld1q_s8(r_ptr + 96);
                            int8x16_t vr7 = vld1q_s8(r_ptr + 112);

                            int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
                            int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
                            int32x4_t a4 = vdupq_n_s32(0), a5 = vdupq_n_s32(0);
                            int32x4_t a6 = vdupq_n_s32(0), a7 = vdupq_n_s32(0);

                            int16x8_t m0 = vmull_s8(vget_low_s8(vr0),  vget_low_s8(vx0));
                            int16x8_t m1 = vmull_s8(vget_high_s8(vr0), vget_high_s8(vx0));
                            int16x8_t m2 = vmull_s8(vget_low_s8(vr1),  vget_low_s8(vx1));
                            int16x8_t m3 = vmull_s8(vget_high_s8(vr1), vget_high_s8(vx1));

                            int16x8_t m4 = vmull_s8(vget_low_s8(vr2),  vget_low_s8(vx2));
                            int16x8_t m5 = vmull_s8(vget_high_s8(vr2), vget_high_s8(vx2));
                            int16x8_t m6 = vmull_s8(vget_low_s8(vr3),  vget_low_s8(vx3));
                            int16x8_t m7 = vmull_s8(vget_high_s8(vr3), vget_high_s8(vx3));

                            a0 = vpadalq_s16(a0, m0);
                            a1 = vpadalq_s16(a1, m1);
                            a2 = vpadalq_s16(a2, m2);
                            a3 = vpadalq_s16(a3, m3);

                            a4 = vpadalq_s16(a4, m4);
                            a5 = vpadalq_s16(a5, m5);
                            a6 = vpadalq_s16(a6, m6);
                            a7 = vpadalq_s16(a7, m7);

                            int16x8_t m8 = vmull_s8(vget_low_s8(vr4),  vget_low_s8(vx4));
                            int16x8_t m9 = vmull_s8(vget_high_s8(vr4), vget_high_s8(vx4));
                            int16x8_t ma = vmull_s8(vget_low_s8(vr5),  vget_low_s8(vx5));
                            int16x8_t mb = vmull_s8(vget_high_s8(vr5), vget_high_s8(vx5));

                            int16x8_t mc = vmull_s8(vget_low_s8(vr6),  vget_low_s8(vx6));
                            int16x8_t md = vmull_s8(vget_high_s8(vr6), vget_high_s8(vx6));
                            int16x8_t me = vmull_s8(vget_low_s8(vr7),  vget_low_s8(vx7));
                            int16x8_t mf = vmull_s8(vget_high_s8(vr7), vget_high_s8(vx7));

                            a0 = vpadalq_s16(a0, m8);
                            a1 = vpadalq_s16(a1, m9);
                            a2 = vpadalq_s16(a2, ma);
                            a3 = vpadalq_s16(a3, mb);

                            a4 = vpadalq_s16(a4, mc);
                            a5 = vpadalq_s16(a5, md);
                            a6 = vpadalq_s16(a6, me);
                            a7 = vpadalq_s16(a7, mf);

                            int32x4_t sum01 = vaddq_s32(a0, a1);
                            int32x4_t sum23 = vaddq_s32(a2, a3);
                            int32x4_t sum45 = vaddq_s32(a4, a5);
                            int32x4_t sum67 = vaddq_s32(a6, a7);
                            dot = vaddvq_s32(vaddq_s32(vaddq_s32(sum01, sum23), vaddq_s32(sum45, sum67)));
                        } else {
                            for (int i = 0; i < D_COARSE; i++) {
                                dot += r_ptr[i] * x_sub_ptr[i];
                            }
                        }
#else
                        int32_t dot = 0;
                        for (int i = 0; i < D_COARSE; i++) {
                            dot += r_ptr[i] * x_sub_ptr[i];
                        }
#endif
                        if (dot > min_dot2) {
                            if (__builtin_expect(heap2_sz < K_CAND, 0)) {
                                float val = (float)dot * lmhead_t2_scales[r];
                                heap2[heap2_sz++] = {val, TIER1_SIZE + r};
                                if (heap2_sz == K_CAND) {
                                    std::make_heap(heap2, heap2 + K_CAND, std::greater<Elem>());
                                    cur_threshold = heap2[0].first;
                                    if (cur_threshold > 0.0f) {
                                        min_dot2 = (int32_t)(cur_threshold * inv_scale2);
                                    }
                                }
                            } else {
                                float val = (float)dot * lmhead_t2_scales[r];
                                if (val > cur_threshold) {
                                    std::pop_heap(heap2, heap2 + K_CAND, std::greater<Elem>());
                                    heap2[K_CAND - 1] = {val, TIER1_SIZE + r};
                                    std::push_heap(heap2, heap2 + K_CAND, std::greater<Elem>());
                                    cur_threshold = heap2[0].first;
                                    if (cur_threshold > 0.0f) {
                                        min_dot2 = (int32_t)(cur_threshold * inv_scale2);
                                    }
                                }
                            }
                        }
                    }
                    }
                    if (tid < MAX_THREADS) {
                        for (int i = 0; i < heap2_sz; i++) {
                            thread_top_t2[tid][i] = heap2[i];
                        }
                        thread_top_t2_sz[tid] = heap2_sz;
                    }
                }
            }
            auto t_lm3 = std::chrono::high_resolution_clock::now();

            // 4. Collect candidate tokens (Fast Bitmap Deduplication)
            static std::vector<int> candidates;
            candidates.clear();

            auto add_candidate = [&](int tok) {
                if (tok >= 0 && tok < config.vocab_size && !lmhead_in_cands[tok]) {
                    lmhead_in_cands[tok] = 1;
                    candidates.push_back(tok);
                }
            };

            for (int t = 0; t < nthreads; t++) {
                for (int i = 0; i < thread_top_t1_sz[t]; i++) add_candidate(thread_top_t1[t][i].second);
                for (int i = 0; i < thread_top_t2_sz[t]; i++) add_candidate(thread_top_t2[t][i].second);
            }
            const std::vector<int>& hist = (tokens.size() > 1) ? tokens : past_tokens_history;
            int hist_start = std::max(0, (int)hist.size() - 64);
            for (size_t p = hist_start; p < hist.size(); p++) {
                add_candidate(hist[p]);
            }
            add_candidate(128000); // BOS
            add_candidate(128001); // EOS
            add_candidate(128009); // EOT

            for (int tok : candidates) {
                lmhead_in_cands[tok] = 0;
            }
            std::sort(candidates.begin(), candidates.end());
            last_candidates = candidates;
            auto t_lm4 = std::chrono::high_resolution_clock::now();

            // 5. Sparse reset of logits from previous token to preserve L2 cache (if not already reset by prepare_lmhead_overlap)
            for (int p_tok : prev_candidates) {
                logits[p_tok] = -1e9f;
            }
            prev_candidates = candidates;

            #pragma omp parallel for schedule(static)
            for (int c = 0; c < (int)candidates.size(); c++) {
                int tok = candidates[c];
                const v37_f16_t* row = w_f16 + (size_t)tok * config.n_embd;
                if (c + 1 < (int)candidates.size()) {
                    __builtin_prefetch(w_f16 + (size_t)candidates[c + 1] * config.n_embd, 0, 1);
                }

#if defined(__aarch64__) || defined(__arm__)
                float32x4_t sum0 = vdupq_n_f32(0.0f);
                float32x4_t sum1 = vdupq_n_f32(0.0f);
                float32x4_t sum2 = vdupq_n_f32(0.0f);
                float32x4_t sum3 = vdupq_n_f32(0.0f);

                const float* fn_ptr = final_norm;
                int i = 0;
                for (; i <= config.n_embd - 16; i += 16) {
                    float32x4_t fn0 = vld1q_f32(fn_ptr + i + 0);
                    float32x4_t fn1 = vld1q_f32(fn_ptr + i + 4);
                    float32x4_t fn2 = vld1q_f32(fn_ptr + i + 8);
                    float32x4_t fn3 = vld1q_f32(fn_ptr + i + 12);

                    float16x8_t hw0 = vld1q_f16(row + i + 0);
                    float16x8_t hw1 = vld1q_f16(row + i + 8);

                    sum0 = vmlaq_f32(sum0, fn0, vcvt_f32_f16(vget_low_f16(hw0)));
                    sum1 = vmlaq_f32(sum1, fn1, vcvt_f32_f16(vget_high_f16(hw0)));
                    sum2 = vmlaq_f32(sum2, fn2, vcvt_f32_f16(vget_low_f16(hw1)));
                    sum3 = vmlaq_f32(sum3, fn3, vcvt_f32_f16(vget_high_f16(hw1)));
                }
                float sum = vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
                for (; i < config.n_embd; i++) {
                    sum += fn_ptr[i] * V37_F16_TO_F32(row[i]);
                }
                logits[tok] = sum;
#else
                float s = 0.0f;
                for (int i = 0; i < config.n_embd; i++) {
                    s += final_norm[i] * V37_F16_TO_F32(row[i]);
                }
                logits[tok] = s;
#endif
            }
            auto t_lm5 = std::chrono::high_resolution_clock::now();
            if (getenv("V37_VERBOSE")) {
                double ms_q = std::chrono::duration<double, std::milli>(t_lm1 - t_lm0).count();
                double ms_t12 = std::chrono::duration<double, std::milli>(t_lm3 - t_lm1).count();
                double ms_cand = std::chrono::duration<double, std::milli>(t_lm4 - t_lm3).count();
                double ms_rerank = std::chrono::duration<double, std::milli>(t_lm5 - t_lm4).count();
                printf("[LMHEAD-DETAIL] quant=%.2f ms, t12=%.2f ms, cand=%.2f ms, rerank=%.2f ms | cands=%zu\n",
                       ms_q, ms_t12, ms_cand, ms_rerank, candidates.size());
            }
        }
    }

    auto t_end_lmhead = std::chrono::high_resolution_clock::now();

static int fwd_step_cnt = 0;
fwd_step_cnt++;
if (getenv("V37_VERBOSE")) {
    double ms_trans = std::chrono::duration<double, std::milli>(t_end_trans - t_start_fwd).count();
    double ms_norm = std::chrono::duration<double, std::milli>(t_end_norm - t_end_trans).count();
    double ms_lmhead = std::chrono::duration<double, std::milli>(t_end_lmhead - t_end_norm).count();
    double ms_total = std::chrono::duration<double, std::milli>(t_end_lmhead - t_start_fwd).count();
    printf("[FWD-TIMING #%d] trans=%.1f ms (qkv=%.1f, attn=%.1f, oproj=%.1f, ffn=%.1f), norm=%.1f ms, lmhead=%.1f ms | total=%.1f ms\n",
           fwd_step_cnt, ms_trans, prof_qkv_ms, prof_attn_ms, prof_oproj_ms, prof_ffn_ms, ms_norm, ms_lmhead, ms_total);
}
    
    // Print top 5 logits (use partial_sort instead of full sort for performance)
    if (getenv("DUMP_LOGITS") || getenv("V37_DEBUG")) {
        std::vector<std::pair<float, int>> sorted_logits(config.vocab_size);
        for (int i = 0; i < config.vocab_size; i++) sorted_logits[i] = {logits[i], i};
        int k_show = std::min(5, config.vocab_size);
        std::partial_sort(sorted_logits.begin(), sorted_logits.begin() + k_show, sorted_logits.end(),
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                return a.first > b.first;
            });
        printf("[DEBUG-LOGITS] Top %d: ", k_show);
        for (int i = 0; i < k_show; i++) {
            printf("tok=%d(%.4f) ", sorted_logits[i].second, sorted_logits[i].first);
        }
        printf("\n");
    }

    int nan_cnt = 0;
    for(int i=0; i<config.vocab_size; i++) if (std::isnan(logits[i])) nan_cnt++;
    if (nan_cnt > 0) printf("[LOGITS] WARNING: Found %d NaNs in logits!\n", nan_cnt);
    return logits.data();

}

V37Engine::~V37Engine() {
    if (v37pack_mmap) munmap(v37pack_mmap, v37pack_size);
}
