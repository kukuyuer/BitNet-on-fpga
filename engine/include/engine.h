#ifndef V37_ENGINE_H
#define V37_ENGINE_H
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include "v37_driver.h"
#include "loader.h"
#include "llama.h"
#include "model_config.h"

// Fallback defaults for backwards compatibility
#ifndef MAX_SEQ_LEN
#define MAX_SEQ_LEN 512
#endif
#ifndef N_EMBD
#define N_EMBD      2560
#endif
#ifndef HEAD_DIM
#define HEAD_DIM    128
#endif
#ifndef N_HEADS
#define N_HEADS     20
#endif
#ifndef N_KV_HEADS
#define N_KV_HEADS  5
#endif
#ifndef N_LAYERS
#define N_LAYERS    30
#endif
#ifndef N_FFN
#define N_FFN       6912
#endif
#ifndef VOCAB_SIZE
#define VOCAB_SIZE  128256
#endif

// V37Pack tensor entry: pre-swizzled CMA-ready data
struct V37PackEntry {
    uint8_t*   data;
    size_t     size;
    v37_mode_t mode;
};

struct LayerTensorCache {
    const float* scales[7];
    const uint8_t* raw_w[7];
    int slots[7];
    int fused_gate_up_slot;
    int fused_qkv_slot;
    std::vector<float> norm_attn;
    std::vector<float> norm_attn_sub;
    std::vector<float> norm_ffn;
    std::vector<float> norm_ffn_sub;
};

#include <stdlib.h>

template <typename T, size_t Alignment = 64>
struct AlignedAllocator {
    using value_type = T;
    AlignedAllocator() noexcept = default;
    template <typename U> AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}
    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        void* ptr = nullptr;
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }
    void deallocate(T* p, std::size_t) noexcept { free(p); }
    template <typename U> struct rebind { using other = AlignedAllocator<U, Alignment>; };
    bool operator==(const AlignedAllocator&) const noexcept { return true; }
    bool operator!=(const AlignedAllocator&) const noexcept { return false; }
};
template<typename T> using aligned_vector64 = std::vector<T, AlignedAllocator<T, 64>>;

class V37Engine {
public:
    V37Engine(v37_engine_t* e, V37Loader* l, struct llama_model* m, const ModelConfig* cfg = nullptr);
    ~V37Engine();

    float* forward(const std::vector<int>& tokens, int pos, bool compute_logits = true);
    
    // Load pre-packed .v37p file for hardware weights
    bool load_v37pack(const std::string& path);
    
    // Get pre-packed weight data for a tensor name (returns nullptr if not found)
    V37PackEntry* get_prepacked(const std::string& name);

    const std::vector<int>& get_candidates() const { return last_candidates; }
    const ModelConfig& get_config() const { return config; }

private:
    void transformer_block(int l, int pos);
    void rms_norm(float* out, const float* in, const float* weight, int n);
    void load_norm_weight(std::vector<float>& out, const std::string& name, int n);
    struct QuantizedAct {
        std::vector<int8_t> x_q;
        float s;
        int in_dim;
    };
    void quantize_activation_neon(const float* in, int in_dim, QuantizedAct& qact);
    void run_fpga_gemv_prequantized(int slot_id, const QuantizedAct& qact, float* out, int out_dim, const float* row_scales, int ext_stage=0, bool cpu_fallback=false, const uint8_t* cpu_weights=nullptr, const char* layer_name=nullptr, bool reuse_act=false, float* residual=nullptr);
    void run_fpga_gemv_launch(int slot_id, const QuantizedAct& qact, int out_dim, bool reuse_act = false);
    void run_fpga_gemv_wait(float* out, const float* row_scales, float act_scale, float* residual = nullptr);
    void run_fpga_gemv_wait_fused_up(float* ffn_act, const float* gate, const float* row_scales, float act_scale);
    void run_fpga_gemv_wait_f16(void* kv_cache_v_dst, const float* row_scales, float act_scale);
    void run_fpga_gemv_stream(int l, int slot_id, const float* x_in, float* out, int out_dim, int in_dim);
    void run_fpga_gemv_fast(int slot_id, float* x_in, float* out, int out_dim, int in_dim, const float* row_scales, int ext_stage=0, bool cpu_fallback=false, const uint8_t* cpu_weights=nullptr, const char* layer_name=nullptr, int m_offset=0, int total_m=0, const float* hw_scales=nullptr);
    void rope(float* q, float* k, int pos);
    void rope_q(float* q, int pos);
    void rope_k(float* k, int pos);
    void rope_k_f16(const float* k, void* kv_cache_k, int pos);
    void init_rope_table();
    void init_layer_cache();

    v37_engine_t* engine;
    V37Loader* loader;
    struct llama_model* model;
    ModelConfig config;
    
    std::vector<float> hidden_states;
    std::vector<float> logits;
    std::vector<ggml_fp16_t> kv_cache;

    // Dynamic intermediate buffers for transformer_block
    std::vector<float> buf_norm_out;
    std::vector<float> buf_q;
    std::vector<float> buf_k;
    std::vector<float> buf_v;
    std::vector<float> buf_attn_out;
    QuantizedAct       buf_qact_attn;
    std::vector<float> buf_attn_sub_norm_out;
    QuantizedAct       buf_qact_attn_sub;
    std::vector<float> buf_o_out;
    std::vector<float> buf_ffn_norm_out;
    QuantizedAct       buf_qact_ffn;
    std::vector<float> buf_gate;
    std::vector<float> buf_up;
    std::vector<float> buf_ffn_act;
    std::vector<float> buf_ffn_sub_norm_out;
    QuantizedAct       buf_qact_ffn_sub;
    std::vector<float> buf_down;
    std::vector<float> buf_final_norm;
    std::vector<uint8_t> lmhead_in_cands;

    // RoPE dynamic tables
    std::vector<float> rope_cos_table;
    std::vector<float> rope_sin_table;
    bool rope_table_inited;
    
    // V37Pack mmap state
    void*  v37pack_mmap;
    size_t v37pack_size;
    v37_mode_t active_pack_mode;
    std::unordered_map<std::string, V37PackEntry> v37pack_entries;
    bool all_weights_preloaded;
    void preload_all_weights_to_cma();

    std::vector<LayerTensorCache> layers_cache;
    std::vector<float> output_norm_weight;
    bool layer_cache_initialized;

    void init_lmhead_cache();
    bool lmhead_cache_initialized;
    aligned_vector64<int8_t> lmhead_t1_q8;
    aligned_vector64<float>  lmhead_t1_scales;
    aligned_vector64<int8_t> lmhead_t1_coarse_q8;
    aligned_vector64<float>  lmhead_t1_coarse_scales;
    aligned_vector64<int8_t> lmhead_t2_coarse_q8;
    aligned_vector64<float>  lmhead_t2_scales;
    std::vector<int>         lmhead_indices_t1;
    std::vector<int>         lmhead_indices_coarse;
    std::vector<int>         past_tokens_history;
    std::vector<int>         last_candidates;
    std::vector<int>         prev_candidates;
    float                    max_lmhead_t1_scale;
    float                    inv_max_lmhead_t1_scale;
    float                    max_lmhead_t2_scale;
    float                    inv_max_lmhead_t2_scale;
    int                      lmhead_tier1_size;
    int                      lmhead_d1;
    int                      lmhead_d_coarse;
    int                      lmhead_d_s1;
    int                      lmhead_d_s2;
    int                      lmhead_s1_k_per_thread;
    int                      lmhead_twostage_enabled;
    int                      lmhead_top_cand;
    aligned_vector64<int8_t> lmhead_t1_s1_q8;
    aligned_vector64<int8_t> lmhead_t1_s2_q8;
    void                     prepare_lmhead_overlap();
};
#endif

