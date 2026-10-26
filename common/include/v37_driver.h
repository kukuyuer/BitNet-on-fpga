/**
 * V37 Driver Header - v340.0 适配版
 * 
 * 核心对齐：
 *   1. 还原 4 通道 DMA_W 权重发射。
 *   2. 暴露 Q24 Beta 接口，支持硬件级 (Beta * Gamma) 融合缩放。
 *   3. 新增 v37_reset_cma_slots 接口用于 CMA 槽位重置。
 */
#ifndef V37_DRIVER_H
#define V37_DRIVER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    V37_MODE_DIRECT         = 0,
    V37_MODE_INT8           = 0,
    V37_MODE_I2_S_PREPACKED = 4,
    V37_MODE_I2_S           = 5,
    V37_MODE_TL3_COMPACT    = 6,
    V37_MODE_TL3            = 7
} v37_mode_t;

typedef struct v37_sg_desc {
    uint32_t next_desc_ptr;
    uint32_t cfg_val;
    uint32_t act_in_addr;
    uint32_t act_out_addr;
    uint32_t x_length;
    uint32_t weight_addr_0;
    uint32_t weight_length;
    uint32_t y_length;
    float    total_scale;     // Scheme A: FP32 Scale for residual fusion
    uint32_t reserved1;
    uint32_t reserved2;
    uint32_t reserved3;
    uint32_t reserved4;
    uint32_t reserved5;
    uint32_t reserved6;
    uint32_t reserved7;
} v37_sg_desc_t;

typedef struct v37_engine v37_engine_t;

v37_engine_t* v37_create_engine();
void v37_destroy_engine(v37_engine_t* engine);

// 微码管理接口
int v37_load_microcode(v37_engine_t* engine, const uint32_t* ucode_words, int num_insts);
int v37_start_token(v37_engine_t* engine);

// 核心计算接口
int v37_compute_linear_preloaded(v37_engine_t* engine, const int8_t* x, int slot_id, void* y, int m, int k, v37_mode_t mode, const int8_t* act_scales, const float* w_scales, int BK, int stage, int m_offset, int total_m);
int v37_compute_linear_preloaded_opt(v37_engine_t* engine, const int8_t* x, int slot_id, void* y, int m, int k, v37_mode_t mode, const int8_t* act_scales, const float* w_scales, int BK, int stage, int m_offset, int total_m, int reuse_act, float act_scale, float* residual);
int v37_launch_linear_async(v37_engine_t* engine, const int8_t* x, int slot_id, int m, int k, v37_mode_t mode, int stage, int reuse_act);
int v37_wait_linear_async(v37_engine_t* engine, void* y, const float* w_scales, float act_scale, float* residual);
int v37_wait_linear_async_fused_up(v37_engine_t* engine, float* ffn_act, const float* gate, const float* w_scales, float act_scale);
int v37_wait_linear_async_f16(v37_engine_t* engine, void* kv_cache_v_dst, const float* w_scales, float act_scale);
int v37_load_rope_table(v37_engine_t* engine, const float* sin_table, const float* cos_table, int half_dim_beats);

int v37_preload_weights(v37_engine_t* engine, const uint8_t* data, const float* w_scales, size_t size, int slot_id, v37_mode_t mode, int k, int BK);
int v37_compute_linear(v37_engine_t* engine, const int8_t* x, const uint8_t* w, const float* w_scales, void* y, int m, int k, v37_mode_t mode, const int8_t* act_scales, int BK, int stage);
void v37_reset_cma_allocator(v37_engine_t* e, int slot_id);
void v37_load_loopback_rmsnorm_weights(v37_engine_t* engine, const float* weights, int dim);
int v37_preload_fused_gate_up(v37_engine_t* engine, int gate_slot, int up_slot, int fused_slot, int ffn_dim, int hidden_dim, v37_mode_t mode);
int v37_preload_fused_gate_up_buffers(v37_engine_t* engine, const uint8_t* gate_base, const uint8_t* up_base, int fused_slot, int ffn_dim, int hidden_dim, v37_mode_t mode);
int v37_preload_fused_qkv_buffers(v37_engine_t* engine, const uint8_t* q_base, const uint8_t* k_base, const uint8_t* v_base, int fused_slot, int q_dim, int k_dim, int v_dim, int hidden_dim, v37_mode_t mode);
int v37_preload_fused_direct(v37_engine_t* engine, const uint8_t* fused_data, size_t size, int fused_slot);
int v37_launch_linear_async_gate_up(v37_engine_t* engine, const int8_t* x, int fused_slot, int m, int k, v37_mode_t mode);
#ifdef __cplusplus
int v37_wait_linear_async_gate_up(v37_engine_t* engine, int8_t* qact_out, float* quant_s_out, const float* gate_scales, const float* up_scales, float act_scale, const float* sub_norm_weights, int ffn_dim, int act_type = 0);
int v37_compute_ffn_chain(v37_engine_t* engine, const int8_t* x, int gate_slot, int up_slot, int down_slot, int fused_gate_up_slot, float* out, int hidden_dim, int ffn_dim, const float* gate_scales, const float* up_scales, const float* down_scales, float act_scale, const float* sub_norm_weights, float* residual, v37_mode_t mod);
int v37_launch_ffn_chain(v37_engine_t* engine, const int8_t* x, int gate_slot, int up_slot, int down_slot, int fused_gate_up_slot, int hidden_dim, int ffn_dim, const float* gate_scales, const float* up_scales, float act_scale, const float* sub_norm_weights, float* quant_s_out, v37_mode_t mod, int act_type = 0);
#else
int v37_wait_linear_async_gate_up(v37_engine_t* engine, int8_t* qact_out, float* quant_s_out, const float* gate_scales, const float* up_scales, float act_scale, const float* sub_norm_weights, int ffn_dim, int act_type);
int v37_compute_ffn_chain(v37_engine_t* engine, const int8_t* x, int gate_slot, int up_slot, int down_slot, int fused_gate_up_slot, float* out, int hidden_dim, int ffn_dim, const float* gate_scales, const float* up_scales, const float* down_scales, float act_scale, const float* sub_norm_weights, float* residual, v37_mode_t mod);
int v37_launch_ffn_chain(v37_engine_t* engine, const int8_t* x, int gate_slot, int up_slot, int down_slot, int fused_gate_up_slot, int hidden_dim, int ffn_dim, const float* gate_scales, const float* up_scales, float act_scale, const float* sub_norm_weights, float* quant_s_out, v37_mode_t mod, int act_type);
#endif
int v37_wait_ffn_chain_down(v37_engine_t* engine, float* out, const float* down_scales, float quant_s, float* residual);
int v37_compute_qkv_fused(v37_engine_t* engine, const int8_t* x, int fused_slot, float* q_out, float* k_out, void* v_f16_out, int q_dim, int k_dim, int v_dim, int hidden_dim, const float* q_scales, const float* k_scales, const float* v_scales, float act_scale, int pos, void* k_f16_out, const float* rope_cos, const float* rope_sin, float inv_sqrt_head_dim, v37_mode_t mod);

void v37_sg_flush(v37_engine_t* engine);
void* v37_get_vbase(v37_engine_t* engine);
void* v37_get_cma_slot_ptr(v37_engine_t* engine, int slot_id);

#ifdef __cplusplus
}
#endif
#endif
