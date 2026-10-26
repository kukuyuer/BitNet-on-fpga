#include "v37_driver.h"
#include <stdio.h>
#include <stdlib.h>

/**
 * v37_driver_cpu.cpp - CPU Reference Platform Driver Stub
 *
 * Implements the v37 driver interface as pure CPU stubs so that
 * semantic_engine runs in CPU reference mode (AVX2/NEON)
 * without requiring any FPGA hardware, MMIO, or CMA kernel drivers.
 */

int64_t t_prep = 0, t_sync_dev = 0, t_dma = 0, t_sync_cpu = 0, t_post = 0, t_calls = 0;
int64_t t_memcpy = 0;
int64_t t_setup = 0, t_poll = 0;

v37_engine_t* v37_create_engine() {
    return nullptr;
}

void v37_destroy_engine(v37_engine_t* engine) {
    (void)engine;
}

int v37_compute_linear_preloaded(v37_engine_t* engine, const int8_t* x, int slot_id, void* y, int m, int k, v37_mode_t mode, const int8_t* act_scales, const float* w_scales, int BK, int stage, int m_offset, int total_m) {
    (void)engine; (void)x; (void)slot_id; (void)y; (void)m; (void)k; (void)mode;
    (void)act_scales; (void)w_scales; (void)BK; (void)stage; (void)m_offset; (void)total_m;
    return -1;
}

int v37_compute_linear_preloaded_opt(v37_engine_t* engine, const int8_t* x, int slot_id, void* y, int m, int k, v37_mode_t mode, const int8_t* act_scales, const float* w_scales, int BK, int stage, int m_offset, int total_m, int reuse_act, float act_scale, float* residual) {
    (void)engine; (void)x; (void)slot_id; (void)y; (void)m; (void)k; (void)mode;
    (void)act_scales; (void)w_scales; (void)BK; (void)stage; (void)m_offset; (void)total_m;
    (void)reuse_act; (void)act_scale; (void)residual;
    return -1;
}

int v37_launch_linear_async(v37_engine_t* engine, const int8_t* x, int slot_id, int m, int k, v37_mode_t mode, int stage, int reuse_act) {
    (void)engine; (void)x; (void)slot_id; (void)m; (void)k; (void)mode; (void)stage; (void)reuse_act;
    return -1;
}

int v37_wait_linear_async(v37_engine_t* engine, void* y, const float* w_scales, float act_scale, float* residual) {
    (void)engine; (void)y; (void)w_scales; (void)act_scale; (void)residual;
    return -1;
}

int v37_wait_linear_async_fused_up(v37_engine_t* engine, float* ffn_act, const float* gate, const float* w_scales, float act_scale) {
    (void)engine; (void)ffn_act; (void)gate; (void)w_scales; (void)act_scale;
    return -1;
}

int v37_wait_linear_async_f16(v37_engine_t* engine, void* kv_cache_v_dst, const float* w_scales, float act_scale) {
    (void)engine; (void)kv_cache_v_dst; (void)w_scales; (void)act_scale;
    return -1;
}

int v37_load_rope_table(v37_engine_t* engine, const float* sin_table, const float* cos_table, int half_dim_beats) {
    (void)engine; (void)sin_table; (void)cos_table; (void)half_dim_beats;
    return 0;
}

int v37_load_microcode(v37_engine_t* engine, const uint32_t* ucode_words, int num_insts) {
    (void)engine; (void)ucode_words; (void)num_insts;
    return 0;
}

int v37_start_token(v37_engine_t* engine) {
    (void)engine;
    return 0;
}

int v37_preload_weights(v37_engine_t* engine, const uint8_t* data, const float* w_scales, size_t size, int slot_id, v37_mode_t mode, int k, int BK) {
    (void)engine; (void)data; (void)w_scales; (void)size; (void)slot_id; (void)mode; (void)k; (void)BK;
    return 0;
}

int v37_compute_linear(v37_engine_t* engine, const int8_t* x, const uint8_t* w, const float* w_scales, void* y, int m, int k, v37_mode_t mode, const int8_t* act_scales, int BK, int stage) {
    (void)engine; (void)x; (void)w; (void)w_scales; (void)y; (void)m; (void)k; (void)mode; (void)act_scales; (void)BK; (void)stage;
    return -1;
}

void v37_reset_cma_allocator(v37_engine_t* e, int slot_id) {
    (void)e; (void)slot_id;
}

void v37_load_loopback_rmsnorm_weights(v37_engine_t* engine, const float* weights, int dim) {
    (void)engine; (void)weights; (void)dim;
}

int v37_compute_ffn_loopback(v37_engine_t* engine, const int8_t* x, int gate_slot, int up_slot, int down_slot, float* out, int hidden_dim, int ffn_dim, const float* gate_scales, const float* up_scales, const float* down_scales, float act_scale, const float* sub_norm_weights, float* residual, v37_mode_t mod) {
    (void)sub_norm_weights; (void)mod;
    (void)engine; (void)x; (void)gate_slot; (void)up_slot; (void)down_slot; (void)out; (void)hidden_dim; (void)ffn_dim; (void)gate_scales; (void)up_scales; (void)down_scales; (void)act_scale; (void)residual;
    return -1;
}

int v37_preload_fused_gate_up(v37_engine_t* engine, int gate_slot, int up_slot, int fused_slot, int ffn_dim, int hidden_dim, v37_mode_t mode) {
    (void)engine; (void)gate_slot; (void)up_slot; (void)fused_slot; (void)ffn_dim; (void)hidden_dim; (void)mode;
    return 0;
}

int v37_preload_fused_gate_up_buffers(v37_engine_t* engine, const uint8_t* gate_base, const uint8_t* up_base, int fused_slot, int ffn_dim, int hidden_dim, v37_mode_t mode) {
    (void)engine; (void)gate_base; (void)up_base; (void)fused_slot; (void)ffn_dim; (void)hidden_dim; (void)mode;
    return 0;
}

int v37_launch_linear_async_gate_up(v37_engine_t* engine, const int8_t* x, int fused_slot, int m, int k, v37_mode_t mode) {
    (void)engine; (void)x; (void)fused_slot; (void)m; (void)k; (void)mode;
    return -1;
}

int v37_wait_linear_async_gate_up(v37_engine_t* engine, int8_t* qact_out, float* quant_s_out, const float* gate_scales, const float* up_scales, float act_scale, const float* sub_norm_weights, int ffn_dim, int act_type) {
    (void)engine; (void)qact_out; (void)quant_s_out; (void)gate_scales; (void)up_scales; (void)act_scale; (void)sub_norm_weights; (void)ffn_dim; (void)act_type;
    return -1;
}

int v37_compute_ffn_chain(v37_engine_t* engine, const int8_t* x, int gate_slot, int up_slot, int down_slot, int fused_gate_up_slot, float* out, int hidden_dim, int ffn_dim, const float* gate_scales, const float* up_scales, const float* down_scales, float act_scale, const float* sub_norm_weights, float* residual, v37_mode_t mod) {
    (void)engine; (void)x; (void)gate_slot; (void)up_slot; (void)down_slot; (void)fused_gate_up_slot; (void)out; (void)hidden_dim; (void)ffn_dim; (void)gate_scales; (void)up_scales; (void)down_scales; (void)act_scale; (void)sub_norm_weights; (void)residual; (void)mod;
    return -1;
}

int v37_launch_ffn_chain(v37_engine_t* engine, const int8_t* x, int gate_slot, int up_slot, int down_slot, int fused_gate_up_slot, int hidden_dim, int ffn_dim, const float* gate_scales, const float* up_scales, float act_scale, const float* sub_norm_weights, float* quant_s_out, v37_mode_t mod, int act_type) {
    (void)engine; (void)x; (void)gate_slot; (void)up_slot; (void)down_slot; (void)fused_gate_up_slot; (void)hidden_dim; (void)ffn_dim; (void)gate_scales; (void)up_scales; (void)act_scale; (void)sub_norm_weights; (void)quant_s_out; (void)mod; (void)act_type;
    return -1;
}

int v37_wait_ffn_chain_down(v37_engine_t* engine, float* out, const float* down_scales, float quant_s, float* residual) {
    (void)engine; (void)out; (void)down_scales; (void)quant_s; (void)residual;
    return -1;
}

void v37_sg_flush(v37_engine_t* engine) {
    (void)engine;
}

void* v37_get_vbase(v37_engine_t* engine) {
    (void)engine;
    return nullptr;
}

void* v37_get_cma_slot_ptr(v37_engine_t* engine, int slot_id) {
    (void)engine; (void)slot_id;
    return nullptr;
}

int v37_preload_fused_qkv_buffers(v37_engine_t* engine, const uint8_t* q_base, const uint8_t* k_base, const uint8_t* v_base, int fused_slot, int q_dim, int k_dim, int v_dim, int hidden_dim, v37_mode_t mode) {
    (void)engine; (void)q_base; (void)k_base; (void)v_base; (void)fused_slot; (void)q_dim; (void)k_dim; (void)v_dim; (void)hidden_dim; (void)mode;
    return 0;
}

int v37_preload_fused_direct(v37_engine_t* engine, const uint8_t* fused_data, size_t size, int fused_slot) {
    (void)engine; (void)fused_data; (void)size; (void)fused_slot;
    return 0;
}

int v37_compute_qkv_fused(v37_engine_t* engine, const int8_t* x, int fused_slot, float* q_out, float* k_out, void* v_f16_out, int q_dim, int k_dim, int v_dim, int hidden_dim, const float* q_scales, const float* k_scales, const float* v_scales, float act_scale, int pos, void* k_f16_out, const float* rope_cos, const float* rope_sin, float inv_sqrt_head_dim, v37_mode_t mod) {
    (void)engine; (void)x; (void)fused_slot; (void)q_out; (void)k_out; (void)v_f16_out; (void)q_dim; (void)k_dim; (void)v_dim; (void)hidden_dim; (void)q_scales; (void)k_scales; (void)v_scales; (void)act_scale; (void)pos; (void)k_f16_out; (void)rope_cos; (void)rope_sin; (void)inv_sqrt_head_dim; (void)mod;
    return -1;
}
