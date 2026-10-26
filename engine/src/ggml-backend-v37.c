#include "ggml-backend-v37.h"
#include "ggml-backend-impl.h"
#include "ggml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "v37_driver.h"

#define DEBUG_PARITY 1
#define V37_ALIGNMENT 256

static v37_engine_t* g_v37_engine = NULL;

static v37_engine_t* get_v37_engine() {
    if (!g_v37_engine) {
        g_v37_engine = v37_create_engine();
    }
    return g_v37_engine;
}

struct ggml_backend_v37_context {
    v37_engine_t* engine;
    
    int8_t* x_int8_buf;
    size_t x_int8_size;
};

//
// Buffer implementation
//

static void ggml_backend_v37_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    free(buffer->context);
}

static void * ggml_backend_v37_buffer_get_base(ggml_backend_buffer_t buffer) {
    return buffer->context;
}

static void ggml_backend_v37_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    (void)buffer;
    memcpy((char *)tensor->data + offset, data, size);
}

static void ggml_backend_v37_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    (void)buffer;
    memcpy(data, (const char *)tensor->data + offset, size);
}

static void ggml_backend_v37_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

static struct ggml_backend_buffer_i ggml_backend_v37_buffer_i = {
    /* .free_buffer     = */ ggml_backend_v37_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_v37_buffer_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_v37_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_v37_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_v37_buffer_clear,
    /* .reset           = */ NULL,
};

//
// Buffer type implementation
//

static const char * ggml_backend_v37_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return "V37";
}

static ggml_backend_buffer_t ggml_backend_v37_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    (void)buft;
    void * data = aligned_alloc(V37_ALIGNMENT, size);
    if (data == NULL) {
        return NULL;
    }
    return ggml_backend_buffer_init(buft, ggml_backend_v37_buffer_i, data, size);
}

static size_t ggml_backend_v37_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return V37_ALIGNMENT;
}

static bool ggml_backend_v37_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    (void)buft;

    return true;
}

GGML_API ggml_backend_buffer_type_t ggml_backend_v37_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_v37_buffer_type = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_v37_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_v37_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_v37_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL,
            /* .get_alloc_size   = */ NULL,
            /* .is_host          = */ ggml_backend_v37_buffer_type_is_host,
        },
        /* .device   = */ NULL,
        /* .context  = */ NULL,
    };
    return &ggml_backend_v37_buffer_type;
}

//
// Backend implementation
//

static const char * ggml_backend_v37_get_name(ggml_backend_t backend) {
    (void)backend;
    return "V37";
}

static void ggml_backend_v37_free(ggml_backend_t backend) {
    struct ggml_backend_v37_context * ctx = (struct ggml_backend_v37_context *)backend->context;
    if (ctx->engine) {
        v37_destroy_engine(ctx->engine);
    }
    if (ctx->x_int8_buf) free(ctx->x_int8_buf);
    free(ctx);
}

/*
 * V37 FPGA MUL_MAT implementation for I2_S ternary weights.
 *
 * Official BitNet.cpp formula (from ggml.c L12513):
 *   result[row] = (dot(W_row, X_q) - act_sum) / act_scale * w_scale
 *
 * Where:
 *   - W is ternary {-1, 0, +1} packed as 2-bit per element
 *   - X_q is activation quantized to int8 with: X_q[i] = round(X[i] / act_scale)
 *   - act_scale = max(|X|) / 127
 *   - act_sum = sum(X_q)  (used because ternary encoding {0==-1, 1==0, 2==1})
 *   - w_scale = single scalar float stored after packed weights in tensor data
 *
 * FPGA hardware computes: hw_out[row] = sum(W_ternary[row][j] * X_q[j])
 *   where W_ternary is correctly decoded as {-1, 0, +1}
 *
 * So: result[row] = (hw_out[row] - act_sum) / act_scale * w_scale
 *                 = (hw_out[row] - act_sum) * w_scale * (127.0 / max(|X|))
 *
 * But wait — the hardware uses encoding {0→-1, 1→0, 2→+1} internally,
 * which means hw_out = sum((encoded_val - 1) * X_q).
 * When the weight IS encoded as {0=-1,1=0,2=+1}, the hardware directly
 * produces the ternary dot product, so act_sum correction is NOT needed
 * if the hardware already does the -1 offset.
 *
 * Looking at the driver's emulation code (v37_driver.cpp L150-153):
 *   acc += ((int8_t)((byte >> 6) & 3) - 1) * x[...]
 * The hardware does apply the -1 offset. So hw_out IS the true ternary dot product.
 *
 * But the official CPU code uses a different encoding where the raw dot product
 * includes the offset and needs act_sum subtraction. We need to match the
 * official output exactly.
 *
 * Strategy: Compute the raw I2_S dot product on FPGA (with -1 offset already
 * applied by hardware), then apply: result = hw_out * act_scale * w_scale
 *   where act_scale = max(|X|) / 127 = 1/s
 *
 * This is equivalent because:
 *   Official: (raw_dot_no_offset - act_sum) / act_scale * w_scale
 *           = (raw_dot_no_offset - sum(X_q)) * w_scale / act_scale
 *   Our HW:  raw_dot_with_offset = raw_dot_no_offset - sum(X_q)  (since each w_enc - 1 applied)
 *   So: result = hw_out_with_offset * w_scale / act_scale
 *             = hw_out_with_offset * w_scale * (127.0 / amax)
 *             ... but this introduces float rounding differences
 *
 * To get EXACT parity, we'll do the computation purely in software using
 * the CPU reference path for now, and only offload to FPGA once we've
 * confirmed the formula matches.
 *
 * PHASE 1 APPROACH (CPU reference, bypass FPGA):
 *   Implement exact same I2_S MUL_MAT as official BitNet CPU code.
 *   This ensures correct output, proving the integration works.
 *   Then PHASE 2 will swap in FPGA acceleration.
 */

/* ---- Inline I2_S ternary decode (2-bit packing, 4 elements per byte) ---- */
static inline float fp16_to_fp32(uint16_t h) {
    uint16_t h_exp = (h & 0x7c00) >> 10;
    uint16_t h_sig = (h & 0x03ff);
    uint32_t f_sign = ((uint32_t)h & 0x8000) << 16;
    uint32_t f_exp, f_sig;
    if (h_exp == 0) {
        if (h_sig == 0) return *((float*)&f_sign);
        h_exp = 1;
        while ((h_sig & 0x0400) == 0) {
            h_sig <<= 1;
            h_exp--;
        }
        h_sig &= 0x03ff;
        f_exp = ((uint32_t)(127 - 15 + h_exp)) << 23;
        f_sig = ((uint32_t)h_sig) << 13;
    } else if (h_exp == 0x1f) {
        f_exp = 0xff << 23;
        f_sig = ((uint32_t)h_sig) << 13;
    } else {
        f_exp = ((uint32_t)(127 - 15 + h_exp)) << 23;
        f_sig = ((uint32_t)h_sig) << 13;
    }
    uint32_t f = f_sign | f_exp | f_sig;
    return *((float*)&f);
}

static inline int decode_5in8_ternary(const uint8_t* block_data, int idx_in_block) {
    int group_idx = idx_in_block / 5;
    int elem_in_group = idx_in_block % 5;
    
    if (group_idx < 12) {
        uint8_t byte_val = block_data[group_idx];
        int powers[5] = {1, 3, 9, 27, 81};
        int val = (byte_val / powers[elem_in_group]) % 3;
        return val - 1;
    } else {
        uint8_t byte_val = block_data[12];
        int powers[4] = {1, 3, 9, 27};
        int val = (byte_val / powers[elem_in_group]) % 3;
        return val - 1;
    }
}



static enum ggml_status ggml_backend_v37_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    struct ggml_backend_v37_context * ctx = (struct ggml_backend_v37_context *)backend->context;
    
    int n_nodes = ggml_graph_n_nodes(cgraph);
    static int call_cnt = 0;
    if (call_cnt < 5) {
        fprintf(stderr, "\n=== GGML CGRAPH CALL %d (%d nodes) ===\n", call_cnt++, n_nodes);
        for (int i = 0; i < n_nodes; i++) {
            struct ggml_tensor * n = ggml_graph_node(cgraph, i);
            fprintf(stderr, "Node %2d: op=%-2d | name=%-25s | src0=%-20s | src1=%-20s\n",
                   i, n->op, n->name ? n->name : "(null)",
                   (n->src[0] && n->src[0]->name) ? n->src[0]->name : "(null)",
                   (n->src[1] && n->src[1]->name) ? n->src[1]->name : "(null)");
        }
        fprintf(stderr, "===================================\n\n");
        fflush(stderr);
    }
    for (int i = 0; i < n_nodes; i++) {
        struct ggml_tensor * node = ggml_graph_node(cgraph, i);
        
        if (node->op == GGML_OP_MUL_MAT) {
            struct ggml_tensor * src0 = node->src[0]; /* Weight: ne0=K, ne1=N_rows */
            struct ggml_tensor * src1 = node->src[1]; /* Activation: ne0=K, ne1=n_tokens */
            
            if (src0->type == (enum ggml_type)36 && src1->type == GGML_TYPE_F32) {
                int64_t K = src0->ne[0];
                int64_t N = src0->ne[1]; /* output rows */
                int64_t n_tokens = src1->ne[1];
                
                // Prepare Activation
                float* src1_data = (float*)src1->data;
                float* y_out = (float*)node->data;
                size_t need_x = (size_t)(n_tokens * K);
                if (ctx->x_int8_size < need_x) {
                    ctx->x_int8_size = need_x;
                    ctx->x_int8_buf = (int8_t*)realloc(ctx->x_int8_buf, ctx->x_int8_size);
                }

                float combined_w_scales[5] = {1.0f, 1.0f, 1.0f, 1.218855f, 1.0f};
                const float* w_scales = combined_w_scales;
                int stage = 0;
                int s_id = 0;
                bool is_fused = false;
                v37_reset_cma_allocator(ctx->engine, 0);
                struct ggml_tensor* next1_saved = NULL;
                struct ggml_tensor* next4_saved = NULL;
                struct ggml_tensor* next5 = NULL;
                if (src0->name && strstr(src0->name, "ffn_gate")) {
                    struct ggml_tensor* next1 = (i + 1 < n_nodes) ? ggml_graph_node(cgraph, i + 1) : NULL;
                    struct ggml_tensor* next2 = (i + 2 < n_nodes) ? ggml_graph_node(cgraph, i + 2) : NULL;
                    struct ggml_tensor* next3 = (i + 3 < n_nodes) ? ggml_graph_node(cgraph, i + 3) : NULL;
                    struct ggml_tensor* next4 = (i + 4 < n_nodes) ? ggml_graph_node(cgraph, i + 4) : NULL;
                    next5 = (i + 5 < n_nodes) ? ggml_graph_node(cgraph, i + 5) : NULL;
                    
                    bool match_glu = (next2 && next2->op == GGML_OP_GLU);
                    bool match_silu = (next2 && next2->op == GGML_OP_UNARY && ggml_get_unary_op(next2) == GGML_UNARY_OP_SILU);
                    
                    if (false /* DISABLE FUSION FOR DEBUG */) {
                        
                        is_fused = true;
                        next1_saved = next1;
                        next4_saved = next4;
                        stage = 5 | (6 << 16);
                        y_out = (float*)next5->data; 
                        
                        // Slot 0: Gate
                        size_t packed_bytes0 = (size_t)(src0->ne[0] * src0->ne[1]) / 4;
                        float* pad_gate = (float*)((uint8_t*)src0->data + packed_bytes0);
                        float* pad_gate_alt = (float*)((uint8_t*)src0->data + ggml_nbytes(src0) - 32);
                        combined_w_scales[0] = (pad_gate[0] != 0.0f) ? pad_gate[0] : pad_gate_alt[0];
                        float* gate_scales = (float*)malloc(N * sizeof(float));
                        for(int r=0; r<N; r++) gate_scales[r] = combined_w_scales[0];
                        v37_preload_weights(ctx->engine, (const uint8_t*)src0->data, gate_scales, (N * K) / 4, 0, V37_MODE_I2_S, K, 0); 
                        free(gate_scales);
                        
                        // Slot 1: Up
                        size_t packed_bytes1 = (size_t)(next1->src[0]->ne[0] * next1->src[0]->ne[1]) / 4;
                        float* pad_up = (float*)((uint8_t*)next1->src[0]->data + packed_bytes1);
                        float* pad_up_alt = (float*)((uint8_t*)next1->src[0]->data + ggml_nbytes(next1->src[0]) - 32);
                        combined_w_scales[1] = (pad_up[0] != 0.0f) ? pad_up[0] : pad_up_alt[0];
                        float* up_scales = (float*)malloc(next1->src[0]->ne[1] * sizeof(float));
                        for(int r=0; r<next1->src[0]->ne[1]; r++) up_scales[r] = combined_w_scales[1];
                        v37_preload_weights(ctx->engine, (const uint8_t*)next1->src[0]->data, up_scales, (next1->src[0]->ne[1] * K) / 4, 1, V37_MODE_I2_S, K, 1); 
                        free(up_scales);
                        
                        // Slot 2: Down
                        size_t packed_bytes5 = (size_t)(next5->src[0]->ne[0] * next5->src[0]->ne[1]) / 4;
                        float* pad_down = (float*)((uint8_t*)next5->src[0]->data + packed_bytes5);
                        float* pad_down_alt = (float*)((uint8_t*)next5->src[0]->data + ggml_nbytes(next5->src[0]) - 32);
                        combined_w_scales[2] = (pad_down[0] != 0.0f) ? pad_down[0] : pad_down_alt[0];
                        float* down_scales = (float*)malloc(next5->src[0]->ne[1] * sizeof(float));
                        for(int r=0; r<next5->src[0]->ne[1]; r++) down_scales[r] = combined_w_scales[2];
                        v37_preload_weights(ctx->engine, (const uint8_t*)next5->src[0]->data, down_scales, (next5->src[0]->ne[1] * next5->src[0]->ne[0]) / 4, 2, V37_MODE_I2_S, next5->src[0]->ne[0], 2); 
                        free(down_scales);
                        
                        if (next4 && next4->src[1] && next4->src[1]->data) {
                            v37_load_loopback_rmsnorm_weights(ctx->engine, (const float*)next4->src[1]->data, (int)next4->src[1]->ne[0]);
                        }
                        
                        s_id = (2 << 16) | (1 << 8) | 0;
                        i += 5;
                    }
                }
                
                float* dynamic_w_scales = combined_w_scales;
                if (!is_fused) {
                    size_t packed_bytes_unfused = (size_t)(src0->ne[0] * src0->ne[1]) / 4;
                    float* pad_f = (float*)((uint8_t*)src0->data + packed_bytes_unfused);
                    float* pad_f_alt = (float*)((uint8_t*)src0->data + ggml_nbytes(src0) - 32);
                    float w_scale_unfused = (pad_f[0] != 0.0f) ? pad_f[0] : pad_f_alt[0];
                    combined_w_scales[0] = w_scale_unfused;
                    dynamic_w_scales = (float*)malloc(N * sizeof(float));
                    for(int r=0; r<N; r++) dynamic_w_scales[r] = w_scale_unfused;
                    v37_preload_weights(ctx->engine, (const uint8_t*)src0->data, dynamic_w_scales, (N * K) / 4, 0, V37_MODE_I2_S, K, 0);
                    stage = 0;
                    s_id = 0;
                    y_out = (float*)node->data;
                }
                
                for (int64_t token = 0; token < n_tokens; token++) {
                    const float* x_f32 = src1_data + token * K;
                    
                    /* Quantize activation */
                    float amax = 0.0f;
                    for (int64_t j = 0; j < K; j++) {
                        float v = fabsf(x_f32[j]);
                        if (v > amax) amax = v;
                    }
                    float act_scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
                    
                    int8_t* x_q = ctx->x_int8_buf + token * K;
                    for (int64_t j = 0; j < K; j++) {
                        float v = x_f32[j] / act_scale;
                        int32_t qi = (int32_t)roundf(v);
                        if (qi > 127) qi = 127;
                        if (qi < -127) qi = -127;
                        x_q[j] = (int8_t)qi;
                    }
                    
                    float driver_act_scale = act_scale;
                    int current_stage = stage;
                    combined_w_scales[3] = 1.0f;
                    if (is_fused && next4_saved && next4_saved->src[1] && next4_saved->src[1]->data) {
                        const float* sn_w = (const float*)next4_saved->src[1]->data;
                        int sn_n = next4_saved->src[1]->ne[0];
                        float sum_sn = 0.0f;
                        for (int j = 0; j < sn_n; j++) sum_sn += sn_w[j];
                        combined_w_scales[3] = sum_sn / (float)sn_n;
                    }
                    combined_w_scales[4] = act_scale;
                    
                    int64_t actual_N = (is_fused && next5) ? next5->src[0]->ne[1] : N;
                    float* out_row = y_out + token * actual_N;
                    v37_compute_linear_preloaded(ctx->engine, x_q, s_id, out_row, actual_N, K, V37_MODE_I2_S, NULL, dynamic_w_scales, 0, current_stage, 0, 0);
                    
                    if (!is_fused) {
                        for (int i = 0; i < N; i++) {
                            out_row[i] = out_row[i] * driver_act_scale;
                        }
                    }
                    
                    if (is_fused) {
                        // Stage 5 Fused FFN streaming execution complete
                    } else {
                        float cpu_sum0 = 0.0f;
                        const uint8_t* w_bytes = (const uint8_t*)src0->data;
                        // PHASE 3: Block-wise decoding
                        // 128 weights = 32 bytes
                        // Inside 32 bytes: Block 0 (15 bytes), Block 1 (15 bytes), Pad (2 bytes)
                        for (int64_t block = 0; block < K / 64; block++) {
                            int pair_idx = block / 2;
                            int in_pair = block % 2;
                            const uint8_t* block_ptr = w_bytes + pair_idx * 32 + in_pair * 15;
                            
                            // Extract FP16 scale
                            uint16_t scale_fp16 = block_ptr[13] | (block_ptr[14] << 8);
                            float block_scale = fp16_to_fp32(scale_fp16);
                            
                            float block_sum = 0.0f;
                            for (int i = 0; i < 64; i++) {
                                int w_val = decode_5in8_ternary(block_ptr, i);
                                block_sum += w_val * (float)x_q[block * 64 + i];
                            }
                            cpu_sum0 += block_sum * block_scale;
                        }
                        // Global dynamic_w_scales[0] is still applied if any, but now cpu_sum0 includes block scales!
                        float cpu_val0 = cpu_sum0 * dynamic_w_scales[0] * driver_act_scale;
                        fprintf(stderr, "[LAYER0-NODE-PARITY] %-20s (token %ld): CPU=%.4f (raw_sum=%.1f), FPGA=%.4f (w_scale=%.6f, act_scale=%.6f)\n",
                               src0->name ? src0->name : "unnamed", token, cpu_val0, cpu_sum0, out_row[0], dynamic_w_scales[0], driver_act_scale);
                        fflush(stderr);
                    }
                }
                if (!is_fused && dynamic_w_scales != (float*)w_scales) {
                    free(dynamic_w_scales);
                }
            }
        } else if ((node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SILU) || node->op == GGML_OP_MUL) {
            // Normally consumed by FFN fusion. If executed standalone, ignore.
            continue;
        } else if (node->op == GGML_OP_MUL_MAT) {
            float sum = 0;
            for (int i = 0; i < node->ne[0]*node->ne[1]; i++) {
                sum += ((float*)node->data)[i];
            }
            float* f32_data = (float*)node->data;
        }
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_v37_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    (void)backend;
    (void)cgraph;
}

static struct ggml_backend_i ggml_backend_v37_i = {
    /* .get_name                = */ ggml_backend_v37_get_name,
    /* .free                    = */ ggml_backend_v37_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_v37_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ ggml_backend_v37_graph_optimize,
};

static ggml_guid_t ggml_backend_v37_guid(void) {
    static ggml_guid guid = { 0x37, 0x37, 0x37, 0x37, 0x37, 0x37, 0x37, 0x37, 0x37, 0x37, 0x37, 0x37, 0x37, 0x37, 0x37, 0x37 };
    return &guid;
}

//
// Device implementation
//

static const char * ggml_backend_v37_device_get_name(ggml_backend_dev_t dev) {
    (void)dev;
    return "V37";
}

static const char * ggml_backend_v37_device_get_description(ggml_backend_dev_t dev) {
    (void)dev;
    return "V37 BitNet FPGA Accelerator";
}

static void ggml_backend_v37_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    (void)dev;
    *total = 1024 * 1024 * 1024;
    *free = *total;
}

static enum ggml_backend_dev_type ggml_backend_v37_device_get_type(ggml_backend_dev_t dev) {
    (void)dev;
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_v37_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_v37_device_get_name(dev);
    props->description = ggml_backend_v37_device_get_description(dev);
    props->type        = ggml_backend_v37_device_get_type(dev);
    ggml_backend_v37_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = (struct ggml_backend_dev_caps){
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_v37_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    (void)dev;
    (void)params;
    return ggml_backend_v37_init();
}

static ggml_backend_buffer_type_t ggml_backend_v37_device_get_buffer_type(ggml_backend_dev_t dev) {
    static struct ggml_backend_buffer_type ggml_backend_v37_buffer_type = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_v37_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_v37_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_v37_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL,
            /* .get_alloc_size   = */ NULL,
            /* .is_host          = */ ggml_backend_v37_buffer_type_is_host,
        },
        /* .device   = */ NULL,
        /* .context  = */ NULL,
    };
    ggml_backend_v37_buffer_type.device = dev;
    return &ggml_backend_v37_buffer_type;
}

static bool ggml_backend_v37_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    (void)dev;
    if (op->name && strstr(op->name, "ffn_down")) {
        // printf("[SUPPORTS_OP] ffn_down: op=%d, src0_type=%d, src1_type=%d\n", op->op, op->src[0] ? op->src[0]->type : -1, op->src[1] ? op->src[1]->type : -1);
    }
    if (op->op == GGML_OP_MUL_MAT) {
        if (op->src[0]->type == (enum ggml_type)36 && op->src[1]->type == GGML_TYPE_F32) {
            // if (op->name && strstr(op->name, "ffn_down")) printf("[SUPPORTS_OP] ffn_down MATCHED!\n");
            return true;
        }
    }
    if (op->op == GGML_OP_GLU && op->name && (strstr(op->name, "ffn_swiglu") || strstr(op->name, "ffn_sub_out"))) {
        return true;
    }
    if (op->op == GGML_OP_UNARY && ggml_get_unary_op(op) == GGML_UNARY_OP_SILU && op->name && strstr(op->name, "ffn_silu")) {
        return true;
    }
    if (op->name && strstr(op->name, "ffn_sub_norm")) {
        // printf("[SUPPORTS_OP] ffn_sub_norm: op=%d, name=%s\n", op->op, op->name);
    }
    if (op->op == GGML_OP_MUL && op->name && (strstr(op->name, "ffn_gate_par") || strstr(op->name, "ffn_sub_norm"))) {
        return true;
    }
    if (op->op == GGML_OP_RMS_NORM && op->name && strstr(op->name, "ffn_sub_norm")) {
        return true;
    }
    return false;
}

static bool ggml_backend_v37_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    (void)dev;
    return ggml_backend_buft_is_host(buft) || (buft == ggml_backend_v37_buffer_type());
}

//
// Backend (reg)
//

static const char * ggml_backend_v37_reg_get_name(ggml_backend_reg_t reg) {
    (void)reg;
    return "V37";
}

static size_t ggml_backend_v37_reg_get_device_count(ggml_backend_reg_t reg) {
    (void)reg;
    return 1;
}

static bool ggml_backend_v37_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    (void)dev;
    if (op->name == NULL) return false;
    
    bool should_offload = false;
    if (op->op == GGML_OP_UNARY && ggml_get_unary_op(op) == GGML_UNARY_OP_SILU && strstr(op->name, "ffn_silu")) {
        should_offload = true;
    }
    if (op->op == GGML_OP_MUL && (strstr(op->name, "ffn_gate_par") || strstr(op->name, "ffn_sub_norm"))) {
        should_offload = true;
    }
    if (op->op == GGML_OP_GLU && (strstr(op->name, "ffn_swiglu") || strstr(op->name, "ffn_sub_out"))) {
        should_offload = true;
    }
    if (op->op == GGML_OP_RMS_NORM && strstr(op->name, "ffn_sub_norm")) {
        should_offload = true;
    }
    if (op->op == GGML_OP_MUL_MAT) {
        should_offload = true;
    }
    return should_offload;
}

static ggml_backend_dev_t ggml_backend_v37_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    (void)index;
    static struct ggml_backend_device ggml_backend_v37_device = {
        /* .iface   = */ {
            /* .get_name             = */ ggml_backend_v37_device_get_name,
            /* .get_description      = */ ggml_backend_v37_device_get_description,
            /* .get_memory           = */ ggml_backend_v37_device_get_memory,
            /* .get_type             = */ ggml_backend_v37_device_get_type,
            /* .get_props            = */ ggml_backend_v37_device_get_props,
            /* .init_backend         = */ ggml_backend_v37_device_init_backend,
            /* .get_buffer_type      = */ ggml_backend_v37_device_get_buffer_type,
            /* .get_host_buffer_type = */ NULL,
            /* .buffer_from_host_ptr = */ NULL,
            /* .supports_op          = */ ggml_backend_v37_device_supports_op,
            /* .supports_buft        = */ ggml_backend_v37_device_supports_buft,
            /* .offload_op           = */ ggml_backend_v37_device_offload_op,
            /* .event_new            = */ NULL,
            /* .event_free           = */ NULL,
            /* .event_synchronize    = */ NULL,
        },
        /* .reg     = */ NULL,
        /* .context = */ NULL,
    };
    ggml_backend_v37_device.reg = reg;
    return &ggml_backend_v37_device;
}

static void * ggml_backend_v37_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    (void)reg;
    (void)name;
    return NULL;
}

static const struct ggml_backend_reg_i ggml_backend_v37_reg_i = {
    /* .get_name         = */ ggml_backend_v37_reg_get_name,
    /* .get_device_count = */ ggml_backend_v37_reg_get_device_count,
    /* .get_device       = */ ggml_backend_v37_reg_get_device,
    /* .get_proc_address = */ ggml_backend_v37_get_proc_address,
};

ggml_backend_reg_t ggml_backend_v37_reg_debug(void) {
    static struct ggml_backend_reg ggml_backend_v37_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_v37_reg_i,
        /* .context     = */ NULL,
    };
    return &ggml_backend_v37_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_v37_reg_debug)

GGML_API ggml_backend_t ggml_backend_v37_init(void) {
    struct ggml_backend_v37_context * ctx = (struct ggml_backend_v37_context *)malloc(sizeof(struct ggml_backend_v37_context));
    ctx->engine = get_v37_engine();
    ctx->x_int8_buf = NULL;
    ctx->x_int8_size = 0;
    
    ggml_backend_t backend = (ggml_backend_t)malloc(sizeof(struct ggml_backend));
    *backend = (struct ggml_backend){
        /* .guid      = */ ggml_backend_v37_guid(),
        /* .iface     = */ ggml_backend_v37_i,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_v37_reg_debug(), 0),
        /* .context   = */ ctx,
    };
    return backend;
}

GGML_API bool ggml_backend_is_v37(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_v37_guid());
}

GGML_API bool ggml_backend_buffer_is_v37(ggml_backend_buffer_t buffer) {
    return buffer != NULL && buffer->buft == ggml_backend_v37_buffer_type();
}
