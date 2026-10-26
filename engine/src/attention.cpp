#include "engine.h"
#include <cmath>
#include <algorithm>
#if defined(__aarch64__) || defined(__arm__)
#include <arm_neon.h>
#endif

void V37Engine::init_rope_table() {
    if (rope_table_inited) return;
    int max_p = config.max_seq_len;
    int half_dim = config.head_dim / 2;
    rope_cos_table.resize(max_p * half_dim);
    rope_sin_table.resize(max_p * half_dim);

    float base = (config.rope_freq_base > 0.0f) ? config.rope_freq_base : 500000.0f;
    for (int p = 0; p < max_p; p++) {
        for (int i = 0; i < half_dim; i++) {
            float freq = 1.0f / powf(base, (float)(2 * i) / (float)config.head_dim);
            float val = (float)p * freq;
            rope_cos_table[p * half_dim + i] = cosf(val);
            rope_sin_table[p * half_dim + i] = sinf(val);
        }
    }
    rope_table_inited = true;
}

void V37Engine::rope_q(float* q, int pos) {
    init_rope_table();
    if (pos >= config.max_seq_len) pos = config.max_seq_len - 1;
    int half_dim = config.head_dim / 2;
    const float* fcr = rope_cos_table.data() + pos * half_dim;
    const float* fci = rope_sin_table.data() + pos * half_dim;
    float inv_sqrt = 1.0f / std::sqrt((float)config.head_dim);
    bool is_neox = config.has_sub_norm;

#if defined(__aarch64__) || defined(__arm__)
    float32x4_t vinv = vdupq_n_f32(inv_sqrt);
#endif

    for (int h = 0; h < config.n_heads; h++) {
        float* q_head = q + h * config.head_dim;
#if defined(__aarch64__) || defined(__arm__)
        if (is_neox) {
            for (int i = 0; i < half_dim; i += 4) {
                float32x4_t q0 = vld1q_f32(q_head + i);
                float32x4_t q1 = vld1q_f32(q_head + i + half_dim);
                float32x4_t cr = vld1q_f32(fcr + i);
                float32x4_t ci = vld1q_f32(fci + i);

                float32x4_t out0 = vmulq_f32(vsubq_f32(vmulq_f32(q0, cr), vmulq_f32(q1, ci)), vinv);
                float32x4_t out1 = vmulq_f32(vaddq_f32(vmulq_f32(q0, ci), vmulq_f32(q1, cr)), vinv);

                vst1q_f32(q_head + i, out0);
                vst1q_f32(q_head + i + half_dim, out1);
            }
        } else {
            for (int i = 0; i < half_dim; i += 4) {
                float32x4x2_t q_pair = vld2q_f32(q_head + 2 * i);
                float32x4_t cr = vld1q_f32(fcr + i);
                float32x4_t ci = vld1q_f32(fci + i);

                float32x4x2_t out_pair;
                out_pair.val[0] = vmulq_f32(vsubq_f32(vmulq_f32(q_pair.val[0], cr), vmulq_f32(q_pair.val[1], ci)), vinv);
                out_pair.val[1] = vmulq_f32(vaddq_f32(vmulq_f32(q_pair.val[0], ci), vmulq_f32(q_pair.val[1], cr)), vinv);

                vst2q_f32(q_head + 2 * i, out_pair);
            }
        }
#else
        if (is_neox) {
            for (int i = 0; i < half_dim; i++) {
                float q0 = q_head[i];
                float q1 = q_head[i + half_dim];
                q_head[i]            = (q0 * fcr[i] - q1 * fci[i]) * inv_sqrt;
                q_head[i + half_dim] = (q0 * fci[i] + q1 * fcr[i]) * inv_sqrt;
            }
        } else {
            for (int i = 0; i < half_dim; i++) {
                float q0 = q_head[2 * i];
                float q1 = q_head[2 * i + 1];
                q_head[2 * i]     = (q0 * fcr[i] - q1 * fci[i]) * inv_sqrt;
                q_head[2 * i + 1] = (q0 * fci[i] + q1 * fcr[i]) * inv_sqrt;
            }
        }
#endif
    }
}

void V37Engine::rope_k(float* k, int pos) {
    init_rope_table();
    if (pos >= config.max_seq_len) pos = config.max_seq_len - 1;
    int half_dim = config.head_dim / 2;
    const float* fcr = rope_cos_table.data() + pos * half_dim;
    const float* fci = rope_sin_table.data() + pos * half_dim;
    bool is_neox = config.has_sub_norm;

    for (int h = 0; h < config.n_kv_heads; h++) {
        float* k_head = k + h * config.head_dim;
#if defined(__aarch64__) || defined(__arm__)
        if (is_neox) {
            for (int i = 0; i < half_dim; i += 4) {
                float32x4_t k0 = vld1q_f32(k_head + i);
                float32x4_t k1 = vld1q_f32(k_head + i + half_dim);
                float32x4_t cr = vld1q_f32(fcr + i);
                float32x4_t ci = vld1q_f32(fci + i);

                float32x4_t out0 = vsubq_f32(vmulq_f32(k0, cr), vmulq_f32(k1, ci));
                float32x4_t out1 = vaddq_f32(vmulq_f32(k0, ci), vmulq_f32(k1, cr));

                vst1q_f32(k_head + i, out0);
                vst1q_f32(k_head + i + half_dim, out1);
            }
        } else {
            for (int i = 0; i < half_dim; i += 4) {
                float32x4x2_t k_pair = vld2q_f32(k_head + 2 * i);
                float32x4_t cr = vld1q_f32(fcr + i);
                float32x4_t ci = vld1q_f32(fci + i);

                float32x4x2_t out_pair;
                out_pair.val[0] = vsubq_f32(vmulq_f32(k_pair.val[0], cr), vmulq_f32(k_pair.val[1], ci));
                out_pair.val[1] = vaddq_f32(vmulq_f32(k_pair.val[0], ci), vmulq_f32(k_pair.val[1], cr));

                vst2q_f32(k_head + 2 * i, out_pair);
            }
        }
#else
        if (is_neox) {
            for (int i = 0; i < half_dim; i++) {
                float k0 = k_head[i];
                float k1 = k_head[i + half_dim];
                k_head[i]            = k0 * fcr[i] - k1 * fci[i];
                k_head[i + half_dim] = k0 * fci[i] + k1 * fcr[i];
            }
        } else {
            for (int i = 0; i < half_dim; i++) {
                float k0 = k_head[2 * i];
                float k1 = k_head[2 * i + 1];
                k_head[2 * i]     = k0 * fcr[i] - k1 * fci[i];
                k_head[2 * i + 1] = k0 * fci[i] + k1 * fcr[i];
            }
        }
#endif
    }
}

void V37Engine::rope_k_f16(const float* k, void* kv_cache_k, int pos) {
    init_rope_table();
    if (pos >= config.max_seq_len) pos = config.max_seq_len - 1;
    int half_dim = config.head_dim / 2;
    const float* fcr = rope_cos_table.data() + pos * half_dim;
    const float* fci = rope_sin_table.data() + pos * half_dim;
    bool is_neox = config.has_sub_norm;
#if defined(__aarch64__) || defined(__arm__)
    __fp16* dst_f16 = (__fp16*)kv_cache_k;

    for (int h = 0; h < config.n_kv_heads; h++) {
        const float* k_head = k + h * config.head_dim;
        __fp16* dst_head = dst_f16 + h * config.head_dim;
        if (is_neox) {
            for (int i = 0; i < half_dim; i += 4) {
                float32x4_t k0 = vld1q_f32(k_head + i);
                float32x4_t k1 = vld1q_f32(k_head + i + half_dim);
                float32x4_t cr = vld1q_f32(fcr + i);
                float32x4_t ci = vld1q_f32(fci + i);

                float32x4_t out0 = vsubq_f32(vmulq_f32(k0, cr), vmulq_f32(k1, ci));
                float32x4_t out1 = vaddq_f32(vmulq_f32(k0, ci), vmulq_f32(k1, cr));

                vst1_f16(dst_head + i, vcvt_f16_f32(out0));
                vst1_f16(dst_head + i + half_dim, vcvt_f16_f32(out1));
            }
        } else {
            for (int i = 0; i < half_dim; i += 4) {
                float32x4x2_t k_pair = vld2q_f32(k_head + 2 * i);
                float32x4_t cr = vld1q_f32(fcr + i);
                float32x4_t ci = vld1q_f32(fci + i);

                float16x4x2_t h_pair;
                h_pair.val[0] = vcvt_f16_f32(vsubq_f32(vmulq_f32(k_pair.val[0], cr), vmulq_f32(k_pair.val[1], ci)));
                h_pair.val[1] = vcvt_f16_f32(vaddq_f32(vmulq_f32(k_pair.val[0], ci), vmulq_f32(k_pair.val[1], cr)));

                vst2_f16(dst_head + 2 * i, h_pair);
            }
        }
    }
#else
    uint16_t* dst_f16 = (uint16_t*)kv_cache_k;

    for (int h = 0; h < config.n_kv_heads; h++) {
        const float* k_head = k + h * config.head_dim;
        uint16_t* dst_head = dst_f16 + h * config.head_dim;
        if (is_neox) {
            for (int i = 0; i < half_dim; i++) {
                float k0 = k_head[i];
                float k1 = k_head[i + half_dim];
                dst_head[i]            = ggml_fp32_to_fp16(k0 * fcr[i] - k1 * fci[i]);
                dst_head[i + half_dim] = ggml_fp32_to_fp16(k0 * fci[i] + k1 * fcr[i]);
            }
        } else {
            for (int i = 0; i < half_dim; i++) {
                float k0 = k_head[2 * i];
                float k1 = k_head[2 * i + 1];
                dst_head[2 * i]     = ggml_fp32_to_fp16(k0 * fcr[i] - k1 * fci[i]);
                dst_head[2 * i + 1] = ggml_fp32_to_fp16(k0 * fci[i] + k1 * fcr[i]);
            }
        }
    }
#endif
}

// Implement rope with precomputed frequency and cos/sin cache
void V37Engine::rope(float* q, float* k, int pos) {
    rope_q(q, pos);
    rope_k(k, pos);
}
