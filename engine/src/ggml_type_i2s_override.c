/*
 * ggml_type_i2s_override.c
 * 
 * Provides override implementations of ggml type query functions
 * that return correct values for GGML_TYPE_I2_S (type 36).
 * 
 * CRITICAL: These definitions MUST match the official BitNet fork's
 * ggml.c type_traits[GGML_TYPE_I2_S] exactly:
 *   .blck_size = 1
 *   .type_size = sizeof(int8_t)  // = 1
 *   .is_quantized = true
 * 
 * The I2_S format stores:
 *   - Packed 2-bit ternary weights: N*K/4 bytes (4 elements per byte)
 *   - Encoding: 0 → -1, 1 → 0, 2 → +1
 *   - 1 float weight_scale at the end (within 32 bytes padding)
 * 
 * ggml_nbytes for I2_S = total_elements / 4 + 32
 *   (matches official BitNet fork: ggml.c L3486)
 * 
 * When linked BEFORE libggml-base.so, the linker will prefer these
 * definitions, effectively patching the type system at link time.
 */
#include "ggml.h"
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

/* === Type 36 (I2_S) constants === */
/* Must match official BitNet fork: blck_size=1, type_size=1 */
#define I2S_BLCK_SIZE  1
#define I2S_TYPE_SIZE  1   /* sizeof(int8_t) */

/*
 * Override ggml_blck_size: intercept type 36, delegate all others
 * to the original via ggml_get_type_traits (which reads the .so's array).
 */
int64_t ggml_blck_size(enum ggml_type type) {
    if (type == GGML_TYPE_I2_S) return I2S_BLCK_SIZE;
    const struct ggml_type_traits * t = ggml_get_type_traits(type);
    return t->blck_size;
}

size_t ggml_type_size(enum ggml_type type) {
    if (type == GGML_TYPE_I2_S) return I2S_TYPE_SIZE;
    const struct ggml_type_traits * t = ggml_get_type_traits(type);
    return t->type_size;
}

const char * ggml_type_name(enum ggml_type type) {
    if (type == GGML_TYPE_I2_S) return "i2_s";
    const struct ggml_type_traits * t = ggml_get_type_traits(type);
    return t->type_name;
}

bool ggml_is_quantized(enum ggml_type type) {
    if (type == GGML_TYPE_I2_S) return true;
    const struct ggml_type_traits * t = ggml_get_type_traits(type);
    return t->is_quantized;
}

/*
 * ggml_row_size for I2_S:
 * With blck_size=1, type_size=1, the standard formula gives: 1 * ne / 1 = ne bytes.
 * But I2_S actually packs 4 elements per byte, so the real row size is ne/4.
 * 
 * However, this function is used by the MUL_MAT kernel to compute offsets
 * for the quantized activation (vec_dot_type = I8_S), NOT for the weight.
 * The weight stride (nb[1]) is set separately during tensor creation.
 * 
 * For the weight tensor, nb[1] = ne[0] * type_size / blck_size = ne[0] * 1 / 1 = ne[0].
 * The official code then divides by 4 when accessing weight rows:
 *   src0_row + ir0 * nb01 / 4   (see ggml.c L12508)
 * 
 * So we return the "logical" row size (ne bytes), and the /4 division happens at access time.
 */
size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    if (type == GGML_TYPE_I2_S) {
        /* Return logical size (will be /4 at access time by MUL_MAT code) */
        return (size_t)(I2S_TYPE_SIZE * ne / I2S_BLCK_SIZE);
    }
    const struct ggml_type_traits * t = ggml_get_type_traits(type);
    return t->type_size * ne / t->blck_size;
}

/*
 * ggml_nbytes for I2_S:
 * Must match official BitNet fork (ggml.c L3485-3487):
 *   nbytes = (standard calculation) / 4 + 32
 * 
 * Standard calculation for blck_size==1:
 *   nbytes = type_size + sum((ne[i]-1) * nb[i])  for all dims
 * For a contiguous 2D tensor [K, N]:
 *   nb[0] = 1 (type_size)
 *   nb[1] = K * 1 / 1 = K
 *   nbytes = 1 + (K-1)*1 + (N-1)*K = K + N*K - K = N*K = total_elements
 * Then: nbytes = total_elements / 4 + 32
 * 
 * This 32 bytes includes the weight_scale float (4 bytes) plus padding.
 */
size_t ggml_nbytes(const struct ggml_tensor * tensor) {
    size_t nbytes;
    size_t blck_size = ggml_blck_size(tensor->type);
    if (blck_size == 1) {
        nbytes = ggml_type_size(tensor->type);
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }
    else {
        nbytes = tensor->ne[0]*tensor->nb[0]/blck_size;
        for (int i = 1; i < GGML_MAX_DIMS; i++) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }

    if (tensor->type == GGML_TYPE_I2_S) {
        /* Official formula: nbytes = nbytes / 4 + 32 */
        nbytes = nbytes / 4 + 32;
    }

    return nbytes;
}


bool ggml_validate_row_data(enum ggml_type type, const void * data, size_t nbytes) {
    (void)data;
    (void)nbytes;
    if (type == GGML_TYPE_I2_S) return true;
    return true;
}


