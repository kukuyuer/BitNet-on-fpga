#ifndef V37_MODEL_CONFIG_H
#define V37_MODEL_CONFIG_H

#include <stdint.h>
#include <cstdio>

struct ModelConfig {
    int max_seq_len = 0;
    int n_embd      = 0;
    int head_dim    = 0;
    int n_heads     = 0;
    int n_kv_heads  = 0;
    int n_layers    = 0;
    int n_ffn       = 0;
    int vocab_size  = 0;
    float rope_freq_base = 0.0f;
    float rms_norm_eps   = 0.0f;
    bool  has_sub_norm   = true;
    bool  use_silu       = false;

    void print() const {
        printf("[MODEL-CONFIG] Dynamic Parameters:\n");
        printf("  - n_layers   : %d\n", n_layers);
        printf("  - n_embd     : %d\n", n_embd);
        printf("  - head_dim   : %d\n", head_dim);
        printf("  - n_heads    : %d (Q)\n", n_heads);
        printf("  - n_kv_heads : %d (KV, GQA ratio: %d)\n", n_kv_heads, n_kv_heads > 0 ? (n_heads / n_kv_heads) : 1);
        printf("  - n_ffn      : %d\n", n_ffn);
        printf("  - vocab_size : %d\n", vocab_size);
        printf("  - max_seq_len: %d\n", max_seq_len);
        printf("  - rope_base  : %.1f\n", rope_freq_base);
        printf("  - rms_eps    : %.2e\n", rms_norm_eps);
        printf("  - sub_norm   : %s\n", has_sub_norm ? "YES" : "NO");
        printf("  - activation : %s\n", use_silu ? "SwiGLU (SiLU)" : "Squared-ReLU");
    }
};

#endif // V37_MODEL_CONFIG_H
