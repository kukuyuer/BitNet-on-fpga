#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <omp.h>
#include "llama.h"
#include "ggml-backend-v37.h"
#include "engine.h"
#include "loader.h"
#include "v37_driver.h"
#if defined(__aarch64__) || defined(__arm__)
#include <arm_neon.h>
#endif
#include <sys/resource.h>
#include <unistd.h>

int main(int argc, char** argv) {
    struct rlimit rl = {RLIM_INFINITY, RLIM_INFINITY};
    setrlimit(RLIMIT_MEMLOCK, &rl);

    std::string input_path = "";
    std::string prompt = "The hardware accelerator is";
    std::string v37pack_path = "";
    int threads = 4;
    int n_predict = 32;
    float repeat_penalty = 1.10f;
    int repeat_last_n = 64;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printf("Usage: %s [model_file] [options]\n", argv[0]);
            printf("   or: %s -m <model_file> [options]\n\n", argv[0]);
            printf("Options:\n");
            printf("  -m, --model <path>      Input model path (.v37p self-contained or .gguf)\n");
            printf("  -w, --weight <path>     Explicit pre-packed .v37p weight file (Format v3 dual-file)\n");
            printf("  -p, --prompt <text>     Input prompt string (default: \"The hardware accelerator is\")\n");
            printf("  -t, --threads <threads> CPU thread count (default: 4)\n");
            printf("  -n, --predict <tokens>  Max generated tokens (default: 32)\n");
            printf("  -r, --repeat-penalty <f> Repetition penalty (default: 1.10)\n");
            printf("  --repeat-last-n <n>     Repetition penalty window (default: 64)\n");
            printf("  -h, --help              Show this help message\n\n");
            printf("Environment Variables:\n");
            printf("  DISABLE_V37=1           Run on CPU reference mode without FPGA hardware\n");
            return 0;
        } else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            input_path = argv[++i];
        } else if ((arg == "-p" || arg == "--prompt") && i + 1 < argc) {
            prompt = argv[++i];
        } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        } else if ((arg == "-n" || arg == "--predict") && i + 1 < argc) {
            n_predict = std::stoi(argv[++i]);
        } else if ((arg == "-w" || arg == "--weight") && i + 1 < argc) {
            v37pack_path = argv[++i];
        } else if ((arg == "-r" || arg == "--repeat-penalty") && i + 1 < argc) {
            repeat_penalty = std::stof(argv[++i]);
        } else if (arg == "--repeat-last-n" && i + 1 < argc) {
            repeat_last_n = std::stoi(argv[++i]);
        } else if (!arg.empty() && arg[0] != '-') {
            if (input_path.empty()) {
                input_path = arg;
            } else {
                prompt = arg;
            }
        }
    }

    if (input_path.empty()) {
        printf("Error: Model file path not specified!\n");
        printf("Usage: %s <model.v37p> [options]\n", argv[0]);
        printf("   or: %s -m <model.v37p> [options]\n", argv[0]);
        printf("Try '%s --help' for details.\n", argv[0]);
        return 1;
    }

    printf("[MAIN] Initializing llama backend...\n");
    llama_backend_init();
    
    bool v37_enabled = (getenv("DISABLE_V37") == nullptr);
    
    if (v37_enabled) {
        printf("[MAIN] Registering V37 FPGA Backend...\n");
        ggml_backend_reg_t v37_reg = ggml_backend_v37_reg_debug();
        if (v37_reg) {
            ggml_backend_register(v37_reg);
            printf("[MAIN] V37 Backend registered. I2_S MUL_MAT will be handled by V37.\n");
        } else {
            printf("[MAIN] V37 Backend returned NULL, falling back to CPU.\n");
            v37_enabled = false;
        }
    } else {
        printf("[MAIN] DISABLE_V37 set. Using pure CPU mode (reference baseline).\n");
    }

    omp_set_num_threads(threads);

    V37Loader loader(nullptr);
    if (!loader.load_model(input_path)) {
        printf("[MAIN] ERROR: Failed to load model container from %s\n", input_path.c_str());
        return 1;
    }

    llama_model_params mparams = llama_model_default_params();
    struct llama_model * model = nullptr;

    if (loader.is_v37p_format()) {
        printf("[MAIN] Mode: Self-contained V37P unified container detected.\n");
        if (v37pack_path.empty()) {
            v37pack_path = input_path;
        }
        const uint8_t* vocab_data = loader.get_vocab_data();
        uint64_t vocab_size = loader.get_vocab_size();
        if (!vocab_data || vocab_size == 0) {
            printf("[MAIN] ERROR: V37P does not contain embedded vocab metadata! Need GGUF or v4 container.\n");
            return 1;
        }
        std::string temp_vocab_file = "/tmp/v37_vocab_" + std::to_string(getpid()) + ".gguf";
        FILE* fv = fopen(temp_vocab_file.c_str(), "wb");
        if (!fv) {
            printf("[MAIN] ERROR: Failed to create temp vocab file %s\n", temp_vocab_file.c_str());
            return 1;
        }
        fwrite(vocab_data, 1, vocab_size, fv);
        fclose(fv);

        mparams.vocab_only = true;
        model = llama_model_load_from_file(temp_vocab_file.c_str(), mparams);
        unlink(temp_vocab_file.c_str());
        if (!model) {
            printf("[MAIN] ERROR: Failed to load tokenizer vocab from embedded metadata!\n");
            return 1;
        }
        printf("[MAIN] Successfully initialized tokenizer from embedded V37P vocab (%zu KB).\n", (size_t)(vocab_size / 1024));
    } else {
        printf("[MAIN] Mode: Legacy GGUF model container detected (weights dispatched to %s)...\n",
               v37_enabled ? "V37" : "CPU");
        model = llama_model_load_from_file(input_path.c_str(), mparams);
        if (!model) {
            printf("[MAIN] Failed to load llama model from GGUF.\n");
            return 1;
        }
    }

    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    int real_vocab_size = llama_vocab_n_tokens(vocab);
    if (real_vocab_size > 0 && loader.get_config().vocab_size != real_vocab_size) {
        ModelConfig cfg = loader.get_config();
        cfg.vocab_size = real_vocab_size;
        loader.set_config(cfg);
    }
    
    v37_engine_t* engine_hw = v37_enabled ? v37_create_engine() : nullptr;
    if (v37_enabled && !engine_hw) {
        printf("[MAIN] Failed to init V37 engine, falling back to CPU.\n");
        v37_enabled = false;
    }
    
    V37Engine engine(engine_hw, &loader, model);
    
    // Load pre-packed .v37p weights for hardware (offline repacked)
    if (!v37pack_path.empty()) {
        if (engine.load_v37pack(v37pack_path)) {
            printf("[MAIN] Using pre-packed weights from %s\n", v37pack_path.c_str());
        } else {
            printf("[MAIN] WARNING: Failed to load v37pack, falling back to runtime conversion\n");
        }
    }

    printf("\n[%s] BitNet Inference Engine\n", v37_enabled ? "V37 FPGA" : "CPU REF");

    llama_token bos = llama_vocab_bos(vocab);
    llama_token eos = llama_vocab_eos(vocab);
    bool add_bos = (bos != eos) && (bos != -1);
    std::vector<llama_token> token_buf(prompt.length() + 16);
    int n_tok = llama_tokenize(vocab, prompt.c_str(), prompt.length(), token_buf.data(), token_buf.size(), add_bos, true);
    if (n_tok < 0) {
        token_buf.resize(-n_tok);
        n_tok = llama_tokenize(vocab, prompt.c_str(), prompt.length(), token_buf.data(), token_buf.size(), add_bos, true);
    }
    token_buf.resize(n_tok);

    std::vector<int> tokens;
    for (int t = 0; t < n_tok; t++) {
        tokens.push_back((int)token_buf[t]);
    }
    if (tokens.empty()) {
        tokens = {128000, 791, 12035, 65456, 374};
    }
    int n_tokens = tokens.size();
    
    printf("PROMPT: %s (add_bos: %s)\n", prompt.c_str(), add_bos ? "YES" : "NO");
    printf("TOKEN_COUNT: %d\n", n_tokens);
    for (int i=0; i<n_tokens; i++) {
        printf("  token[%d] = %d\n", i, tokens[i]);
    }
    printf("MODE: %s\n", v37_enabled ? "FPGA Accelerated" : "CPU Reference");
    printf("THREADS: %d\n", threads);
    printf("N_PREDICT: %d\n", n_predict);
    printf("REPEAT_PENALTY: %.2f (last_n=%d)\n", repeat_penalty, repeat_last_n);
    printf("---\nOUTPUT: ");
    fflush(stdout);
    
    auto prefill_start_t = std::chrono::high_resolution_clock::now();

    // Prefill prompt (skip LM_HEAD since intermediate prompt logits are discarded)
    for (size_t i = 0; i < tokens.size() - 1; i++) {
        std::vector<int> ctx_tokens = { (int)tokens[i] };
        engine.forward(ctx_tokens, i, false);
    }
    auto prefill_end_t = std::chrono::high_resolution_clock::now();
    double prefill_sec = std::chrono::duration<double>(prefill_end_t - prefill_start_t).count();

    int generated_tokens = 0;
    int target_len = tokens.size() + n_predict;
    
    std::vector<int> current_tokens;
    current_tokens.push_back(tokens.back());

    std::vector<int> all_tokens = tokens;
    int vocab_size = engine.get_config().vocab_size;
    std::vector<uint8_t> pen_applied(vocab_size, 0);
    std::string generated_text = "";
    
    auto decode_start_t = std::chrono::high_resolution_clock::now();
    double fwd_total_ms = 0.0;

    for (int i = tokens.size() - 1; i < target_len - 1; i++) {
        auto t_fwd0 = std::chrono::high_resolution_clock::now();
        float* logits = engine.forward(all_tokens, i);
        auto t_fwd1 = std::chrono::high_resolution_clock::now();
        fwd_total_ms += std::chrono::duration<double, std::milli>(t_fwd1 - t_fwd0).count();
        
        if (i == (int)(tokens.size() - 1) && getenv("DUMP_LOGITS")) {
            FILE* f = fopen("engine_logits.bin", "wb");
            fwrite(logits, sizeof(float), vocab_size, f);
            fclose(f);
            printf("[DEBUG] Dumped %d engine logits to engine_logits.bin\n", vocab_size);
        }
        
        // Fast Repetition Penalty (Zero heap allocation)
        if (repeat_penalty != 1.0f) {
            int start_idx = std::max(0, (int)all_tokens.size() - repeat_last_n);
            for (size_t p = start_idx; p < all_tokens.size(); p++) {
                int past_tok = all_tokens[p];
                if (past_tok >= 0 && past_tok < vocab_size && !pen_applied[past_tok]) {
                    pen_applied[past_tok] = 1;
                    if (logits[past_tok] <= 0) {
                        logits[past_tok] *= repeat_penalty;
                    } else {
                        logits[past_tok] /= repeat_penalty;
                    }
                }
            }
            for (size_t p = start_idx; p < all_tokens.size(); p++) {
                int past_tok = all_tokens[p];
                if (past_tok >= 0 && past_tok < vocab_size) {
                    pen_applied[past_tok] = 0;
                }
            }
        }
        
        int best_token = -1;
        float max_logit = -1e10f;
        const auto& cands = engine.get_candidates();
        if (!cands.empty()) {
            for (int tok : cands) {
                if (logits[tok] > max_logit) {
                    max_logit = logits[tok];
                    best_token = tok;
                }
            }
        } else {
#if defined(__aarch64__) || defined(__arm__)
            float32x4_t vmax_vec = vdupq_n_f32(-1e10f);
            int v = 0;
            for (; v <= vocab_size - 4; v += 4) {
                vmax_vec = vmaxq_f32(vmax_vec, vld1q_f32(logits + v));
            }
            max_logit = vmaxvq_f32(vmax_vec);
            for (; v < vocab_size; v++) {
                if (logits[v] > max_logit) max_logit = logits[v];
            }
            for (int j = 0; j < vocab_size; j++) {
                if (logits[j] == max_logit) { best_token = j; break; }
            }
#else
            for (int v = 0; v < vocab_size; v++) {
                if (logits[v] > max_logit) { max_logit = logits[v]; best_token = v; }
            }
#endif
        }
        
        if (getenv("DUMP_LOGITS")) {
            std::vector<std::pair<float, int>> sorted_logits(vocab_size);
            for (int v = 0; v < vocab_size; v++) sorted_logits[v] = {logits[v], v};
            int k_top = std::min(5, vocab_size);
            std::partial_sort(sorted_logits.begin(), sorted_logits.begin() + k_top, sorted_logits.end(),
                [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                    return a.first > b.first;
                });
            printf("[SAMPLER-DEBUG] i=%d best_token=%d Top %d: ", i, best_token, k_top);
            for(int k=0; k<k_top; k++) printf("tok=%d(%.4f) ", sorted_logits[k].second, sorted_logits[k].first);
            printf("\n");
        }
        all_tokens.push_back(best_token);
        
        if (llama_vocab_is_eog(vocab, best_token) || best_token == 128001 || best_token == 128009) {
            printf("\n\n[EOG] Token %d generated. Stopping.\n", best_token);
            break;
        }

        generated_tokens++;
    }

    auto decode_end_t = std::chrono::high_resolution_clock::now();
    for (size_t t = tokens.size(); t < all_tokens.size(); t++) {
        char buf[128];
        int n_chars = llama_token_to_piece(vocab, all_tokens[t], buf, sizeof(buf), 0, true);
        if (n_chars > 0) {
            generated_text += std::string(buf, n_chars);
        }
    }
    std::cout << generated_text << std::flush;
    double decode_sec = std::chrono::duration<double>(decode_end_t - decode_start_t).count();
    double total_sec = prefill_sec + decode_sec;

    printf("\n\n==== Performance Report ====\n");
    printf("Mode: %s\n", v37_enabled ? "V37 FPGA" : "CPU Reference");
    printf("Prefill:  %zu tokens in %.2f ms (%.2f TPS)\n", tokens.size() - 1, prefill_sec * 1000.0, (tokens.size() - 1) / prefill_sec);
    printf("Decode:   %d tokens in %.3f sec (%.2f ms/token)\n", generated_tokens, decode_sec, decode_sec * 1000.0 / generated_tokens);
    printf("Speed:    %.2f TPS (Pure Decode), %.2f TPS (End-to-End)\n", generated_tokens / decode_sec, generated_tokens / total_sec);
    printf("Forward:  avg %.2f ms/token (%.2f TPS)\n", fwd_total_ms / generated_tokens, 1000.0 / (fwd_total_ms / generated_tokens));
    printf("============================\n");
    
    extern int64_t t_dma, t_memcpy, t_setup, t_poll, t_calls, t_sync_cpu, t_post;
    printf("FPGA TIMING: dma=%ld ms (setup=%ld ms, poll=%ld ms), memcpy=%ld ms, sync_cpu=%ld ms, post=%ld ms. calls=%ld\n", t_dma/1000, t_setup/1000, t_poll/1000, t_memcpy/1000, t_sync_cpu/1000, t_post/1000, t_calls);

    // llama_free(ctx);
    // llama_model_free(model);
    // llama_backend_free();
    if (engine_hw) v37_destroy_engine(engine_hw);
    exit(0);
}
