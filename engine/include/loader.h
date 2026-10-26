#ifndef V37_LOADER_H
#define V37_LOADER_H
#include <string>
#include <vector>
#include <map>
#include <stdint.h>
#include "v37_driver.h"
#include "model_config.h"

struct V37Tensor {
    uint64_t offset;
    uint64_t size;
    uint64_t elements;
    uint32_t type;
    uint32_t n_dims;
    uint64_t ne[4];
};

class V37Loader {
public:
    V37Loader(v37_engine_t* e);
    ~V37Loader();
    bool load_model(const std::string& path);
    float* get_embd(int token);
    std::string get_token_text(int token);
    int get_token_id(const std::string& text);
    bool load_weight_to_slot(int layer_idx, const std::string& tensor_name, int slot_id);
    void* get_tensor_data(const std::string& name);
    uint32_t get_tensor_type(const std::string& name);
    const float* get_i2_scales(const std::string& name);
    uint32_t get_tensor_phys(const std::string& name);
    void load_matrix_to_cma(int l, int t);
    
    const ModelConfig& get_config() const { return config; }
    void set_config(const ModelConfig& cfg) { config = cfg; }
    const V37Tensor* get_tensor_info(const std::string& name) const;
    const std::map<std::string, V37Tensor>& get_all_tensors() const { return tensors; }

    bool is_v37p_format() const { return is_v37p; }
    uint64_t get_vocab_offset() const { return vocab_gguf_offset; }
    uint64_t get_vocab_size() const { return vocab_gguf_size; }
    const uint8_t* get_vocab_data() const { return (is_v37p && vocab_gguf_offset > 0) ? ((const uint8_t*)mmap_addr + vocab_gguf_offset) : nullptr; }

private:
    void infer_config_from_metadata_and_tensors();

    v37_engine_t* engine;
    void* mmap_addr;
    size_t mmap_size;
    bool is_v37p;
    uint64_t vocab_gguf_offset;
    uint64_t vocab_gguf_size;
    std::vector<std::string> vocab;
    std::map<std::string, V37Tensor> tensors;
    std::vector<float> embd_cache;
    std::map<std::string, std::vector<float>> scale_cache;
    ModelConfig config;
};
#endif
