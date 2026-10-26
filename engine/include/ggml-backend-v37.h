#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// Backend API
GGML_API ggml_backend_t ggml_backend_v37_init(void);
GGML_API bool ggml_backend_is_v37(ggml_backend_t backend);
GGML_API ggml_backend_reg_t ggml_backend_v37_reg(void);
GGML_API ggml_backend_reg_t ggml_backend_v37_reg_debug(void);

// Buffer type API
GGML_API ggml_backend_buffer_type_t ggml_backend_v37_buffer_type(void);

// Helper to check if a buffer is a V37 buffer
GGML_API bool ggml_backend_buffer_is_v37(ggml_backend_buffer_t buffer);

#ifdef  __cplusplus
}
#endif
