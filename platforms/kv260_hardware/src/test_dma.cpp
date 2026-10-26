#include "v37_driver.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

int main() {
    printf("[TEST] Creating engine...\n");
    v37_engine_t* e = v37_create_engine();
    if (!e) {
        printf("[TEST] Failed to create engine\n");
        return 1;
    }
    printf("[TEST] Engine created.\n");
    
    // Allocate some space for weights in CMA
    printf("[TEST] Preloading 128 bytes of dummy weights...\n");
    uint8_t dummy_w[128] = {0};
    v37_preload_weights(e, dummy_w, NULL, 128, 0, V37_MODE_I2_S_PREPACKED, 64, 0);
    
    printf("[TEST] Preparing input...\n");
    int8_t dummy_x[64] = {0};
    float dummy_y[64] = {0};
    
    printf("[TEST] Running compute linear...\n");
    // This will trigger the DMA
    int ret = v37_compute_linear_preloaded(e, dummy_x, 0, dummy_y, 4, 64, V37_MODE_I2_S, NULL, NULL, 0, 0, 0, 4);
    
    printf("[TEST] Compute finished. ret = %d\n", ret);
    return 0;
}
