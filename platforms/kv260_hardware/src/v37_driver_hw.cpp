#include <sched.h>
#include "v37_driver.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <cmath>
#if defined(__aarch64__) || defined(__arm__)
#include <arm_neon.h>
#endif
#include <omp.h>
std::unordered_map<const uint8_t*, uint32_t> g_cma_offsets;

#define VCORE_BASE      0xA0000000UL
#define DMA_X_BASE      0xA0010000UL

struct block_i2_s {
    uint16_t beta;
    uint8_t padding[14];
    uint8_t qs[16];
};

typedef struct {
    uint32_t phys_addr;
    int valid;
} v37_weight_slot_t;

#include <sys/ioctl.h>

struct v37_compute_cmd {
    uint32_t cfg_val;
    uint32_t pe_out_beats;
    uint32_t scale_bits;
    uint32_t x_phys;
    uint32_t x_len;
    uint32_t y_phys;
    uint32_t y_len;
    uint32_t w_phys[4];
    uint32_t w_len[4];
    float    total_scale;
    uint8_t  do_soft_reset;
    uint8_t  mm2s_active;
    uint8_t  s2mm_active;
};
#define V37_IOCTL_COMPUTE _IOW('V', 1, struct v37_compute_cmd)

struct v37_engine {
    int fd_mem; int fd_udma; int fd_v37;
    int fd_sync_dev; int fd_sync_cpu;
    volatile uint32_t* vcore;
    volatile uint32_t* dma_x;
    volatile uint32_t* dma_w[4];
    volatile uint32_t* sg_dispatcher;
    v37_sg_desc_t* desc_pool;
    uint32_t desc_count;
    
    uint8_t*  vbase;
    uint32_t  phys;
    size_t    udma_size;
    v37_weight_slot_t w_slots[512];
    uint32_t ws_next_off;
    
    void* cma_vbase;
    uint32_t current_cma_offset;
    uint32_t slot_offsets[512];
    
    struct {
        int m;
        int k;
        int s_id;
        int stage;
        uint32_t s2mm_out_beats;
        uint32_t y_off;
        std::chrono::high_resolution_clock::time_point start_t;
        std::chrono::high_resolution_clock::time_point t_before_poll;
        uint64_t start_vct;
        bool active;
    } async_job;
    bool ucode_mode;
};

static inline void wreg(volatile uint32_t* base, uint32_t off, uint32_t val) { 
    if (base != MAP_FAILED && base != nullptr) { base[off >> 2] = val; }
}
static inline uint32_t rreg(volatile uint32_t* base, uint32_t off) { 
    if (base == MAP_FAILED || base == nullptr) return 0xFFFFFFFB;
    return base[off >> 2]; 
}

static inline uint64_t arm64_get_cntvct() {
#if defined(__aarch64__)
    uint64_t val;
    asm volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
#else
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
#endif
}

static inline uint64_t arm64_get_cntfrq() {
#if defined(__aarch64__)
    uint64_t val;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(val));
    return val;
#else
    return 1000000000ULL;
#endif
}

static uint64_t g_ticks_per_us = 100; // Default: 100MHz = 100 ticks per microsecond
static float g_pl_clk_scale = 1.0f;    // PL clock scaling factor relative to 100MHz
static inline void init_timer_freq() {
    static bool inited = false;
    if (!inited) {
        uint64_t frq = arm64_get_cntfrq();
        if (frq >= 1000000) {
            g_ticks_per_us = frq / 1000000;
        } else {
            g_ticks_per_us = 100;
        }
        inited = true;
    }
}

static inline void cpu_relax() {
#if defined(__aarch64__) || defined(__arm__)
    // 256 NOPs backoff + yield: lowers CPU load and power consumption while yielding core execution slots
    for (int p = 0; p < 256; p++) {
        asm volatile("nop");
    }
    asm volatile("yield" ::: "memory");
#elif defined(__x86_64__)
    for (int p = 0; p < 32; p++) {
        asm volatile("pause");
    }
#endif
}

static inline void hybrid_wait_s2mm(v37_engine_t* e, std::chrono::high_resolution_clock::time_point start_t, int m = 2048, int k = 2048, uint64_t start_vct = 0) {
    (void)e;
    init_timer_freq();

    static int low_power_mode = -1;
    if (low_power_mode < 0) {
        const char* env = getenv("V37_LOW_POWER_WAIT");
        // Default is 0 (zero-syscall high performance spin with yield backoff)
        low_power_mode = (env && atoi(env) == 1) ? 1 : 0;
    }
    if (low_power_mode) {
        auto now = std::chrono::high_resolution_clock::now();
        int elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - start_t).count();
        if (m > 6912) {
            if (elapsed < 350) {
                usleep(250);
            }
        } else if (m > 2048 || k > 2048) {
            if (elapsed < 180) {
                usleep(100);
            }
        } else {
            if (elapsed < 40) {
                usleep(30);
            }
        }
        return;
    }

    static int spin_fast = -1;
    if (spin_fast == -1) {
        const char* fsp = getenv("V37_SPIN_FAST");
        spin_fast = (fsp && atoi(fsp) > 0) ? atoi(fsp) : 118;
    }
    if (spin_fast == 1) {
        // Direct zero-delay hardware status register polling (bypasses CPU safe_spin idle delay)
        return;
    }

    // Zero-syscall predictive spin wait (Phase 1):
    // In the first safe time window, avoid flooding AXI-Lite MMIO reads to keep bus free for DMA
    uint64_t safe_spin_us = 0;
    if (m > 6912) {
        safe_spin_us = (uint64_t)(1300 * g_pl_clk_scale); // Fused Gate+Up (13824x2560)
    } else if (k > 4096) {
        safe_spin_us = (uint64_t)(640 * g_pl_clk_scale);  // Down GEMV (2560x6912)
    } else if (m > 3000) {
        safe_spin_us = (uint64_t)(340 * g_pl_clk_scale);  // Fused QKV (3840x2560)
    } else if (m >= 2048) {
        safe_spin_us = (uint64_t)(230 * g_pl_clk_scale);  // Standard GEMV (2560x2560)
    } else {
        safe_spin_us = (uint64_t)(50 * g_pl_clk_scale);   // Attention K, V (640x2560)
    }

    if (spin_fast > 1) {
        safe_spin_us = (safe_spin_us * spin_fast) / 100;
    }

    uint64_t elapsed_us = 0;
    if (start_vct > 0) {
        uint64_t cur_vct = arm64_get_cntvct();
        elapsed_us = (cur_vct > start_vct) ? ((cur_vct - start_vct) / g_ticks_per_us) : 0;
    } else {
        auto now = std::chrono::high_resolution_clock::now();
        elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(now - start_t).count();
    }

    if (elapsed_us < safe_spin_us) {
        uint64_t remain_ticks = (safe_spin_us - elapsed_us) * g_ticks_per_us;
        uint64_t spin_start = arm64_get_cntvct();
        while ((arm64_get_cntvct() - spin_start) < remain_ticks) {
            cpu_relax();
        }
    }
}



v37_engine_t* v37_create_engine() {
    init_timer_freq();
    v37_engine_t* e = (v37_engine_t*)calloc(1, sizeof(v37_engine_t));
    e->fd_v37 = open("/dev/v37", O_RDWR);
    if (e->fd_v37 < 0) {
        printf("[V37 WARNING] Failed to open /dev/v37, will use fallback MMIO if possible.\n");
    }
    
    e->fd_mem = open("/dev/mem", O_RDWR | O_SYNC);
    if (e->fd_mem < 0) {
        perror("[V37 ERROR] Failed to open /dev/mem");
    }
    e->vcore = (volatile uint32_t*)mmap(NULL, 0x1000, PROT_READ|PROT_WRITE, MAP_SHARED, e->fd_mem, VCORE_BASE);
    if (e->vcore == MAP_FAILED) perror("[V37 ERROR] Failed to mmap VCORE_BASE");
    
    e->dma_x = (volatile uint32_t*)mmap(NULL, 0x1000, PROT_READ|PROT_WRITE, MAP_SHARED, e->fd_mem, DMA_X_BASE);
    
    for(int i=0; i<4; i++) {
        e->dma_w[i] = (volatile uint32_t*)mmap(NULL, 0x1000, PROT_READ|PROT_WRITE, MAP_SHARED, e->fd_mem, 0xA0020000 + i*0x10000);
        if (e->dma_w[i] == MAP_FAILED) perror("[V37 ERROR] Failed to mmap DMA_W");
    }
    // Force sg_dispatcher to MAP_FAILED to avoid AXI bus crashes on unmapped or buggy addresses
    e->sg_dispatcher = (volatile uint32_t*)MAP_FAILED;
    e->fd_udma = open("/dev/udmabuf0", O_RDWR);
    if (e->fd_udma < 0) perror("[V37 ERROR] Failed to open /dev/udmabuf0");
    
    if (getenv("V37_DEBUG")) printf("[V37 INIT] e->vcore=%p, e->dma_x=%p, MAP_FAILED=%p\n", e->vcore, e->dma_x, MAP_FAILED);

    // Auto-detect PL clock frequency from ZynqMP CRL_APB PL0_REF_CTRL (0xFF5E00C0)
    if (e->fd_mem >= 0) {
        volatile uint32_t* crl_apb = (volatile uint32_t*)mmap(NULL, 0x1000, PROT_READ, MAP_SHARED, e->fd_mem, 0xFF5E0000);
        if (crl_apb != MAP_FAILED && crl_apb != NULL) {
            uint32_t pl0_ctrl = crl_apb[0xC0 >> 2];
            uint32_t div0 = (pl0_ctrl >> 8) & 0x3F;
            if (div0 > 0 && div0 <= 63) {
                float pl_mhz = 1500.0f / (float)div0;
                g_pl_clk_scale = (float)div0 / 15.0f; // 15=100MHz (1.0), 12=125MHz (0.8), 11=136.36MHz (0.733)
                printf("[V37] Auto-detected PL0 clock: %.2f MHz (div0=%u, spin_scale=%.3f)\n", pl_mhz, div0, g_pl_clk_scale);
            }
            munmap((void*)crl_apb, 0x1000);
        }
    }
    
    // Reset AXI DMAs to clear any stuck state from previous killed processes
    if (e->dma_x != MAP_FAILED) {
        wreg(e->dma_x, 0x00, 0x04); // Reset MM2S
        wreg(e->dma_x, 0x30, 0x04); // Reset S2MM
    }
    for(int i=0; i<4; i++) {
        if (e->dma_w[i] != MAP_FAILED) {
            wreg(e->dma_w[i], 0x00, 0x04); // Reset MM2S
        }
    }
    // Wait for reset to complete
    for (volatile int delay = 0; delay < 10000; delay++) {}

    // Pre-start DMA channels: set Run/Stop (RS=1). In Direct Register mode, RS stays 1 indefinitely.
    if (e->dma_x != MAP_FAILED) {
        wreg(e->dma_x, 0x00, 0x01); // Start MM2S channel (RS=1)
        wreg(e->dma_x, 0x30, 0x01); // Start S2MM channel (RS=1)
    }
    for (int i = 0; i < 4; i++) {
        if (e->dma_w[i] != MAP_FAILED) {
            wreg(e->dma_w[i], 0x00, 0x01); // Start MM2S channel (RS=1)
        }
    }

    if (e->vcore != MAP_FAILED && e->vcore) {
        wreg(e->vcore, 0x10, 0x01000000);
        wreg(e->vcore, 0x14, 0x01000000);
    }

    char buf[128];
    int fd_p = open("/sys/class/u-dma-buf/udmabuf0/phys_addr", O_RDONLY);
    if (fd_p >= 0) {
        if (read(fd_p, buf, 63) > 0) {
            e->phys = (uint32_t)strtoull(buf, NULL, 16);
        } else {
            e->phys = 0x38c00000;
        }
        close(fd_p);
    } else {
        e->phys = 0x38c00000;
    }
    if (getenv("V37_DEBUG")) printf("[V37-INIT] Parsed udmabuf0 phys addr: 0x%X\n", e->phys);
    int fd_s = open("/sys/class/u-dma-buf/udmabuf0/size", O_RDONLY);
    if (fd_s >= 0) { (void)!read(fd_s, buf, 63); e->udma_size = (size_t)strtoull(buf, NULL, 0); close(fd_s); }
    if (e->udma_size == 0) e->udma_size = 650UL * 1024 * 1024;
    e->vbase = (uint8_t*)mmap(NULL, e->udma_size, PROT_READ|PROT_WRITE, MAP_SHARED, e->fd_udma, 0);
    // e->vbase = (uint8_t*)mmap(NULL, e->udma_size, PROT_READ|PROT_WRITE, MAP_SHARED, e->fd_mem, e->phys);
    if (e->vbase == MAP_FAILED) {
        printf("[V37-WARNING] Failed to map /dev/mem for udmabuf. Using malloc for simulation.\n");
        e->vbase = (uint8_t*)malloc(e->udma_size);
    }
    e->ws_next_off = 0x800000u;
    e->cma_vbase = e->vbase;
    e->current_cma_offset = 0x800000u;
    e->desc_pool = (v37_sg_desc_t*)(e->vbase + 0x200000u);
    e->desc_count = 0;
    e->ucode_mode = false;
    return e;
}

void* v37_get_vbase(v37_engine_t* engine) {
    if (!engine) return NULL;
    return engine->vbase;
}

int v37_load_microcode(v37_engine_t* e, const uint32_t* ucode_words, int num_insts) {
    if (!e || e->vcore == MAP_FAILED || !ucode_words || num_insts <= 0 || num_insts > 256) return -1;
    e->ucode_mode = true;
    // Program total instruction count (0x004)
    wreg(e->vcore, 0x04, (uint32_t)(num_insts & 0xFF));
    // Write microcode memory directly via 0x400 window (0x400 / 4 = 256)
    volatile uint32_t* ucode_window = e->vcore + (0x400 >> 2);
    for (int i = 0; i < num_insts * 4; i++) {
        ucode_window[i] = ucode_words[i];
    }
    uint32_t status = rreg(e->vcore, 0x00);
    printf("[V37-UCODE] Loaded %d microcode instructions into FPGA BRAM (status=0x%08X)\n", num_insts, status);
    return 0;
}

int v37_start_token(v37_engine_t* e) {
    if (!e || e->vcore == MAP_FAILED) return -1;
    wreg(e->vcore, 0x00, 0x01); // 1-cycle auto-clearing token_start pulse (resets PC to 0)
    return 0;
}
#include <sys/ioctl.h>

struct udmabuf_sync {
    int sync_direction;
    int sync_offset;
    int sync_size;
};
#define UDMABUF_SYNC_FOR_CPU        _IOW('U', 2, struct udmabuf_sync)
#define UDMABUF_SYNC_FOR_DEVICE     _IOW('U', 3, struct udmabuf_sync)

static inline void cpu_clean_dcache(const void* ptr, size_t size) {
#if defined(__aarch64__)
    const char* p = (const char*)ptr;
    for (size_t i = 0; i < size; i += 64) {
        __asm__ volatile("dc cvac, %0" : : "r"(p + i) : "memory");
    }
    __asm__ volatile("dsb sy" ::: "memory");
#elif defined(__x86_64__)
    __asm__ volatile("mfence" ::: "memory");
#endif
}

static inline void cpu_invalidate_dcache(const void* ptr, size_t size) {
#if defined(__aarch64__)
    const char* p = (const char*)ptr;
    for (size_t i = 0; i < size; i += 64) {
        __asm__ volatile("dc civac, %0" : : "r"(p + i) : "memory");
    }
    __asm__ volatile("dsb sy" ::: "memory");
#elif defined(__x86_64__)
    __asm__ volatile("mfence" ::: "memory");
#endif
}

static void sync_cma_for_device(v37_engine_t* e, uint32_t offset, uint32_t size) {
    if (e && e->vbase) {
        cpu_clean_dcache((const uint8_t*)e->vbase + offset, size);
    }
}

void sync_cma_for_cpu(v37_engine_t* e, uint32_t offset, uint32_t size) {
    if (e && e->vbase) {
        cpu_invalidate_dcache((const uint8_t*)e->vbase + offset, size);
    }
}

int64_t t_prep = 0, t_sync_dev = 0, t_dma = 0, t_sync_cpu = 0, t_post = 0, t_calls = 0;
int64_t t_memcpy = 0;
int64_t t_setup = 0, t_poll = 0;

int v37_compute_linear_preloaded_opt(v37_engine_t* e, const int8_t* x, int s_id, void* y, int m, int k, v37_mode_t mod, const int8_t* act_scales, const float* w_scales, int BK, int stage, int m_offset, int total_m, int reuse_act, float act_scale, float* residual);

int v37_compute_linear_preloaded(v37_engine_t* e, const int8_t* x, int s_id, void* y, int m, int k, v37_mode_t mod, const int8_t* act_scales, const float* w_scales, int BK, int stage, int m_offset, int total_m) {
    return v37_compute_linear_preloaded_opt(e, x, s_id, y, m, k, mod, act_scales, w_scales, BK, stage, m_offset, total_m, 0, 1.0f, NULL);
}

int v37_compute_linear_preloaded_opt(v37_engine_t* e, const int8_t* x, int s_id, void* y, int m, int k, v37_mode_t mod, const int8_t* act_scales, const float* w_scales, int BK, int stage, int m_offset, int total_m, int reuse_act, float act_scale, float* residual) {
    if (e == NULL) return -1;
    
    float w_scale_val = w_scales ? w_scales[0] : 1.0f;
    uint32_t stage_base_tmp = stage & 0xF;
    int check_slot = (stage_base_tmp == 5) ? (s_id & 0xFF) : ((stage_base_tmp == 4) ? (s_id & 0xFFFF) : s_id);
    if (!e || check_slot < 0 || check_slot >= 512 || e->slot_offsets[check_slot] == 0) return -2;
    
    const char* emu_env = getenv("V37_EMU");
    if (emu_env && strcmp(emu_env, "0") != 0) {
        float* out = (float*)y;
        uint32_t kb = (k + 63) / 64;
        uint32_t total_beats = kb;
        uint32_t ch_stride = total_beats * 16; 
        
        uint8_t* dest = (uint8_t*)e->cma_vbase + e->slot_offsets[s_id];
        float w_scale_val = w_scales ? w_scales[0] : 1.0f;
        
        for (int row = 0; row < m; row++) {
            int ch_idx = row % 4;
            int row_in_ch = row / 4;
            uint32_t ch_base = ch_idx * ((m + 3) / 4) * ch_stride + row_in_ch * ch_stride;
            
            int64_t total_acc = 0;
            uint32_t pld_idx = 0;
            uint32_t dest_beat = 0;
            
            while (pld_idx < kb) {
                for (int b = 0; b < 4 && pld_idx < kb; b++) {
                    uint8_t* dst_ptr = dest + ch_base + dest_beat * 16;
                    int32_t sum_B = 0;
                    int32_t sum_A = 0;
                    for (int xi = 0; xi < 16; xi++) {
                        uint8_t byte = dst_ptr[xi];
                        int w0 = ((int8_t)((byte >> 6) & 3) - 1);
                        int w1 = ((int8_t)((byte >> 4) & 3) - 1);
                        int w2 = ((int8_t)((byte >> 2) & 3) - 1);
                        int w3 = ((int8_t)((byte >> 0) & 3) - 1);
                        if (pld_idx * 64 + xi + 0  < (uint32_t)k) sum_A += w0 * x[pld_idx * 64 + xi + 0];
                        if (pld_idx * 64 + xi + 16 < (uint32_t)k) sum_A += w1 * x[pld_idx * 64 + xi + 16];
                        if (pld_idx * 64 + xi + 32 < (uint32_t)k) sum_A += w2 * x[pld_idx * 64 + xi + 32];
                        if (pld_idx * 64 + xi + 48 < (uint32_t)k) sum_A += w3 * x[pld_idx * 64 + xi + 48];
                    }
                    
                    int block_A_idx = (pld_idx * 64) / 32;
                    int block_B_idx = (pld_idx * 64 + 32) / 32;
                    
                    int shift_A = (act_scales ? act_scales[block_A_idx] : 0) + 16;
                    int shift_B = (act_scales ? act_scales[block_B_idx] : 0) + 16;
                    
                    int64_t shifted_A = shift_A >= 0 ? ((int64_t)sum_A << shift_A) : ((int64_t)sum_A >> (-shift_A));
                    int64_t shifted_B = shift_B >= 0 ? ((int64_t)sum_B << shift_B) : ((int64_t)sum_B >> (-shift_B));
                    
                    total_acc += shifted_A + shifted_B;
                    
                    dest_beat++; pld_idx++;
                }
            }
            if (row == 0 && (s_id == 0 || s_id == 2) && getenv("V37_DEBUG")) {
            }
            out[row] = (float)(total_acc / 65536);
        }
        
        static int printed_emu = 0;
        if (printed_emu < 5 && getenv("V37_DEBUG")) {
            printed_emu++;
        }
        return 0;
    }

    t_calls++;
    // start_t moved into chunk loop
    
    uint32_t k_beats = (k + 63) / 64;
    k_beats = (k_beats + 3) & ~3; // Pad to multiple of 4 for hardware state machine
    
    if (stage == 0) {
        // printf("[DEBUG-COMPUTE] e=%p, e->dma_x=%p, e->vcore=%p\n", e, e->dma_x, e->vcore);
    }
    
    // Chunk m to prevent BRAM overflow
    bool comp_en_global = (mod & (1 << 24));
    int chunk_size = (BK == 0) ? 16384 : BK; // Set chunk_size to 16384, N_hidden fits in one chunk.
    int num_chunks = (m + chunk_size - 1) / chunk_size;
    // printf("[DEBUG-COMPUTE] stage=%d, m=%d, BK=%d, chunk_size=%d, num_chunks=%d\n", stage, m, BK, chunk_size, num_chunks);
    
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        auto start_t = std::chrono::high_resolution_clock::now();
        uint64_t start_vct = arm64_get_cntvct();
        int current_m = std::min(chunk_size, m - chunk * chunk_size);
        
        uint32_t y_off = 0x600000u;
        volatile int32_t* dest = (volatile int32_t*)((uint8_t*)e->vbase + y_off);
        // for (int i = 0; i < current_m; i++) dest[i] = 0; // MUST remain commented to avoid dirtying CPU cache!

        // [OPT] Removed redundant chunk-level soft_reset — the per-pass reset at L341 handles this.

        uint32_t hw_rmsnorm = (stage >> 2) & 1;
        uint32_t h_trans = (stage >> 3) & 1;
        bool loopback_en = false; // Never loopback in full engine mode
        
        uint32_t stage_base = stage & 0xF;
        bool is_stage4     = (stage_base == 4);
        bool save_gate_en  = (stage_base == 1);
        bool swiglu_en     = (stage_base == 2);
        bool silu_en       = (stage_base == 2);
        uint32_t mode_val = (uint32_t)mod;
        bool comp_en_global = (mode_val & (1 << 24)) != 0;
        bool save_res_en   = (stage >> 5) & 1;
        
        if (k == 192) {
            printf("[DEBUG] mode_val=0x%08X, comp_en_global=%d\n", mode_val, comp_en_global);
        }
        uint32_t kb_loop = (k + 63) / 64;
        kb_loop = (kb_loop + 3) & ~3;
        uint32_t pe_k_beats = kb_loop;

        uint32_t real_hw_mode = ((uint32_t)mod & 0xFF);
        if (real_hw_mode == V37_MODE_I2_S_PREPACKED) {
            real_hw_mode = V37_MODE_I2_S;
        }

        uint32_t cfg_val = (pe_k_beats << 16) | (save_gate_en << 15) | (swiglu_en << 14) | (save_res_en << 13) | (loopback_en << 12) | (h_trans << 9) | (hw_rmsnorm << 8) | real_hw_mode;
        
        uint32_t pulses = (current_m + 3) / 4;
        uint32_t pe_out_beats = pulses;
        uint32_t comp_bit = comp_en_global ? (1 << 24) : 0;
        
        // Extract rope flags from stage (bits 25, 26, 27)
        uint32_t rope_en = (stage >> 25) & 1;
        uint32_t load_rope_en = (stage >> 26) & 1;
        uint32_t half_dim_beats = (stage >> 27) & 0xFF; // Usually 16 for rot_dim=128, LANES=4
        if (rope_en) {
            cfg_val |= (1 << 10); // cfg_word[10] = rope_en
        }
        if (load_rope_en) {
            cfg_val |= (1 << 11); // cfg_word[11] = load_rope_en
        }
        // half_dim_beats mapped to cfg_word[31:24]
        cfg_val |= (half_dim_beats << 24);
        
        // Extract gamma_shift (swiglu_shift) from stage [21:16]
        uint32_t gamma_shift = (stage >> 16) & 0x3F;
        // swiglu_shift is cfg_word[52:48]. cfg_word[63:32] is mapped to 0x0C.
        // Therefore, we shift by 16 to place it at bits [20:16] of 0x0C.
        uint32_t shift_bits = (gamma_shift & 0x1F) << 16;
        
        uint32_t s2mm_out_beats = pulses;
        
        if (chunk == 0 && getenv("V37_DEBUG")) {
            printf("[V37-DRIVER-DEBUG] Starting chunk 0! k=%d, m=%d, current_m=%d, pulses=%d\n", k, m, current_m, pulses); fflush(stdout);
        }
        
        if (k == 256) {
            // printf("[DEBUG-VCORE] cfg_val=0x%08X, 0x0C=0x%08X, pe_out_beats=%u\n", cfg_val, (pe_out_beats & 0xFFFF) | (1 << 24), pe_out_beats);
        }
        // Hardware w_gamma multiplier uses Q24 fixed point format.
        // We pass 1.0 (0x01000000) so the hardware outputs the raw MAC sum.
        // The driver will apply the floating-point scaling in software.
        uint32_t scale_bits = 0x01000000;
        
        // [OPT-P2] Zero-pad activation buffer before memcpy, only if not loopback
        if (!loopback_en) {
            if (!reuse_act) {
                uint32_t act_dma_sz = pe_k_beats * 64;
                uint8_t* act_ptr = (uint8_t*)e->cma_vbase + 0x400000u;
                memcpy(act_ptr, x, k);
                if (act_dma_sz > (uint32_t)k) {
                    memset(act_ptr + k, 0, act_dma_sz - k);
                }
                sync_cma_for_device(e, 0x400000u, act_dma_sz);
            }
            
            // Debug: dump first few x_q values written to CMA
            static bool act_printed = false;
            if (!act_printed && s_id == 0 && (stage & 0xFF) == 0) {
                act_printed = true;
                const int8_t* x_check = (const int8_t*)x;
                // printf("[V37-DRIVER-DEBUG] x_q[0..9] = %d %d %d %d %d %d %d %d %d %d\n",
                //        x_check[0], x_check[1], x_check[2], x_check[3], x_check[4],
                //        x_check[5], x_check[6], x_check[7], x_check[8], x_check[9]);
                // Also verify CMA content matches
                volatile int8_t* cma_x = (volatile int8_t*)((uint8_t*)e->cma_vbase + 0x400000u + 512);
                // printf("[V37-DRIVER-DEBUG] cma[0..9] = %d %d %d %d %d %d %d %d %d %d\n",
                //        cma_x[0], cma_x[1], cma_x[2], cma_x[3], cma_x[4],
                //        cma_x[5], cma_x[6], cma_x[7], cma_x[8], cma_x[9]);
            }
        }
        
        uint32_t s2mm_bytes = s2mm_out_beats * 16;
        
        bool is_stage5 = (stage_base == 5);
        bool ffn_stream_mode = is_stage5; // FORCE streaming mode for FFN fusion
        int passes = is_stage4 ? 2 : (is_stage5 ? 3 : 1);
        
        // =============================================================
        if (getenv("V37_DEBUG")) {
            printf("[V37-DRIVER-TRACE] Before checking ffn_stream_mode (is_stage5=%d, stage=%d)\n", is_stage5, stage); fflush(stdout);
        }
        if (ffn_stream_mode && e->vcore != MAP_FAILED) {
            if (getenv("V37_DEBUG")) {
                printf("[V37-DRIVER-DEBUG] Entering ffn_stream_mode! ext_stage=%d\n", stage);
                fflush(stdout);
            }
            
            // Pre-compute per-pass configurations
            struct { uint32_t cfg_val; uint32_t pe_out_beats_reg; uint32_t total_scale_q24; int slot; uint32_t pass_k; uint32_t pass_m; uint32_t out_beats; } pinfo[3];
            for (int p = 0; p < 3; p++) {
                bool p_save_gate = (p == 0);
                bool p_swiglu    = (p == 1);
                bool p_loopback  = (p == 1);
                bool p_quantize  = (p == 1);
                bool p_dma_out   = (p == 2);
                bool p_res_fusion = ((stage >> 6) & 1) ? (p == 2) : false;
                
                pinfo[p].pass_k = (p == 2) ? current_m : k;
                pinfo[p].pass_m = (p == 2) ? k : current_m;
                pinfo[p].out_beats = (pinfo[p].pass_m + 3) / 4;
                
                uint32_t p_kb = (pinfo[p].pass_k + 63) / 64;
                p_kb = (p_kb + 3) & ~3;
                
                bool p_save_res     = is_stage5 ? (p == 0) : save_res_en;
                pinfo[p].cfg_val = (p_kb << 16) | (p_save_gate << 15) | (p_swiglu << 14) |
                    (p_save_res << 13) | (p_loopback << 12) | (p_quantize << 6) |
                    (h_trans << 9) | (hw_rmsnorm << 8) | (p_res_fusion << 7) | real_hw_mode;
                
                uint32_t p_pulses = (pinfo[p].pass_m + 3) / 4;
                uint32_t p_dma_out_bit = p_dma_out ? (1 << 25) : 0;
                pinfo[p].pe_out_beats_reg = (p_pulses & 0xFFFF) | comp_bit | shift_bits | p_dma_out_bit;
                float act_scale = w_scales ? w_scales[4] : 1.0f;
                float p_scale = 1.0f;
                if (is_stage5) {
                    if (p == 0) p_scale = std::abs(w_scales[0]) / act_scale;
                    else if (p == 1) p_scale = std::abs(w_scales[1]) / act_scale;
                    else p_scale = 1.0f; // Output unscaled INT32 MACs for Pass 2!
                } else {
                    float s_val = w_scales ? w_scales[3] : 1.0f;
                    p_scale = std::abs(w_scales ? w_scales[p] : 1.0f) / s_val;
                }

                pinfo[p].total_scale_q24 = (uint32_t)(p_scale * 16777216.0f);
                
                if (p == 0) pinfo[p].slot = s_id & 0xFF;
                else if (p == 1) pinfo[p].slot = (s_id >> 8) & 0xFF;
                else pinfo[p].slot = (s_id >> 16) & 0xFF;
            }
            
            // Pre-compute Pass 2 undo_factor total_scale and write it to MMIO
            // The hardware FSM will read this from the total_scale input port
            // (We can't know amax_k/R_fixed yet, so we write w_scale_down_q24 * 1.0
            //  and let the driver POLL amax_k/R_fixed after completion to apply final undo.)
            // ALTERNATIVE: For now, use a two-step approach:
            //   1. Hardware FSM uses MMIO total_scale for Pass 2 (pre-loaded)
            //   2. After S2MM complete, no undo recalculation needed if we set it right.
            // Since undo_factor depends on runtime amax, we can't pre-compute it.
            // SOLUTION: The driver pre-writes total_scale = 1.0Q24 to MMIO.
            // After Pass 1 DynQuant completes, FSM reads total_scale from MMIO.
            // We set MMIO total_scale = undo * w_scale_down AFTER activations are loaded
            // BUT BEFORE the DMA finishes Pass 1. This requires CPU involvement.
            //
            // PRAGMATIC: Keep the hardware undo_factor path simple.
            // Set cfg_word_core bit 59 to enable streaming FSM.
            // The Pass 2 header's total_scale is ignored by FSM (it reads MMIO instead).
            // Driver CANNOT pre-compute undo. So:
            //   Option A: Skip undo in hardware, apply in software post-S2MM → loses streaming benefit
            //   Option B: Hardware divider (complex)
            //   Option C: Driver pre-writes an estimate, then corrects in postprocessing
            //
            // Going with Option A for correctness: hardware outputs raw scaled INT32,
            // driver applies undo_factor in post-processing after reading amax_k/R_fixed.
            // This still saves 2 of 3 CPU round-trips (Pass 0→1 and Pass 1→2 config).
            
            // Build concatenated weight buffer for all 4 channels
            // Layout per channel: [hdr0][payload0][hdr1][payload1][hdr2][payload2]
            uint32_t total_w_len[4] = {0, 0, 0, 0};
            
            // First, calculate total per-channel DMA length
            for (int p = 0; p < 3; p++) {
                uint32_t p_kb = (pinfo[p].pass_k + 63) / 64;
                p_kb = (p_kb + 3) & ~3;
                uint32_t true_m_p = pinfo[p].pass_m;
                uint32_t m_blocks = (true_m_p + 3) / 4;
                uint32_t ch_stride;
                uint32_t channel_sz;
                uint32_t matrix_beats;

                if (real_hw_mode == V37_MODE_TL3_COMPACT) {
                    ch_stride = p_kb * 13;
                    uint32_t total_payload = m_blocks * ch_stride;
                    matrix_beats = (total_payload + 15) / 16;
                    channel_sz = matrix_beats * 16 + 16;
                    channel_sz = (channel_sz + 63) & ~63;
                } else {
                    uint32_t total_beats_p = p_kb;
                    if (real_hw_mode != V37_MODE_I2_S && real_hw_mode != V37_MODE_TL3)
                        total_beats_p = p_kb + (p_kb + 3) / 4;
                    ch_stride = total_beats_p * 16;
                    channel_sz = ch_stride * m_blocks + 16; // +16 for header
                    channel_sz = (channel_sz + 63) & ~63;
                    matrix_beats = m_blocks * total_beats_p;
                }
                total_w_len[0] += matrix_beats * 16 + 16; // per-pass per-channel
            }
            // All 4 channels have same length structure
            for (int i = 1; i < 4; i++) total_w_len[i] = total_w_len[0];
            
            // Write in-band headers into weight slot data
            for (int p = 0; p < 3; p++) {
                uint32_t p_kb = (pinfo[p].pass_k + 63) / 64;
                p_kb = (p_kb + 3) & ~3;
                uint32_t m_blocks = (pinfo[p].pass_m + 3) / 4;
                uint32_t ch_stride;
                uint32_t channel_sz;
                uint32_t matrix_beats;

                if (real_hw_mode == V37_MODE_TL3_COMPACT) {
                    ch_stride = p_kb * 13;
                    uint32_t total_payload = m_blocks * ch_stride;
                    matrix_beats = (total_payload + 15) / 16;
                    channel_sz = matrix_beats * 16 + 16;
                    channel_sz = (channel_sz + 63) & ~63;
                } else {
                    uint32_t total_beats_p = p_kb;
                    if (real_hw_mode != V37_MODE_I2_S && real_hw_mode != V37_MODE_TL3)
                        total_beats_p = p_kb + (p_kb + 3) / 4;
                    ch_stride = total_beats_p * 16;
                    channel_sz = ch_stride * m_blocks + 16;
                    channel_sz = (channel_sz + 63) & ~63;
                    matrix_beats = m_blocks * total_beats_p;
                }
                
                uint8_t* slot_base = (uint8_t*)e->cma_vbase + e->slot_offsets[pinfo[p].slot];
                
                for (int i = 0; i < 4; i++) {
                    uint32_t ch_base = i * channel_sz;
                    volatile uint32_t* hdr_ptr = (volatile uint32_t*)(slot_base + ch_base);
                    
                    if (i == 0) {
                        // Write full config into channel 0 header
                        hdr_ptr[0] = pinfo[p].cfg_val;             // cfg_word[31:0]
                        hdr_ptr[1] = pinfo[p].pe_out_beats_reg;    // cfg_word[63:32]
                        hdr_ptr[2] = pinfo[p].total_scale_q24;     // total_scale Q24
                        hdr_ptr[3] = matrix_beats;                  // current_matrix_beats
                    } else if (i == 1) {
                        hdr_ptr[0] = p_kb; hdr_ptr[1] = 0xDEADBEEF;
                        hdr_ptr[2] = 0; hdr_ptr[3] = 0;
                    } else {
                        hdr_ptr[0] = 0; hdr_ptr[1] = 0; hdr_ptr[2] = 0; hdr_ptr[3] = 0;
                    }
                }
                // Sync this pass's weight data
                for (int i = 0; i < 4; i++) {
                    uint32_t ch_base = i * channel_sz;
                    sync_cma_for_device(e, e->slot_offsets[pinfo[p].slot] + ch_base, channel_sz);
                }
            }
            
            // Configure VCORE: set ffn_stream_en bit (bit 59 = 0x0C bit 27)
            uint32_t stream_cfg_val = pinfo[0].cfg_val;
            uint32_t stream_pe_out = pinfo[0].pe_out_beats_reg;
            wreg(e->vcore, 0x08, stream_cfg_val);
            wreg(e->vcore, 0x0C, stream_pe_out | (1 << 27) | (1 << 29)); // bit 27 = ffn_stream_en, bit 29 = use_sq_relu (cfg_word[61])
            wreg(e->vcore, 0x10, 0x01000000); // gamma = 1.0
            // Pre-load Pass 2 total_scale into MMIO for hardware FSM to read
            // Since we can't pre-compute undo, set to 1.0 Q24; driver will apply undo in post-processing
            wreg(e->vcore, 0x14, pinfo[2].total_scale_q24);
            
            // Soft reset to latch config
            wreg(e->vcore, 0x00, 2);
            for (volatile int d = 0; d < 10; d++) {}
            wreg(e->vcore, 0x00, 0);
            
            // Reset all DMAs
            int rc = 0;
            wreg(e->dma_x, 0x00, 4);
            while(rreg(e->dma_x, 0x00) & 4) { if(++rc > 100000) break; }
            wreg(e->dma_x, 0x30, 4); rc=0;
            while(rreg(e->dma_x, 0x30) & 4) { if(++rc > 100000) break; }
            wreg(e->dma_x, 0x30, 4); rc=0;
            while(rreg(e->dma_x, 0x30) & 4) { if(++rc > 100000) break; }
            for (int i = 0; i < 4; i++) {
                wreg(e->dma_w[i], 0x00, 4); rc=0;
                while(rreg(e->dma_w[i], 0x00) & 4) { if(++rc > 100000) break; }
            }
            
            // Start S2MM for Pass 2 output (only pass with dma_out_en)
            uint32_t final_m = k; // Pass 2: M_out = original K (hidden_dim)
            uint32_t s2mm_final_beats = (final_m + 3) / 4;
            uint32_t s2mm_final_bytes = s2mm_final_beats * 16;
            wreg(e->dma_x, 0x34, 0x7000);
            wreg(e->dma_x, 0x48, e->phys + y_off);
            wreg(e->dma_x, 0x30, 0x01);
            wreg(e->dma_x, 0x58, s2mm_final_bytes);
            
            // Start activation DMA (only needed once for Pass 0)
            wreg(e->dma_x, 0x04, 0x7000);
            wreg(e->dma_x, 0x18, e->phys + 0x400000u);
            wreg(e->dma_x, 0x00, 0x01);
            wreg(e->dma_x, 0x28, pe_k_beats * 64);
            
            // --- Per-pass weight DMA loop ---
            // Hardware FSM handles config switching autonomously via in-band headers.
            // Driver only needs to start/restart weight DMAs for each pass (no MMIO config writes).
            // This saves ~30μs of MMIO overhead compared to the legacy per-pass approach.
            for (int p = 0; p < 3; p++) {
                uint32_t p_kb = (pinfo[p].pass_k + 63) / 64;
                p_kb = (p_kb + 3) & ~3;
                uint32_t m_blocks_p = (pinfo[p].pass_m + 3) / 4;
                uint32_t channel_sz_p;
                uint32_t w_len_p;

                if (real_hw_mode == V37_MODE_TL3_COMPACT) {
                    uint32_t ch_stride_p = p_kb * 13;
                    uint32_t total_payload = m_blocks_p * ch_stride_p;
                    uint32_t matrix_beats_p = (total_payload + 15) / 16;
                    channel_sz_p = matrix_beats_p * 16 + 16;
                    channel_sz_p = (channel_sz_p + 63) & ~63;
                    w_len_p = matrix_beats_p * 16 + 16;
                } else {
                    uint32_t total_beats_p = p_kb;
                    if (real_hw_mode != V37_MODE_I2_S && real_hw_mode != V37_MODE_TL3)
                        total_beats_p = p_kb + (p_kb + 3) / 4;
                    channel_sz_p = total_beats_p * 16 * m_blocks_p + 16;
                    channel_sz_p = (channel_sz_p + 63) & ~63;
                    w_len_p = m_blocks_p * total_beats_p * 16 + 16;
                }

                for (int i = 0; i < 4; i++) {
                    uint32_t ch_base = i * channel_sz_p;
                    uint32_t w_phys = e->phys + e->slot_offsets[pinfo[p].slot] + ch_base;
                    wreg(e->dma_w[i], 0x18, w_phys);
                    wreg(e->dma_w[i], 0x00, 0x01);
                    wreg(e->dma_w[i], 0x28, w_len_p);
                }

                // Wait for all 4 weight DMA channels to complete
                for (int i = 0; i < 4; i++) {
                    int stuck = 0;
                    uint32_t w_stat;
                    while (!((w_stat = rreg(e->dma_w[i], 0x04)) & 0x1002)) {
                        cpu_relax();
                        if (++stuck > 2000000) {
                            printf("[V37-FFN-STREAM] weight DMA[%d] hang at pass=%d! stat=%08X\n", i, p, w_stat);
                            return -3;
                        }
                    }
                    wreg(e->dma_w[i], 0x04, 0x7000); // W1C clear status
                }
                // Reset DMAs for next pass (skip on last pass)
                if (p < 2) {
                    for (int i = 0; i < 4; i++) {
                        wreg(e->dma_w[i], 0x00, 4);
                        int r = 0;
                        while (rreg(e->dma_w[i], 0x00) & 4) { if (++r > 100000) break; }
                    }
                }
            }
            
            // Poll S2MM completion (Pass 2 output)
            {
                int stuck_cnt = 0;
                uint32_t stat;
                while (!((stat = rreg(e->dma_x, 0x34)) & 0x1002)) {
                    cpu_relax();
                    if (++stuck_cnt > 5000000) {
                        printf("[V37-FFN-STREAM] S2MM TIMEOUT! stat=%08X\n", stat);
                        return -3;
                    }
                }
                if (getenv("V37_DEBUG")) {
                    printf("[V37-FFN-STREAM] S2MM Complete: stat=%08X, dbg_cnt=%u\n", stat, rreg(e->vcore, 0x14));
                }
            }
            
            // Apply undo_factor in post-processing
            // Read amax_k and R_fixed from hardware register
            uint32_t amax_reg = rreg(e->vcore, 0x18);
            uint32_t amax_k = (amax_reg >> 16) & 0x1F;
            uint32_t r_fixed = amax_reg & 0xFFFF;
            float undo_factor = (r_fixed > 0) ? ((float)(1ULL << (amax_k + 8)) / (float)r_fixed) : 1.0f;
            float p2_w_scale = std::abs(w_scales ? w_scales[2] : 1.0f);
            float final_scale = p2_w_scale * undo_factor;

            if (getenv("V37_DEBUG")) {
                printf("[V37-FFN-STREAM] undo: amax_k=%u, r_fixed=%u, undo_factor=%.6f, final_scale=%.6f\n", amax_k, r_fixed, undo_factor, final_scale);
            }
            
            // Read output and apply undo
            sync_cma_for_cpu(e, y_off, s2mm_final_bytes);
            int32_t* hw_out = (int32_t*)((uint8_t*)e->cma_vbase + y_off);
            if (residual) {
                float* res_ptr = (float*)residual;
                for (uint32_t i = 0; i < (uint32_t)k; i++) {
                    res_ptr[i] += (float)hw_out[i] * final_scale;
                }
            } else if (y) {
                float* y_out = (float*)y;
                for (uint32_t i = 0; i < (uint32_t)k; i++) {
                    y_out[i] = (float)hw_out[i] * final_scale;
                }
            }
            
            if (getenv("V37_DEBUG")) {
                printf("[V37-DRIVER-DEBUG] Exiting ffn_stream_mode!\n");
                fflush(stdout);
            }
            
            // Skip the legacy per-pass loop
            goto ffn_stream_done;
        }
        
        for (int pass = 0; pass < passes; pass++) {
            bool pass_save_gate = is_stage4 ? (pass == 0) : (is_stage5 ? (pass == 0) : save_gate_en);
            bool pass_swiglu    = is_stage4 ? (pass == 1) : (is_stage5 ? (pass == 1) : swiglu_en);
            bool pass_loopback  = is_stage5 ? (pass == 1) : loopback_en;
            bool pass_quantize  = is_stage5 ? (pass == 1) : false;
            bool dma_out_en = is_stage5 ? (pass == 2) : (!pass_save_gate && !loopback_en && !save_res_en);
            
            int pass_slot;
            if (is_stage4) pass_slot = (pass == 0) ? (s_id & 0xFFFF) : ((s_id >> 16) & 0xFFFF);
            else if (is_stage5) {
                if (pass == 0) pass_slot = s_id & 0xFF;
                else if (pass == 1) pass_slot = (s_id >> 8) & 0xFF;
                else pass_slot = (s_id >> 16) & 0xFF;
            } else pass_slot = s_id;

            uint32_t pass_k = is_stage5 ? ((pass == 2) ? current_m : k) : k;
            uint32_t pass_m = is_stage5 ? ((pass == 2) ? k : current_m) : current_m;

            uint32_t kb_loop = (pass_k + 63) / 64;
            kb_loop = (kb_loop + 3) & ~3;
            uint32_t pe_k_beats = kb_loop;

            bool pass_res_fusion = false;
            if (((stage >> 6) & 1)) {
                pass_res_fusion = is_stage5 ? (pass == 2) : true;
            }

            uint32_t cfg_val = (pe_k_beats << 16) | (pass_save_gate << 15) | (pass_swiglu << 14) | (save_res_en << 13) | (pass_loopback << 12) | (pass_quantize << 6) | (h_trans << 9) | (hw_rmsnorm << 8) | (pass_res_fusion << 7) | real_hw_mode;
            struct v37_compute_cmd cmd = {0};
            if (is_stage5) {
                float act_scale = w_scales ? w_scales[4] : 1.0f;
                if (pass == 0) {
                    cmd.total_scale = std::abs(w_scales ? w_scales[0] : 1.0f) / act_scale;
                } else if (pass == 1) {
                    cmd.total_scale = std::abs(w_scales ? w_scales[1] : 1.0f) / act_scale;
                } else {
                    uint32_t amax_reg = rreg(e->vcore, 0x18);
                    uint32_t amax_k = (amax_reg >> 16) & 0x1F;
                    uint32_t r_fixed = amax_reg & 0xFFFF;
                    float undo_factor = (r_fixed > 0) ? ((float)(1ULL << (amax_k + 8)) / (float)r_fixed) : 1.0f;
                    cmd.total_scale = std::abs(w_scales ? w_scales[2] : 1.0f) * undo_factor;
                    printf("[V37-DRIVER-DEBUG] Pass 2: amax_k=%u, r_fixed=%u, undo_factor=%f, cmd.total_scale=%f\n", amax_k, r_fixed, undo_factor, cmd.total_scale);
                }
            } else {
                cmd.total_scale = 1.0f; // raw INT32 MAC for normal layers
            }
            cmd.cfg_val = cfg_val;
            uint32_t pulses = (pass_m + 3) / 4;
            uint32_t pe_out_beats = pulses;
            uint32_t dma_out_bit = dma_out_en ? (1 << 25) : 0;
            // rmsnorm_loopback_en: bit 26 of pe_out_beats → dynamic_cfg_word[58]
            // We DO NOT want RMSNorm on the SwiGLU output! It must go to DynQuant!
            uint32_t rmsnorm_loopback_bit = 0;
            uint32_t pass_comp_bit = comp_bit;
            cmd.pe_out_beats = (pe_out_beats & 0xFFFF) | pass_comp_bit | shift_bits | dma_out_bit | rmsnorm_loopback_bit;
            cmd.scale_bits = scale_bits;
            cmd.do_soft_reset = 1;
            
            bool s2mm_active = dma_out_en;
            cmd.s2mm_active = s2mm_active;
            if (s2mm_active) {
                cmd.y_phys = e->phys + y_off;
                cmd.y_len = s2mm_bytes;
            }

            bool mm2s_active = (pass == 0) && (stage != 9) && !reuse_act && (getenv("V37_SKIP_ACT_DMA") == nullptr);
            cmd.mm2s_active = mm2s_active;
            if (mm2s_active) {
                cmd.x_phys = e->phys + 0x400000u;
                cmd.x_len = pe_k_beats * 64;
            }
            
            uint32_t kb = (pass_k + 63) / 64;
            kb = (kb + 3) & ~3;
            pe_k_beats = kb;
            
            uint32_t true_m = is_stage5 ? pass_m : ((total_m > 0) ? total_m : m);
            uint32_t m_blocks = (true_m + 3) / 4;
            uint32_t current_m_blocks = (pass_m + 3) / 4;
            uint32_t ch_stride;
            uint32_t channel_sz;
            uint32_t current_matrix_beats;

            if (real_hw_mode == V37_MODE_TL3_COMPACT) {
                ch_stride = pe_k_beats * 13;
                uint32_t total_payload = m_blocks * ch_stride;
                uint32_t total_beats = (total_payload + 15) / 16;
                channel_sz = total_beats * 16 + 16;
                channel_sz = (channel_sz + 63) & ~63;
                uint32_t current_payload = current_m_blocks * ch_stride;
                current_matrix_beats = (current_payload + 15) / 16;
            } else {
                uint32_t total_beats = pe_k_beats;
                // hardware v37_weight_unpacker expects 1 header per 4 payload beats for most modes,
                // but V9.5 RTL fixed I2_S and TL3 modes to not require dummy headers.
                if (real_hw_mode != V37_MODE_I2_S && real_hw_mode != V37_MODE_TL3) {
                    total_beats = pe_k_beats + (pe_k_beats + 3) / 4;
                }
                ch_stride = total_beats * 16;
                channel_sz = m_blocks * ch_stride + 16;
                channel_sz = (channel_sz + 63) & ~63;
                current_matrix_beats = current_m_blocks * total_beats;
            }
            
            uint8_t* slot_base = (uint8_t*)e->cma_vbase + e->slot_offsets[pass_slot];
            
            static std::unordered_map<uint64_t, bool> slot_hdr_synced;
            uint64_t slot_key = ((uint64_t)pass_slot << 32) | (chunk * chunk_size + m_offset);
            bool need_hdr_sync = !slot_hdr_synced[slot_key];

            if (need_hdr_sync) {
                for(int i=0; i<4; i++) {
                    uint32_t ch_base = i * channel_sz;
                    uint32_t chunk_offset_addr = is_stage5 ? 0 : (((chunk * chunk_size + m_offset) / 4) * ch_stride);
                    
                    volatile uint32_t* hdr_ptr = (volatile uint32_t*)(slot_base + ch_base + chunk_offset_addr);
                    
                    if (i == 0) {
                        hdr_ptr[0] = cmd.cfg_val;
                        hdr_ptr[1] = cmd.pe_out_beats;
                        if (is_stage5) {
                            uint32_t gamma_q24 = 0x01000000;
                            if (pass == 0 || pass == 1) {
                                gamma_q24 = (uint32_t)(cmd.total_scale * 64.0f * 16777216.0f);
                            }
                            hdr_ptr[2] = gamma_q24;
                        } else {
                            hdr_ptr[2] = 0x01000000; // 1.0 in Q8.24 format
                        }
                        hdr_ptr[3] = current_matrix_beats;
                    } else if (i == 1) {
                        hdr_ptr[0] = pe_k_beats;
                        hdr_ptr[1] = 0xDEADBEEF;
                        hdr_ptr[2] = 0; hdr_ptr[3] = 0;
                    } else {
                        hdr_ptr[0] = 0; hdr_ptr[1] = 0; hdr_ptr[2] = 0; hdr_ptr[3] = 0;
                    }
                }
                
                for(int i=0; i<4; i++) {
                    uint32_t ch_base = i * channel_sz;
                    uint32_t chunk_offset_addr = is_stage5 ? 0 : (((chunk * chunk_size + m_offset) / 4) * ch_stride);
                    sync_cma_for_device(e, e->slot_offsets[pass_slot] + ch_base + chunk_offset_addr, 64);
                }
                slot_hdr_synced[slot_key] = true;
            }

            for(int i=0; i<4; i++) {
                uint32_t ch_base = i * channel_sz;
                uint32_t chunk_offset_addr = is_stage5 ? 0 : (((chunk * chunk_size + m_offset) / 4) * ch_stride);
                cmd.w_phys[i] = e->phys + e->slot_offsets[pass_slot] + ch_base + chunk_offset_addr;
                cmd.w_len[i] = current_matrix_beats * 16 + 16;
            }


            // --- SG-DMA APPEND LOGIC ---
            if (e->sg_dispatcher != MAP_FAILED) {
                if (e->desc_count < 65535) {
                    v37_sg_desc_t* desc = &e->desc_pool[e->desc_count];
                    desc->cfg_val = cmd.cfg_val;
                    desc->act_in_addr = cmd.mm2s_active ? cmd.x_phys : 0;
                    desc->act_out_addr = cmd.s2mm_active ? cmd.y_phys : 0;
                    desc->x_length = cmd.mm2s_active ? cmd.x_len : 0;
                    desc->y_length = cmd.s2mm_active ? cmd.y_len : 0;
                    desc->weight_addr_0 = cmd.w_phys[0];
                    desc->weight_length = cmd.w_len[0];
                    uint32_t scale_bits_val = cmd.scale_bits;
                    memcpy(&desc->total_scale, &scale_bits_val, sizeof(float));
                    
                    if (e->desc_count > 0) {
                        uint32_t curr_phys = e->phys + 0x200000u + (e->desc_count * sizeof(v37_sg_desc_t));
                        e->desc_pool[e->desc_count - 1].next_desc_ptr = curr_phys;
                    }
                    desc->next_desc_ptr = 0;
                    e->desc_count++;
                    cmd.do_soft_reset = 0; // Handled by first desc? Hardware handles it.
                }
            } else if (e->fd_v37 >= 0) {
                if (ioctl(e->fd_v37, V37_IOCTL_COMPUTE, &cmd) < 0) {
                    perror("V37 ioctl failed");
                    return -3;
                }
            } else {
                // ============================================================
                // FALLBACK: User-space MMIO
                // ============================================================
                // STEP 1: Configure VCORE registers FIRST (BEFORE soft_reset!)
                // Rationale: RTL global_buffer loads cfg_word_core into dynamic_cfg_word on soft_reset.
                // If we reset first and write cfg_word after, the initial dynamic_cfg_word will be stale/zero.
                if (chunk == 0) {
                    // printf("WRITING CFG_VAL: 0x%08X (swiglu_en=%d)\n", cmd.cfg_val, (cmd.cfg_val>>14)&1); 
                    wreg(e->vcore, 0x08, cmd.cfg_val);
                    wreg(e->vcore, 0x0C, cmd.pe_out_beats);
                }
                uint32_t gamma_q24 = 0x01000000;
                if (is_stage5 && (pass == 0 || pass == 1)) {
                    gamma_q24 = (uint32_t)(cmd.total_scale * 64.0f * 16777216.0f);
                }
                wreg(e->vcore, 0x10, gamma_q24);
                
                uint32_t q24_scale = (uint32_t)(cmd.total_scale * 16777216.0f);
                wreg(e->vcore, 0x14, q24_scale);
                if (chunk == 0 && (stage & 0xFF) == 5) {
                    // printf("[V37-DRIVER-DEBUG] Pass %d: total_scale_f=%.6f, q24_scale=%u (0x%08X), gamma_q24=%u\n", pass, cmd.total_scale, q24_scale, q24_scale, gamma_q24);
                }
                
                // STEP 2: Pulse soft_reset AFTER registers are set
                wreg(e->vcore, 0x00, 2); // assert soft_reset
                for (volatile int delay_idx = 0; delay_idx < 10; delay_idx++) {}
                wreg(e->vcore, 0x00, 0); // clear soft_reset

                // STEP 2: Reset ALL DMAs ONLY on error recovery or if explicitly forced
                static bool force_dma_reset = (getenv("V37_FORCE_DMA_RESET") && atoi(getenv("V37_FORCE_DMA_RESET")) == 1);
                if (force_dma_reset) {
                    int reset_cnt = 0;
                    wreg(e->dma_x, 0x00, 4); // MM2S
                    while(rreg(e->dma_x, 0x00) & 4) { if(++reset_cnt > 100000) break; }
                    
                    if (cmd.s2mm_active) {
                        wreg(e->dma_x, 0x30, 4); // S2MM reset
                        reset_cnt = 0; while(rreg(e->dma_x, 0x30) & 4) { if(++reset_cnt > 100000) break; }
                        wreg(e->dma_x, 0x30, 4); // S2MM Double reset (flush FIFO)
                        reset_cnt = 0; while(rreg(e->dma_x, 0x30) & 4) { if(++reset_cnt > 100000) break; }
                    }
                    
                    for (int i = 0; i < 4; i++) {
                        wreg(e->dma_w[i], 0x00, 4);
                        reset_cnt = 0; while(rreg(e->dma_w[i], 0x00) & 4) { if(++reset_cnt > 100000) break; }
                    }
                }
        
                if (cmd.s2mm_active) {
                    // Clear S2MM status first (W1C to clear any stale IRQ bits)
                    wreg(e->dma_x, 0x34, 0x7000);
                    wreg(e->dma_x, 0x48, cmd.y_phys); 
                    wreg(e->dma_x, 0x30, 0x01); 
                    wreg(e->dma_x, 0x58, cmd.y_len);
                }
        
                if (cmd.mm2s_active) {
                    wreg(e->dma_x, 0x04, 0x7000); // Clear MM2S status
                    wreg(e->dma_x, 0x18, cmd.x_phys);
                    wreg(e->dma_x, 0x00, 0x01);
                    wreg(e->dma_x, 0x28, cmd.x_len);
                    
                    int stuck_cnt = 0;
                    uint32_t mm2s_stat;
                    while (!((mm2s_stat = rreg(e->dma_x, 0x04)) & 0x1002)) {
                        cpu_relax();
                        if (++stuck_cnt > 10000000) { printf("[DEBUG] Hung at act_sram fill! chunk=%d, stat=%08X\n", chunk, mm2s_stat); return -3; }
                    }
                }

                for(int i=0; i<4; i++) {
                    wreg(e->dma_w[i], 0x18, cmd.w_phys[i]);
                    wreg(e->dma_w[i], 0x00, 0x01); 
                    wreg(e->dma_w[i], 0x28, cmd.w_len[i]);
                }

                auto t_before_poll = std::chrono::high_resolution_clock::now();
                t_setup += std::chrono::duration_cast<std::chrono::microseconds>(t_before_poll - start_t).count();

                if (cmd.s2mm_active) {
                    hybrid_wait_s2mm(e, start_t, current_m, k, start_vct);
                    int stuck_cnt = 0;
                    uint32_t final_s2mm_stat = 0;
                    while (!((final_s2mm_stat = rreg(e->dma_x, 0x34)) & 0x1002)) {
                        cpu_relax();
                        if (++stuck_cnt > 10000000) { 
                            // Read debug registers on hang
                            uint32_t dbg_cnt = rreg(e->vcore, 0x14);
                            uint32_t w0_stat = rreg(e->dma_w[0], 0x04);
                            uint32_t mm2s_stat = rreg(e->dma_x, 0x04);
                            printf("[DEBUG] Hung at S2MM Poll! chunk=%d, stage=%d, stat=%08X, vcore_dbg=%u, w0_stat=%08X, mm2s=%08X\n", 
                                   chunk, stage, final_s2mm_stat, dbg_cnt, w0_stat, mm2s_stat); 
                            return -3; 
                        }
                    }
                    wreg(e->dma_x, 0x34, 0x7000); 
                    
                    for (int i = 0; i < 4; i++) {
                        wreg(e->dma_w[i], 0x04, 0x7000);
                    }
                } else {
                    for (int i = 0; i < 4; i++) {
                        int stuck_cnt = 0;
                        uint32_t w_stat;
                        while (!((w_stat = rreg(e->dma_w[i], 0x04)) & 0x1002)) {
                            cpu_relax();
                            if (++stuck_cnt > 1000000) { printf("[DEBUG] Hung at dma_w[%d] MM2S Poll! chunk=%d, stage=%d, stat=%08X\n", i, chunk, stage, w_stat); return -3; }
                        }
                        wreg(e->dma_w[i], 0x04, 0x7000);
                    }
                }
                auto t_after_poll = std::chrono::high_resolution_clock::now();
                t_poll += std::chrono::duration_cast<std::chrono::microseconds>(t_after_poll - t_before_poll).count();
            } // end fallback else block
        } // end pass loop
        ffn_chunk_done:;
        
        if (chunk == 0) {
           // printf("[DEBUG-DMA] MM2S Stat: %08X, S2MM Stat: %08X, VCORE Stat: %08X\n", rreg(e->dma_x, 0x04), rreg(e->dma_x, 0x34), rreg(e->vcore, 0x14));
        }

        if (y) {
            uint32_t final_m = is_stage5 ? k : current_m;
            // V2.0 RTL: S2MM bus is always 128-bit (16 bytes/beat).
            uint32_t final_s2mm_bytes = s2mm_out_beats * 16;
            uint32_t final_pulses = (final_m + 3) / 4;
            
            auto t1 = std::chrono::high_resolution_clock::now();
            sync_cma_for_cpu(e, y_off, final_s2mm_bytes);
            auto t_after_sync = std::chrono::high_resolution_clock::now();
            t_sync_cpu += std::chrono::duration_cast<std::chrono::microseconds>(t_after_sync - t1).count();
            
            if (comp_en_global) {
                const uint16_t* hw_out_16 = (const uint16_t*)((const uint8_t*)e->vbase + y_off);
                float* y_out = (float*)y;
                for (uint32_t beat = 0; beat < s2mm_out_beats; beat++) {
                    const uint16_t* beat_ptr = hw_out_16 + beat * 8;
                    uint32_t row_base = chunk * chunk_size + beat * 4;
                    auto bf16_to_fp32 = [](uint16_t b16) {
                        uint32_t tmp = (uint32_t)b16 << 16;
                        float val; memcpy(&val, &tmp, 4);
                        return val;
                    };
                    for (int p = 0; p < 4; p++) {
                        uint32_t row_idx = row_base + p;
                        if (row_idx < (uint32_t)m) {
                            y_out[row_idx] = bf16_to_fp32(beat_ptr[p]);
                        }
                    }
                }
            } else {
                // Hardware outputs INT32 MAC sums when comp_bit is 0 (Uncompressed mode)
                // Direct 128-bit NEON load from CMA DDR, convert to float, scale, and store to y_out or accumulate to residual
                float* y_out = (float*)y;
                const int32_t* src_i32 = (const int32_t*)((const uint8_t*)e->vbase + y_off);
                float p_scale = 1.0f;
                if (is_stage5) {
                    float s_val = w_scales ? w_scales[3] : 1.0f;
                    p_scale = std::abs(w_scales ? w_scales[2] : 1.0f) / s_val;
                }
                
                uint32_t base_idx = chunk * chunk_size;
                float* dst_f32 = y_out ? y_out + base_idx : nullptr;
                float inv_s = (act_scale > 0.0f) ? (1.0f / act_scale) : 1.0f;

#if defined(__aarch64__) || defined(__arm__)
                if (w_scales && !is_stage5) {
                    float32x4_t vinv_s = vdupq_n_f32(inv_s);
                    if (residual) {
                        float* res_ptr = (float*)residual + base_idx;
                        int i = 0;
                        for (; i <= current_m - 16; i += 16) {
                            float32x4_t v0 = vcvtq_f32_s32(vld1q_s32(src_i32 + i + 0));
                            float32x4_t v1 = vcvtq_f32_s32(vld1q_s32(src_i32 + i + 4));
                            float32x4_t v2 = vcvtq_f32_s32(vld1q_s32(src_i32 + i + 8));
                            float32x4_t v3 = vcvtq_f32_s32(vld1q_s32(src_i32 + i + 12));

                            float32x4_t sc0 = vmulq_f32(vld1q_f32(w_scales + base_idx + i + 0), vinv_s);
                            float32x4_t sc1 = vmulq_f32(vld1q_f32(w_scales + base_idx + i + 4), vinv_s);
                            float32x4_t sc2 = vmulq_f32(vld1q_f32(w_scales + base_idx + i + 8), vinv_s);
                            float32x4_t sc3 = vmulq_f32(vld1q_f32(w_scales + base_idx + i + 12), vinv_s);

                            float32x4_t res0 = vld1q_f32(res_ptr + i + 0);
                            float32x4_t res1 = vld1q_f32(res_ptr + i + 4);
                            float32x4_t res2 = vld1q_f32(res_ptr + i + 8);
                            float32x4_t res3 = vld1q_f32(res_ptr + i + 12);

                            vst1q_f32(res_ptr + i + 0,  vmlaq_f32(res0, v0, sc0));
                            vst1q_f32(res_ptr + i + 4,  vmlaq_f32(res1, v1, sc1));
                            vst1q_f32(res_ptr + i + 8,  vmlaq_f32(res2, v2, sc2));
                            vst1q_f32(res_ptr + i + 12, vmlaq_f32(res3, v3, sc3));
                        }
                        for (; i < current_m; i++) {
                            res_ptr[i] += ((float)src_i32[i] * w_scales[base_idx + i]) * inv_s;
                        }
                    } else if (dst_f32) {
                        int i = 0;
                        for (; i <= current_m - 16; i += 16) {
                            float32x4_t v0 = vcvtq_f32_s32(vld1q_s32(src_i32 + i + 0));
                            float32x4_t v1 = vcvtq_f32_s32(vld1q_s32(src_i32 + i + 4));
                            float32x4_t v2 = vcvtq_f32_s32(vld1q_s32(src_i32 + i + 8));
                            float32x4_t v3 = vcvtq_f32_s32(vld1q_s32(src_i32 + i + 12));

                            float32x4_t sc0 = vmulq_f32(vld1q_f32(w_scales + base_idx + i + 0), vinv_s);
                            float32x4_t sc1 = vmulq_f32(vld1q_f32(w_scales + base_idx + i + 4), vinv_s);
                            float32x4_t sc2 = vmulq_f32(vld1q_f32(w_scales + base_idx + i + 8), vinv_s);
                            float32x4_t sc3 = vmulq_f32(vld1q_f32(w_scales + base_idx + i + 12), vinv_s);

                            vst1q_f32(dst_f32 + i + 0,  vmulq_f32(v0, sc0));
                            vst1q_f32(dst_f32 + i + 4,  vmulq_f32(v1, sc1));
                            vst1q_f32(dst_f32 + i + 8,  vmulq_f32(v2, sc2));
                            vst1q_f32(dst_f32 + i + 12, vmulq_f32(v3, sc3));
                        }
                        for (; i < current_m; i++) {
                            dst_f32[i] = ((float)src_i32[i] * w_scales[base_idx + i]) * inv_s;
                        }
                    }
                } else if (p_scale == 1.0f) {
                    if (dst_f32) {
                        int i = 0;
                        for (; i <= current_m - 16; i += 16) {
                            int32x4_t v0 = vld1q_s32(src_i32 + i + 0);
                            int32x4_t v1 = vld1q_s32(src_i32 + i + 4);
                            int32x4_t v2 = vld1q_s32(src_i32 + i + 8);
                            int32x4_t v3 = vld1q_s32(src_i32 + i + 12);

                            vst1q_f32(dst_f32 + i + 0,  vcvtq_f32_s32(v0));
                            vst1q_f32(dst_f32 + i + 4,  vcvtq_f32_s32(v1));
                            vst1q_f32(dst_f32 + i + 8,  vcvtq_f32_s32(v2));
                            vst1q_f32(dst_f32 + i + 12, vcvtq_f32_s32(v3));
                        }
                        for (; i < current_m; i++) {
                            dst_f32[i] = (float)src_i32[i];
                        }
                    }
                } else {
                    if (dst_f32) {
                        float32x4_t vscale = vdupq_n_f32(p_scale);
                        int i = 0;
                        for (; i <= current_m - 16; i += 16) {
                            int32x4_t v0 = vld1q_s32(src_i32 + i + 0);
                            int32x4_t v1 = vld1q_s32(src_i32 + i + 4);
                            int32x4_t v2 = vld1q_s32(src_i32 + i + 8);
                            int32x4_t v3 = vld1q_s32(src_i32 + i + 12);

                            vst1q_f32(dst_f32 + i + 0,  vmulq_f32(vcvtq_f32_s32(v0), vscale));
                            vst1q_f32(dst_f32 + i + 4,  vmulq_f32(vcvtq_f32_s32(v1), vscale));
                            vst1q_f32(dst_f32 + i + 8,  vmulq_f32(vcvtq_f32_s32(v2), vscale));
                            vst1q_f32(dst_f32 + i + 12, vmulq_f32(vcvtq_f32_s32(v3), vscale));
                        }
                        for (; i < current_m; i++) {
                            dst_f32[i] = (float)src_i32[i] * p_scale;
                        }
                    }
                }
#else
                if (w_scales && !is_stage5) {
                    if (residual) {
                        float* res_ptr = (float*)residual + base_idx;
                        for (int i = 0; i < current_m; i++) {
                            res_ptr[i] += ((float)src_i32[i] * w_scales[base_idx + i]) * inv_s;
                        }
                    } else if (dst_f32) {
                        for (int i = 0; i < current_m; i++) {
                            dst_f32[i] = ((float)src_i32[i] * w_scales[base_idx + i]) * inv_s;
                        }
                    }
                } else if (p_scale == 1.0f) {
                    if (dst_f32) for (int i = 0; i < current_m; i++) dst_f32[i] = (float)src_i32[i];
                } else {
                    if (dst_f32) for (int i = 0; i < current_m; i++) dst_f32[i] = (float)src_i32[i] * p_scale;
                }
#endif
            }

            auto t3 = std::chrono::high_resolution_clock::now();
            t_post += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t_after_sync).count();
        }
        
        ffn_stream_done:;
        auto end_t = std::chrono::high_resolution_clock::now();
        t_dma += std::chrono::duration_cast<std::chrono::microseconds>(end_t - start_t).count();
    }
    return 0;
}

int v37_launch_linear_async(v37_engine_t* e, const int8_t* x, int s_id, int m, int k, v37_mode_t mod, int stage, int reuse_act) {
    if (!e || s_id < 0 || s_id >= 512 || e->slot_offsets[s_id] == 0) return -2;

    auto start_t = std::chrono::high_resolution_clock::now();
    uint64_t start_vct = arm64_get_cntvct();
    t_calls++;

    uint32_t k_beats = (k + 63) / 64;
    k_beats = (k_beats + 3) & ~3; // Pad to multiple of 4
    uint32_t pe_k_beats = k_beats;
    uint32_t pulses = (m + 3) / 4;
    uint32_t pe_out_beats = pulses;
    uint32_t y_off = 0x600000u;
    uint32_t s2mm_bytes = pulses * 16;

    v37_mode_t real_hw_mode = (v37_mode_t)(mod & 0xFFFF);
    uint32_t hw_mode_to_prog = real_hw_mode;
    const char* force_env = getenv("V37_FORCE_HW_MODE");
    if (force_env) {
        hw_mode_to_prog = (uint32_t)atoi(force_env);
    }
    uint32_t cfg_val = (pe_k_beats << 16) | hw_mode_to_prog;

    if (!reuse_act) {
        uint32_t act_dma_sz = pe_k_beats * 64;
        uint8_t* act_ptr = (uint8_t*)e->cma_vbase + 0x400000u;
        memcpy(act_ptr, x, k);
        if (act_dma_sz > (uint32_t)k) {
            memset(act_ptr + k, 0, act_dma_sz - k);
        }
        cpu_clean_dcache(act_ptr, pe_k_beats * 64);
    } else if (reuse_act == 2) {
        // [Stage 1 Direct-CMA]: x is already written directly into CMA buffer (0x400000u) by caller (zero-copy)
        uint32_t act_dma_sz = pe_k_beats * 64;
        uint8_t* act_ptr = (uint8_t*)e->cma_vbase + 0x400000u;
        if (act_dma_sz > (uint32_t)k) {
            memset(act_ptr + k, 0, act_dma_sz - k);
        }
        cpu_clean_dcache(act_ptr, pe_k_beats * 64);
    }

    uint32_t dma_out_bit = (1 << 25);
    uint32_t pe_out_reg = (pe_out_beats & 0xFFFF) | dma_out_bit;

    if (!e->ucode_mode) {
        wreg(e->vcore, 0x08, cfg_val);
        wreg(e->vcore, 0x0C, pe_out_reg);
        // 0x10 and 0x14 are statically initialized to 0x01000000 at engine creation

        wreg(e->vcore, 0x00, 2);
        for (volatile int delay_idx = 0; delay_idx < 10; delay_idx++) {}
        wreg(e->vcore, 0x00, 0);
    }

    wreg(e->dma_x, 0x34, 0x7000);
    wreg(e->dma_x, 0x48, e->phys + y_off);
    wreg(e->dma_x, 0x58, s2mm_bytes);

    if (reuse_act == 0 || reuse_act == 2) {
        wreg(e->dma_x, 0x04, 0x7000);
        wreg(e->dma_x, 0x18, e->phys + 0x400000u);
        wreg(e->dma_x, 0x28, pe_k_beats * 64);
        // [Stage 1 Non-blocking MM2S]: Concurrent streaming with 4-channel weight DMAs
    }

    uint32_t m_blocks = (m + 3) / 4;
    uint32_t ch_stride;
    uint32_t channel_sz;
    uint32_t current_matrix_beats;
    uint32_t hdr_bytes = e->ucode_mode ? 0 : 16;

    if (real_hw_mode == V37_MODE_TL3_COMPACT) {
        ch_stride = pe_k_beats * 13;
        uint32_t total_payload = m_blocks * ch_stride;
        current_matrix_beats = (total_payload + 15) / 16;
        channel_sz = current_matrix_beats * 16 + hdr_bytes;
        channel_sz = (channel_sz + 63) & ~63;
    } else {
        uint32_t total_beats = pe_k_beats;
        if (real_hw_mode != V37_MODE_I2_S && real_hw_mode != V37_MODE_TL3) {
            total_beats = pe_k_beats + (pe_k_beats + 3) / 4;
        }
        ch_stride = total_beats * 16;
        channel_sz = m_blocks * ch_stride + hdr_bytes;
        channel_sz = (channel_sz + 63) & ~63;
        current_matrix_beats = m_blocks * total_beats;
    }
    uint32_t w_len = current_matrix_beats * 16 + hdr_bytes;

    for (int i = 0; i < 4; i++) {
        uint32_t ch_base = i * channel_sz;
        wreg(e->dma_w[i], 0x18, e->phys + e->slot_offsets[s_id] + ch_base);
        wreg(e->dma_w[i], 0x28, w_len);
    }

    auto t_before_poll = std::chrono::high_resolution_clock::now();
    t_setup += std::chrono::duration_cast<std::chrono::microseconds>(t_before_poll - start_t).count();

    e->async_job.m = m;
    e->async_job.k = k;
    e->async_job.s_id = s_id;
    e->async_job.stage = stage;
    e->async_job.s2mm_out_beats = pulses;
    e->async_job.y_off = y_off;
    e->async_job.start_t = start_t;
    e->async_job.start_vct = start_vct;
    e->async_job.t_before_poll = t_before_poll;
    e->async_job.active = true;

    return 0;
}

int v37_wait_linear_async(v37_engine_t* e, void* y, const float* w_scales, float act_scale, float* residual) {
    if (!e || !e->async_job.active) return -1;

    int m = e->async_job.m;
    uint32_t y_off = e->async_job.y_off;
    uint32_t s2mm_out_beats = e->async_job.s2mm_out_beats;
    auto t_before_poll = e->async_job.t_before_poll;
    auto start_t = e->async_job.start_t;

    hybrid_wait_s2mm(e, start_t, m, e->async_job.k, e->async_job.start_vct);

    int stuck_cnt = 0;
    uint32_t final_s2mm_stat = 0;
    while (!((final_s2mm_stat = rreg(e->dma_x, 0x34)) & 0x1002)) {
        cpu_relax();
        if (++stuck_cnt > 10000000) {
            printf("[DEBUG] Hung at S2MM Poll in wait_async! stat=%08X\n", final_s2mm_stat);
            e->async_job.active = false;
            return -3;
        }
    }
    // wreg(e->dma_x, 0x34, 0x7000); // Cleared by next DMA launch

    auto t_after_poll = std::chrono::high_resolution_clock::now();
    t_poll += std::chrono::duration_cast<std::chrono::microseconds>(t_after_poll - t_before_poll).count();

    static int cal_print_cnt = 0;
    if (getenv("V37_BENCH_CALIBRATE") && cal_print_cnt++ < 20) {
        printf("[CALIBRATE] AsyncGEMV m=%d, k=%d -> actual_wait = %ld us\n", m, e->async_job.k,
               (long)std::chrono::duration_cast<std::chrono::microseconds>(t_after_poll - start_t).count());
    }

    uint32_t final_s2mm_bytes = s2mm_out_beats * 16;
    auto t1 = std::chrono::high_resolution_clock::now();
    sync_cma_for_cpu(e, y_off, final_s2mm_bytes);
    auto t_after_sync = std::chrono::high_resolution_clock::now();
    t_sync_cpu += std::chrono::duration_cast<std::chrono::microseconds>(t_after_sync - t1).count();

    float* y_out = (float*)y;
    const int32_t* src_i32 = (const int32_t*)((const uint8_t*)e->vbase + y_off);
    float inv_s = (act_scale > 0.0f) ? (1.0f / act_scale) : 1.0f;

#if defined(__aarch64__) || defined(__arm__)
    if (w_scales) {
        float32x4_t vinv_s = vdupq_n_f32(inv_s);
        if (residual) {
            float* res_ptr = (float*)residual;
            if (m >= 8192) {
                #pragma omp parallel for schedule(static)
                for (int b = 0; b < 4; b++) {
                    int start = b * (m / 4);
                    int end = (b == 3) ? m : (start + (m / 4));
                    int i = start;
                    for (; i <= end - 16; i += 16) {
                        float32x4_t v0 = vcvtq_f32_s32(vld1q_s32(src_i32 + i + 0));
                        float32x4_t v1 = vcvtq_f32_s32(vld1q_s32(src_i32 + i + 4));
                        float32x4_t v2 = vcvtq_f32_s32(vld1q_s32(src_i32 + i + 8));
                        float32x4_t v3 = vcvtq_f32_s32(vld1q_s32(src_i32 + i + 12));

                        float32x4_t sc0 = vmulq_f32(vld1q_f32(w_scales + i + 0), vinv_s);
                        float32x4_t sc1 = vmulq_f32(vld1q_f32(w_scales + i + 4), vinv_s);
                        float32x4_t sc2 = vmulq_f32(vld1q_f32(w_scales + i + 8), vinv_s);
                        float32x4_t sc3 = vmulq_f32(vld1q_f32(w_scales + i + 12), vinv_s);

                        float32x4_t res0 = vld1q_f32(res_ptr + i + 0);
                        float32x4_t res1 = vld1q_f32(res_ptr + i + 4);
                        float32x4_t res2 = vld1q_f32(res_ptr + i + 8);
                        float32x4_t res3 = vld1q_f32(res_ptr + i + 12);

                        vst1q_f32(res_ptr + i + 0,  vmlaq_f32(res0, v0, sc0));
                        vst1q_f32(res_ptr + i + 4,  vmlaq_f32(res1, v1, sc1));
                        vst1q_f32(res_ptr + i + 8,  vmlaq_f32(res2, v2, sc2));
                        vst1q_f32(res_ptr + i + 12, vmlaq_f32(res3, v3, sc3));
                    }
                    for (; i < end; i++) {
                        res_ptr[i] += ((float)src_i32[i] * w_scales[i]) * inv_s;
                    }
                }
            } else {
                int i = 0;
                for (; i <= m - 16; i += 16) {
                    float32x4_t v0 = vcvtq_f32_s32(vld1q_s32(src_i32 + i + 0));
                    float32x4_t v1 = vcvtq_f32_s32(vld1q_s32(src_i32 + i + 4));
                    float32x4_t v2 = vcvtq_f32_s32(vld1q_s32(src_i32 + i + 8));
                    float32x4_t v3 = vcvtq_f32_s32(vld1q_s32(src_i32 + i + 12));

                    float32x4_t sc0 = vmulq_f32(vld1q_f32(w_scales + i + 0), vinv_s);
                    float32x4_t sc1 = vmulq_f32(vld1q_f32(w_scales + i + 4), vinv_s);
                    float32x4_t sc2 = vmulq_f32(vld1q_f32(w_scales + i + 8), vinv_s);
                    float32x4_t sc3 = vmulq_f32(vld1q_f32(w_scales + i + 12), vinv_s);

                    float32x4_t res0 = vld1q_f32(res_ptr + i + 0);
                    float32x4_t res1 = vld1q_f32(res_ptr + i + 4);
                    float32x4_t res2 = vld1q_f32(res_ptr + i + 8);
                    float32x4_t res3 = vld1q_f32(res_ptr + i + 12);

                    vst1q_f32(res_ptr + i + 0,  vmlaq_f32(res0, v0, sc0));
                    vst1q_f32(res_ptr + i + 4,  vmlaq_f32(res1, v1, sc1));
                    vst1q_f32(res_ptr + i + 8,  vmlaq_f32(res2, v2, sc2));
                    vst1q_f32(res_ptr + i + 12, vmlaq_f32(res3, v3, sc3));
                }
                for (; i < m; i++) {
                    res_ptr[i] += ((float)src_i32[i] * w_scales[i]) * inv_s;
                }
            }
        } else if (y_out) {
            if (m >= 8192) {
                #pragma omp parallel for schedule(static)
                for (int b = 0; b < 4; b++) {
                    int start = b * (m / 4);
                    int end = (b == 3) ? m : (start + (m / 4));
                    int i = start;
                    for (; i <= end - 16; i += 16) {
                        float32x4_t v0 = vcvtq_f32_s32(vld1q_s32(src_i32 + i + 0));
                        float32x4_t v1 = vcvtq_f32_s32(vld1q_s32(src_i32 + i + 4));
                        float32x4_t v2 = vcvtq_f32_s32(vld1q_s32(src_i32 + i + 8));
                        float32x4_t v3 = vcvtq_f32_s32(vld1q_s32(src_i32 + i + 12));

                        float32x4_t sc0 = vmulq_f32(vld1q_f32(w_scales + i + 0), vinv_s);
                        float32x4_t sc1 = vmulq_f32(vld1q_f32(w_scales + i + 4), vinv_s);
                        float32x4_t sc2 = vmulq_f32(vld1q_f32(w_scales + i + 8), vinv_s);
                        float32x4_t sc3 = vmulq_f32(vld1q_f32(w_scales + i + 12), vinv_s);

                        vst1q_f32(y_out + i + 0,  vmulq_f32(v0, sc0));
                        vst1q_f32(y_out + i + 4,  vmulq_f32(v1, sc1));
                        vst1q_f32(y_out + i + 8,  vmulq_f32(v2, sc2));
                        vst1q_f32(y_out + i + 12, vmulq_f32(v3, sc3));
                    }
                    for (; i < end; i++) {
                        y_out[i] = ((float)src_i32[i] * w_scales[i]) * inv_s;
                    }
                }
            } else {
                int i = 0;
                for (; i <= m - 16; i += 16) {
                    float32x4_t v0 = vcvtq_f32_s32(vld1q_s32(src_i32 + i + 0));
                    float32x4_t v1 = vcvtq_f32_s32(vld1q_s32(src_i32 + i + 4));
                    float32x4_t v2 = vcvtq_f32_s32(vld1q_s32(src_i32 + i + 8));
                    float32x4_t v3 = vcvtq_f32_s32(vld1q_s32(src_i32 + i + 12));

                    float32x4_t sc0 = vmulq_f32(vld1q_f32(w_scales + i + 0), vinv_s);
                    float32x4_t sc1 = vmulq_f32(vld1q_f32(w_scales + i + 4), vinv_s);
                    float32x4_t sc2 = vmulq_f32(vld1q_f32(w_scales + i + 8), vinv_s);
                    float32x4_t sc3 = vmulq_f32(vld1q_f32(w_scales + i + 12), vinv_s);

                    vst1q_f32(y_out + i + 0,  vmulq_f32(v0, sc0));
                    vst1q_f32(y_out + i + 4,  vmulq_f32(v1, sc1));
                    vst1q_f32(y_out + i + 8,  vmulq_f32(v2, sc2));
                    vst1q_f32(y_out + i + 12, vmulq_f32(v3, sc3));
                }
                for (; i < m; i++) {
                    y_out[i] = ((float)src_i32[i] * w_scales[i]) * inv_s;
                }
            }
        }
    } else {
        if (y_out) {
            for (int i = 0; i < m; i++) y_out[i] = (float)src_i32[i];
        }
    }
#else
    if (w_scales) {
        if (residual) {
            float* res_ptr = (float*)residual;
            for (int i = 0; i < m; i++) res_ptr[i] += ((float)src_i32[i] * w_scales[i]) * inv_s;
        } else if (y_out) {
            for (int i = 0; i < m; i++) y_out[i] = ((float)src_i32[i] * w_scales[i]) * inv_s;
        }
    } else if (y_out) {
        for (int i = 0; i < m; i++) y_out[i] = (float)src_i32[i];
    }
#endif

    auto t3 = std::chrono::high_resolution_clock::now();
    t_post += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t_after_sync).count();
    t_dma += std::chrono::duration_cast<std::chrono::microseconds>(t3 - start_t).count();

    e->async_job.active = false;
    return 0;
}

int v37_wait_linear_async_fused_up(v37_engine_t* e, float* ffn_act, const float* gate, const float* w_scales, float act_scale) {
    if (!e || !e->async_job.active) return -1;

    int m = e->async_job.m;
    uint32_t y_off = e->async_job.y_off;
    uint32_t s2mm_out_beats = e->async_job.s2mm_out_beats;
    auto t_before_poll = e->async_job.t_before_poll;
    auto start_t = e->async_job.start_t;

    hybrid_wait_s2mm(e, start_t, m, e->async_job.k, e->async_job.start_vct);

    int stuck_cnt = 0;
    uint32_t final_s2mm_stat = 0;
    while (!((final_s2mm_stat = rreg(e->dma_x, 0x34)) & 0x1002)) {
        cpu_relax();
        if (++stuck_cnt > 10000000) {
            printf("[DEBUG] Hung at S2MM Poll in wait_async! stat=%08X\n", final_s2mm_stat);
            e->async_job.active = false;
            return -3;
        }
    }
    // wreg(e->dma_x, 0x34, 0x7000); // Cleared by next DMA launch

    auto t_after_poll = std::chrono::high_resolution_clock::now();
    t_poll += std::chrono::duration_cast<std::chrono::microseconds>(t_after_poll - t_before_poll).count();

    uint32_t final_s2mm_bytes = s2mm_out_beats * 16;
    auto t1 = std::chrono::high_resolution_clock::now();
    sync_cma_for_cpu(e, y_off, final_s2mm_bytes);
    auto t_after_sync = std::chrono::high_resolution_clock::now();
    t_sync_cpu += std::chrono::duration_cast<std::chrono::microseconds>(t_after_sync - t1).count();

    const int32_t* src_i32 = (const int32_t*)((const uint8_t*)e->vbase + y_off);
    float inv_s = (act_scale > 0.0f) ? (1.0f / act_scale) : 1.0f;

#if defined(__aarch64__) || defined(__arm__)
    if (w_scales && ffn_act && gate) {
        float32x4_t vinv_s = vdupq_n_f32(inv_s);
        #pragma omp parallel for schedule(static)
        for (int b = 0; b < 4; b++) {
            int start = b * (m / 4);
            int end = (b == 3) ? m : (start + (m / 4));
            int i = start;
            for (; i <= end - 16; i += 16) {
                float32x4_t v0 = vcvtq_f32_s32(vld1q_s32(src_i32 + i + 0));
                float32x4_t v1 = vcvtq_f32_s32(vld1q_s32(src_i32 + i + 4));
                float32x4_t v2 = vcvtq_f32_s32(vld1q_s32(src_i32 + i + 8));
                float32x4_t v3 = vcvtq_f32_s32(vld1q_s32(src_i32 + i + 12));

                float32x4_t sc0 = vmulq_f32(vld1q_f32(w_scales + i + 0), vinv_s);
                float32x4_t sc1 = vmulq_f32(vld1q_f32(w_scales + i + 4), vinv_s);
                float32x4_t sc2 = vmulq_f32(vld1q_f32(w_scales + i + 8), vinv_s);
                float32x4_t sc3 = vmulq_f32(vld1q_f32(w_scales + i + 12), vinv_s);

                float32x4_t u0 = vmulq_f32(v0, sc0);
                float32x4_t u1 = vmulq_f32(v1, sc1);
                float32x4_t u2 = vmulq_f32(v2, sc2);
                float32x4_t u3 = vmulq_f32(v3, sc3);

                float32x4_t g0 = vld1q_f32(gate + i + 0);
                float32x4_t g1 = vld1q_f32(gate + i + 4);
                float32x4_t g2 = vld1q_f32(gate + i + 8);
                float32x4_t g3 = vld1q_f32(gate + i + 12);

                vst1q_f32(ffn_act + i + 0,  vmulq_f32(g0, u0));
                vst1q_f32(ffn_act + i + 4,  vmulq_f32(g1, u1));
                vst1q_f32(ffn_act + i + 8,  vmulq_f32(g2, u2));
                vst1q_f32(ffn_act + i + 12, vmulq_f32(g3, u3));
            }
            for (; i < end; i++) {
                float u = ((float)src_i32[i] * w_scales[i]) * inv_s;
                ffn_act[i] = gate[i] * u;
            }
        }
    }
#else
    if (w_scales && ffn_act && gate) {
        for (int i = 0; i < m; i++) {
            float u = ((float)src_i32[i] * w_scales[i]) * inv_s;
            ffn_act[i] = gate[i] * u;
        }
    }
#endif

    auto t3 = std::chrono::high_resolution_clock::now();
    t_post += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t_after_sync).count();
    t_dma += std::chrono::duration_cast<std::chrono::microseconds>(t3 - start_t).count();

    e->async_job.active = false;
    return 0;
}

int v37_wait_linear_async_f16(v37_engine_t* e, void* kv_cache_v_dst, const float* w_scales, float act_scale) {
    if (!e || !e->async_job.active) return -1;

    int m = e->async_job.m;
    uint32_t y_off = e->async_job.y_off;
    uint32_t s2mm_out_beats = e->async_job.s2mm_out_beats;
    auto t_before_poll = e->async_job.t_before_poll;
    auto start_t = e->async_job.start_t;

    hybrid_wait_s2mm(e, start_t, m, e->async_job.k, e->async_job.start_vct);

    int stuck_cnt = 0;
    uint32_t final_s2mm_stat = 0;
    while (!((final_s2mm_stat = rreg(e->dma_x, 0x34)) & 0x1002)) {
        cpu_relax();
        if (++stuck_cnt > 10000000) {
            printf("[DEBUG] Hung at S2MM Poll in wait_async! stat=%08X\n", final_s2mm_stat);
            e->async_job.active = false;
            return -3;
        }
    }
    // wreg(e->dma_x, 0x34, 0x7000); // Cleared by next DMA launch

    auto t_after_poll = std::chrono::high_resolution_clock::now();
    t_poll += std::chrono::duration_cast<std::chrono::microseconds>(t_after_poll - t_before_poll).count();

    uint32_t final_s2mm_bytes = s2mm_out_beats * 16;
    auto t1 = std::chrono::high_resolution_clock::now();
    sync_cma_for_cpu(e, y_off, final_s2mm_bytes);
    auto t_after_sync = std::chrono::high_resolution_clock::now();
    t_sync_cpu += std::chrono::duration_cast<std::chrono::microseconds>(t_after_sync - t1).count();

    const int32_t* src_i32 = (const int32_t*)((const uint8_t*)e->vbase + y_off);
    float inv_s = (act_scale > 0.0f) ? (1.0f / act_scale) : 1.0f;
    uint16_t* dst_u16 = (uint16_t*)kv_cache_v_dst;

#if defined(__aarch64__) || defined(__arm__)
    if (w_scales && dst_u16) {
        float32x4_t vinv_s = vdupq_n_f32(inv_s);
        int i = 0;
        for (; i <= m - 8; i += 8) {
            float32x4_t v0 = vcvtq_f32_s32(vld1q_s32(src_i32 + i + 0));
            float32x4_t v1 = vcvtq_f32_s32(vld1q_s32(src_i32 + i + 4));

            float32x4_t sc0 = vmulq_f32(vld1q_f32(w_scales + i + 0), vinv_s);
            float32x4_t sc1 = vmulq_f32(vld1q_f32(w_scales + i + 4), vinv_s);

            float32x4_t r0 = vmulq_f32(v0, sc0);
            float32x4_t r1 = vmulq_f32(v1, sc1);

            float16x4_t h0 = vcvt_f16_f32(r0);
            float16x4_t h1 = vcvt_f16_f32(r1);

            vst1q_u16(dst_u16 + i, vreinterpretq_u16_f16(vcombine_f16(h0, h1)));
        }
        for (; i < m; i++) {
            float val = ((float)src_i32[i] * w_scales[i]) * inv_s;
            __fp16 h = (__fp16)val;
            memcpy(dst_u16 + i, &h, sizeof(uint16_t));
        }
    }
#else
    if (w_scales && dst_u16) {
        for (int i = 0; i < m; i++) {
            float val = ((float)src_i32[i] * w_scales[i]) * inv_s;
            // scalar fallback
            uint16_t h = 0;
            dst_u16[i] = h;
        }
    }
#endif

    auto t3 = std::chrono::high_resolution_clock::now();
    t_post += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t_after_sync).count();
    t_dma += std::chrono::duration_cast<std::chrono::microseconds>(t3 - start_t).count();

    e->async_job.active = false;
    return 0;
}

// Removed v37_repack_weights_to_cpu

void v37_load_loopback_rmsnorm_weights(v37_engine_t* e, const float* weights, int dim) {
    if (!e || e->vcore == MAP_FAILED) return;

    // Use shared rmsnorm_prep_inst: set bit 16 (load_rms_weight) to enter weight load mode.
    // This overwrites the current RMSNorm weights — caller must reload attn/ffn_norm
    // weights if switching back to input pre-norm mode.
    // Write bit 28 of 0x0C (dynamic_cfg_word[60]) to enter load_rms_weight mode
    uint32_t cfg = rreg(e->vcore, 0x0C);
    wreg(e->vcore, 0x0C, cfg | (1 << 28));
    
    // 2. FP32 → Q15 转换，写入 CMA
    int16_t* dst = (int16_t*)((uint8_t*)e->vbase + 0x400000);
    for (int i = 0; i < dim; i++) {
        float val = weights[i] * 32768.0f;
        dst[i] = (int16_t)std::max(-32768.0f, std::min(32767.0f, val));
    }
    
    // 3. DMA 发送
    int beats = (dim + 31) / 32;  // 每beat 32个INT16
    int bytes = beats * 64;       // 每beat 64字节 (512bit)
    sync_cma_for_device(e, 0x400000, bytes);
    
    // Clear MM2S status
    wreg(e->dma_x, 0x04, 0x7000);
    wreg(e->dma_x, 0x18, e->phys + 0x400000);
    wreg(e->dma_x, 0x00, 0x01); // Start MM2S
    wreg(e->dma_x, 0x28, bytes); // Length
    
    // 4. 等待完成
    int stuck_cnt = 0;
    while (!(rreg(e->dma_x, 0x04) & 0x1002)) {
        if (++stuck_cnt > 1000000) {
            printf("[V37 ERROR] Hung during loopback RMSNorm weight load! stat=%08X\n", rreg(e->dma_x, 0x04));
            break;
        }
    }
    wreg(e->dma_x, 0x04, 0x7000); // clear status
    
    // 5. 恢复 bit 16
    wreg(e->vcore, 0x0C, cfg);
}

int v37_preload_weights(v37_engine_t* e, const uint8_t* d, const float* w_scales, size_t sz, int s_id, v37_mode_t mod, int k, int BK) {
    (void)BK;
    if(!e) return -1;
    
    int padded_k = (k + 255) & ~255;
    uint32_t real_mode = ((uint32_t)mod & 0xFF);
    
    if (real_mode == V37_MODE_I2_S) {
        int padded_k = (k + 255) & ~255;
        int m = (sz * 4) / padded_k; // Use padded_k since test_03 allocates based on padded_k
        
        uint32_t kb = (k + 63) / 64;
        kb = (kb + 3) & ~3;
        uint32_t total_beats = kb; // PLD only for I2_S
        uint32_t ch_stride = total_beats * 16; 
        uint32_t channel_sz = ch_stride * ((m+3)/4) + 16; // +16 for Command Header
        channel_sz = (channel_sz + 63) & ~63; // Pad to 64-byte alignment
        
        if (g_cma_offsets.find(d) != g_cma_offsets.end()) {
            e->slot_offsets[s_id] = g_cma_offsets[d];
            printf("CACHE HIT!\n"); return 0;
        }
        
        uint32_t alloc_sz = 4 * channel_sz;
        uint32_t cma_offset = e->current_cma_offset;
        
        g_cma_offsets[d] = cma_offset;
        e->slot_offsets[s_id] = cma_offset;
        e->current_cma_offset = (e->current_cma_offset + alloc_sz + 4095) & ~4095;
        
        if (e->current_cma_offset > e->udma_size) {
            printf("[V37-ERROR] CMA overflow! alloc_sz=%u offset=0x%X max=0x%lX\n", alloc_sz, e->current_cma_offset, e->udma_size);
            return -1;
        }
        
        uint8_t* dest = (uint8_t*)e->cma_vbase + cma_offset;
        
        volatile uint32_t* clear_ptr = (volatile uint32_t*)dest;
        for (uint32_t i = 0; i < alloc_sz / 4; i++) {
            clear_ptr[i] = 0;
        }
        
        // --- PACK QK_I2_S=128 GGUF LAYOUT TO FPGA SIMD LAYOUT ---
        // The GGUF I2_S tensor uses QK_I2_S=128 MSB-first format (ACT_PARALLEL mode):
        //   Each 128-weight block → 32 bytes
        //   byte[block*32 + group_pos]:
        //     bits 6-7 = col block*128 + group_pos + 0*32  (group 0)
        //     bits 4-5 = col block*128 + group_pos + 1*32  (group 1)
        //     bits 2-3 = col block*128 + group_pos + 2*32  (group 2)
        //     bits 0-1 = col block*128 + group_pos + 3*32  (group 3)
        //   Values: {0, 1, 2} → {-1, 0, +1}
        //
        // Per-row byte count in GGUF = padded_k / 4 (with w_scale appended after)
        
        for (int row = 0; row < m; row++) {
            int ch_idx = row % 4;
            int row_in_ch = row / 4;
            uint32_t ch_base = ch_idx * channel_sz + 16 + row_in_ch * ch_stride;
            
            uint32_t pld_idx = 0;
            uint32_t dest_beat = 0;
            
            // Extract the per-channel scale and convert to Q24 format for the PE Array.
            // IMPORTANT: The RTL's fixed-point multiplier in compute_tile cannot handle
            // negative beta values (produces 0). To support negative w_scales, we use
            // abs(w_scale) for beta and flip the ternary weight encoding (+1 <-> -1).
            float ch_scale = w_scales ? w_scales[0] : 1.0f;
            bool negate_weights = (ch_scale < 0.0f);
            float abs_scale = negate_weights ? -ch_scale : ch_scale;
            uint32_t beta_q24 = (uint32_t)(abs_scale * 16777216.0f); // Q24, always positive
            
            int padded_k_local = (k + 255) & ~255;
            int row_byte_offset = row * (padded_k_local / 4);
            
            while (pld_idx < kb) {
                // V9.5 RTL fixed the I2_S unpacking bug. No dummy header is required.
                for (int b = 0; b < 4 && pld_idx < kb; b++) {
                    uint8_t* dst_ptr = dest + ch_base + dest_beat * 16;
                    for (int x = 0; x < 16; x++) {
                        uint8_t q[4] = {1, 1, 1, 1}; // Default to 1 (maps to 0 in hw)
                        for (int i = 0; i < 4; i++) {
                            int col = pld_idx * 64 + x + i * 16;
                            if (col < k) {
                                // QK_I2_S=128 MSB-first decode
                                int block_idx  = col / 128;
                                int col_in_blk = col % 128;
                                int group_idx  = col_in_blk / 32;
                                int group_pos  = col_in_blk % 32;
                                int gguf_byte_idx = row_byte_offset + block_idx * 32 + group_pos;
                                int shift = 6 - 2 * group_idx;  // MSB-first
                                uint8_t code = (d[gguf_byte_idx] >> shift) & 3;
                                // If w_scale is negative, flip +1 (10) <-> -1 (00), keep 0 (01,11) unchanged
                                if (negate_weights) {
                                    if (code == 0) code = 2;       // -1 -> +1
                                    else if (code == 2) code = 0;  // +1 -> -1
                                }
                                q[i] = code;
                            }
                        }
                        dst_ptr[x] = (q[0] << 6) | (q[1] << 4) | (q[2] << 2) | (q[3] << 0);
                    }
                    dest_beat++; pld_idx++;
                }
            }
        }
        
        if (k == 2560 || k == 2048) {
            // Debug printing is disabled.
        }
        sync_cma_for_device(e, cma_offset, alloc_sz);
    } else {
        e->slot_offsets[s_id] = e->current_cma_offset;
        e->current_cma_offset = (e->current_cma_offset + sz + 4095) & ~4095;
        memcpy((uint8_t*)e->cma_vbase + e->slot_offsets[s_id], d, sz);
        sync_cma_for_device(e, e->slot_offsets[s_id], sz);
    }
    
    return 0;
}

extern "C" void v37_reset_cma_allocator(v37_engine_t* e, int slot_id) {
    if (e) {
        memset(e->slot_offsets, 0, sizeof(e->slot_offsets));
        g_cma_offsets.clear();
        e->current_cma_offset = 0x800000; // Reset to 8MB, avoiding activation/output buffers
    }
}

int v37_compute_linear(v37_engine_t* e, const int8_t* x, const uint8_t* w, const float* w_scales, void* y, int m, int k, v37_mode_t mod, const int8_t* act_scales, int BK, int stage) {
    (void)BK;
    (void)stage;
    if(!e || !x || !y) return -1;
    
    if (mod != V37_MODE_I2_S_PREPACKED) {
        int padded_k = (k + 255) & ~255;
        size_t w_sz = (((uint32_t)mod & 0xFF) == V37_MODE_I2_S) ? (m * padded_k) / 4 : (m * padded_k); 
        v37_preload_weights(e, w, w_scales, w_sz, 0, mod, k, 0);
    }
    return v37_compute_linear_preloaded(e, x, 0, y, m, k, mod, act_scales, w_scales, BK, stage, 0, m);
}

void v37_destroy_engine(v37_engine_t* e) { 
    g_cma_offsets.clear();
    if(e) {
        if(e->vcore != MAP_FAILED && e->vcore) munmap((void*)e->vcore, 0x1000);
        if(e->dma_x != MAP_FAILED && e->dma_x) munmap((void*)e->dma_x, 0x1000);
        for(int i=0; i<4; i++) {
            if(e->dma_w[i] != MAP_FAILED && e->dma_w[i]) munmap((void*)e->dma_w[i], 0x1000);
        }
        if(e->vbase != MAP_FAILED && e->vbase) munmap((void*)e->vbase, e->udma_size);
        if(e->fd_v37 >= 0) close(e->fd_v37);
        if(e->fd_mem >= 0) close(e->fd_mem);
        if(e->fd_udma >= 0) close(e->fd_udma);
        free(e); 
    }
}

void v37_sg_flush(v37_engine_t* e) {
    if (e->desc_count == 0) return;
    if (e->sg_dispatcher == MAP_FAILED) return;
    
    // Ensure all descriptor writes are visible to FPGA
    sync_cma_for_device(e, 0x200000u, e->desc_count * sizeof(v37_sg_desc_t));
    
    // Physical address of first descriptor
    uint32_t head_phys = e->phys + 0x200000u;
    
    // Write head_ptr to 0x00
    wreg(e->sg_dispatcher, 0x00, head_phys);
    
    // Start dispatcher (write 1 to 0x04)
    wreg(e->sg_dispatcher, 0x04, 1);
    
    // Wait for completion (poll state == 0 after starting)
    for (volatile int i=0; i<1000; i++); // Wait for IDLE -> READ_DESC transition
    
    int stuck_cnt = 0;
    while (rreg(e->sg_dispatcher, 0x08) != 0) { // ST_IDLE is 0
        if (++stuck_cnt > 10000000) {
            printf("[V37 ERROR] SG Dispatcher Hung! state = %d\n", rreg(e->sg_dispatcher, 0x08));
            break;
        }
    }
    
    e->desc_count = 0; // Reset for next batch
    
    // Wait an additional 50us to ensure the PE array has fully drained the AXI-Stream FIFO
    // before any subsequent soft_reset is pulsed by the next matrix operation.
    usleep(50);
}

int v37_load_rope_table(v37_engine_t* engine, const float* sin_table, const float* cos_table, int half_dim_beats) {
    if (!engine || !sin_table || !cos_table) return -1;
    
    // The DMA sends 512-bit (64-byte) beats. rope_inst only reads the lower 128-bit (16-byte) per beat.
    // Each BRAM entry requires two 128-bit beats (Cos then Sin).
    // So we need 2 beats per half_dim_beat.
    int total_beats = half_dim_beats * 2;
    int total_bytes = total_beats * 16; // DMA transfer size (16 bytes per beat for 128-bit DMA)
    
    // Allocate a temporary buffer for interleaving
    uint8_t* sincos_buf = (uint8_t*)calloc(1, total_bytes);
    if (!sincos_buf) return -1;
    
    for (int i = 0; i < half_dim_beats; i++) {
        // First beat: Cosine
        float* beat1 = (float*)(sincos_buf + (i * 2) * 16);
        for (int j = 0; j < 4; j++) {
            beat1[j] = cos_table[i * 4 + j];
        }
        // Second beat: Sine
        float* beat2 = (float*)(sincos_buf + (i * 2 + 1) * 16);
        for (int j = 0; j < 4; j++) {
            beat2[j] = sin_table[i * 4 + j];
        }
    }
    
    // Prevent race condition by dynamically allocating a 4KB chunk in a 4MB ring buffer
    // since Llama.cpp queues multiple tokens during prompt eval.
    uint32_t base_offset = 0x400000u;
    if (engine->sg_dispatcher != MAP_FAILED) {
        base_offset += (engine->desc_count * 4096) % (4 * 1024 * 1024);
    }
    
    // Copy to CMA memory
    memcpy((uint8_t*)engine->vbase + base_offset, sincos_buf, total_bytes);
    free(sincos_buf);
    
    sync_cma_for_device(engine, base_offset, total_bytes);
    
    // Configure cfg_word to trigger ROPE BRAM loading
    struct v37_compute_cmd cmd = {0};
    uint32_t load_rope_en = 1;
    uint32_t rope_en = 0; // Not enabling normal processing during load
    uint32_t cfg_val = (load_rope_en << 11) | (rope_en << 10);
    // Add half_dim_beats to cfg_word[31:24]
    cfg_val |= (half_dim_beats << 24);
    
    cmd.cfg_val = cfg_val;
    cmd.pe_out_beats = 0; 
    cmd.scale_bits = 0x01000000;
    cmd.do_soft_reset = 1;
    
    cmd.s2mm_active = false;
    cmd.mm2s_active = false; // Activation DMA disabled
    cmd.x_phys = 0;
    cmd.x_len = 0;
    
    // Send via Weight DMA 0
    cmd.w_phys[0] = engine->phys + base_offset;
    cmd.w_len[0] = total_bytes;
    for (int i = 1; i < 4; i++) {
        cmd.w_phys[i] = 0;
        cmd.w_len[i] = 0;
    }
    
    // Execute command
    if (engine->sg_dispatcher != MAP_FAILED) {
        if (engine->desc_count < 65535) {
            v37_sg_desc_t* desc = &engine->desc_pool[engine->desc_count];
            desc->cfg_val = cmd.cfg_val;
            desc->act_in_addr = 0;
            desc->act_out_addr = 0;
            desc->x_length = 0;
            desc->y_length = 0;
            desc->weight_addr_0 = cmd.w_phys[0];
            desc->weight_length = cmd.w_len[0];
            uint32_t one_val = 0x01000000;
            memcpy(&desc->total_scale, &one_val, sizeof(float));
            
            if (engine->desc_count > 0) {
                uint32_t curr_phys = engine->phys + 0x200000u + (engine->desc_count * sizeof(v37_sg_desc_t));
                engine->desc_pool[engine->desc_count - 1].next_desc_ptr = curr_phys;
            }
            desc->next_desc_ptr = 0;
            engine->desc_count++;
        }
    } else if (engine->fd_v37 >= 0) {
        if (ioctl(engine->fd_v37, V37_IOCTL_COMPUTE, &cmd) < 0) {
            perror("V37 ROPE LOAD ioctl failed");
            return -3;
        }
    } else {
        wreg(engine->vcore, 0x00, 2);
        for (volatile int delay_idx = 0; delay_idx < 10; delay_idx++) {}
        wreg(engine->vcore, 0x00, 0);
        
        wreg(engine->sg_dispatcher, 0x30, cmd.w_phys[0]);
        wreg(engine->sg_dispatcher, 0x34, cmd.w_len[0]);
        wreg(engine->vcore, 0x04, cmd.cfg_val);
        
        wreg(engine->vcore, 0x00, 1);
        wreg(engine->sg_dispatcher, 0x00, 1);
        
        while ((rreg(engine->vcore, 0x0C) & 1) == 0) {}
    }
    return 0;
}

extern "C" void* v37_get_cma_slot_ptr(v37_engine_t* engine, int slot_id) {
    if (!engine) return NULL;
    return (uint8_t*)engine->cma_vbase + engine->slot_offsets[slot_id];
}

int v37_compute_ffn_loopback(v37_engine_t* e, const int8_t* x, int gate_slot, int up_slot, int down_slot, float* out, int hidden_dim, int ffn_dim, const float* gate_scales, const float* up_scales, const float* down_scales, float act_scale, const float* sub_norm_weights, float* residual, v37_mode_t mod) {
    if (!e) return -1;

    static std::vector<float> gate_buf(ffn_dim);
    static std::vector<float> ffn_act(ffn_dim);
    static std::vector<float> ffn_sub_norm_out(ffn_dim);
    static std::vector<int8_t> qact_sub_x_q((ffn_dim + 63) & ~63);

    // 1. Launch Gate GEMV on FPGA asynchronously
    v37_launch_linear_async(e, x, gate_slot, ffn_dim, hidden_dim, mod, 0, 0);

    // 2. Wait Gate
    v37_wait_linear_async(e, gate_buf.data(), gate_scales, act_scale, nullptr);

    // 3. Launch Up GEMV on FPGA asynchronously (reuse_act = 1: activation is already in act_sram!)
    v37_launch_linear_async(e, x, up_slot, ffn_dim, hidden_dim, mod, 0, 1);

    // 4. Overlap: while FPGA computes Up (~460us), CPU computes Squared-ReLU on Gate using NEON!
#if defined(__aarch64__) || defined(__arm__)
    float32x4_t zero = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i <= ffn_dim - 16; i += 16) {
        float32x4_t g0 = vmaxq_f32(vld1q_f32(gate_buf.data() + i + 0), zero);
        float32x4_t g1 = vmaxq_f32(vld1q_f32(gate_buf.data() + i + 4), zero);
        float32x4_t g2 = vmaxq_f32(vld1q_f32(gate_buf.data() + i + 8), zero);
        float32x4_t g3 = vmaxq_f32(vld1q_f32(gate_buf.data() + i + 12), zero);

        vst1q_f32(gate_buf.data() + i + 0,  vmulq_f32(g0, g0));
        vst1q_f32(gate_buf.data() + i + 4,  vmulq_f32(g1, g1));
        vst1q_f32(gate_buf.data() + i + 8,  vmulq_f32(g2, g2));
        vst1q_f32(gate_buf.data() + i + 12, vmulq_f32(g3, g3));
    }
    for (; i < ffn_dim; i++) {
        float val = gate_buf[i];
        float r = (val > 0.0f) ? val : 0.0f;
        gate_buf[i] = r * r;
    }
#else
    for (int i = 0; i < ffn_dim; i++) {
        float val = gate_buf[i];
        float r = (val > 0.0f) ? val : 0.0f;
        gate_buf[i] = r * r;
    }
#endif

    // 5. Wait Up & Fused Multiply into ffn_act directly from DMA
    v37_wait_linear_async_fused_up(e, ffn_act.data(), gate_buf.data(), up_scales, act_scale);

    // 6. Sub-Norm (RMSNorm) with NEON vectorization and double accumulator for BitNet golden precision
    if (sub_norm_weights) {
#if defined(__aarch64__) || defined(__arm__)
        float64x2_t sum_d0 = vdupq_n_f64(0.0);
        float64x2_t sum_d1 = vdupq_n_f64(0.0);
        int j = 0;
        for (; j <= ffn_dim - 8; j += 8) {
            float32x4_t v0 = vld1q_f32(ffn_act.data() + j + 0);
            float32x4_t v1 = vld1q_f32(ffn_act.data() + j + 4);
            sum_d0 = vmlaq_f64(sum_d0, vcvt_f64_f32(vget_low_f32(v0)), vcvt_f64_f32(vget_low_f32(v0)));
            sum_d1 = vmlaq_f64(sum_d1, vcvt_f64_f32(vget_high_f32(v0)), vcvt_f64_f32(vget_high_f32(v0)));
            sum_d0 = vmlaq_f64(sum_d0, vcvt_f64_f32(vget_low_f32(v1)), vcvt_f64_f32(vget_low_f32(v1)));
            sum_d1 = vmlaq_f64(sum_d1, vcvt_f64_f32(vget_high_f32(v1)), vcvt_f64_f32(vget_high_f32(v1)));
        }
        double sum = vgetq_lane_f64(sum_d0, 0) + vgetq_lane_f64(sum_d0, 1) +
                     vgetq_lane_f64(sum_d1, 0) + vgetq_lane_f64(sum_d1, 1);
        for (; j < ffn_dim; j++) sum += (double)(ffn_act[j] * ffn_act[j]);
        float rms = 1.0f / std::sqrt((float)(sum / ffn_dim) + 1e-5f);
        float32x4_t vrms = vdupq_n_f32(rms);
        j = 0;
        for (; j <= ffn_dim - 16; j += 16) {
            float32x4_t a0 = vld1q_f32(ffn_act.data() + j + 0);
            float32x4_t a1 = vld1q_f32(ffn_act.data() + j + 4);
            float32x4_t a2 = vld1q_f32(ffn_act.data() + j + 8);
            float32x4_t a3 = vld1q_f32(ffn_act.data() + j + 12);

            float32x4_t w0 = vld1q_f32(sub_norm_weights + j + 0);
            float32x4_t w1 = vld1q_f32(sub_norm_weights + j + 4);
            float32x4_t w2 = vld1q_f32(sub_norm_weights + j + 8);
            float32x4_t w3 = vld1q_f32(sub_norm_weights + j + 12);

            vst1q_f32(ffn_sub_norm_out.data() + j + 0,  vmulq_f32(vmulq_f32(a0, vrms), w0));
            vst1q_f32(ffn_sub_norm_out.data() + j + 4,  vmulq_f32(vmulq_f32(a1, vrms), w1));
            vst1q_f32(ffn_sub_norm_out.data() + j + 8,  vmulq_f32(vmulq_f32(a2, vrms), w2));
            vst1q_f32(ffn_sub_norm_out.data() + j + 12, vmulq_f32(vmulq_f32(a3, vrms), w3));
        }
        for (; j < ffn_dim; j++) ffn_sub_norm_out[j] = ffn_act[j] * rms * sub_norm_weights[j];
#else
        double sum = 0.0;
        for (int j = 0; j < ffn_dim; j++) sum += (double)(ffn_act[j] * ffn_act[j]);
        float rms = 1.0f / std::sqrt((float)(sum / ffn_dim) + 1e-5f);
        for (int j = 0; j < ffn_dim; j++) ffn_sub_norm_out[j] = ffn_act[j] * rms * sub_norm_weights[j];
#endif
    } else {
        memcpy(ffn_sub_norm_out.data(), ffn_act.data(), ffn_dim * sizeof(float));
    }

    // 7. Dynamic Quantization (DynQuant) with NEON SIMD
    float quant_s = 1.0f;
    int8_t* x_q = qact_sub_x_q.data();
#if defined(__aarch64__) || defined(__arm__)
    float32x4_t vm = vdupq_n_f32(0.00001f);
    int k_idx = 0;
    for (; k_idx <= ffn_dim - 16; k_idx += 16) {
        float32x4_t v0 = vabsq_f32(vld1q_f32(ffn_sub_norm_out.data() + k_idx + 0));
        float32x4_t v1 = vabsq_f32(vld1q_f32(ffn_sub_norm_out.data() + k_idx + 4));
        float32x4_t v2 = vabsq_f32(vld1q_f32(ffn_sub_norm_out.data() + k_idx + 8));
        float32x4_t v3 = vabsq_f32(vld1q_f32(ffn_sub_norm_out.data() + k_idx + 12));
        vm = vmaxnmq_f32(vm, vmaxnmq_f32(vmaxnmq_f32(v0, v1), vmaxnmq_f32(v2, v3)));
    }
    float max_val = vmaxvq_f32(vm);
    for (; k_idx < ffn_dim; k_idx++) {
        float a = std::fabs(ffn_sub_norm_out[k_idx]);
        if (a > max_val) max_val = a;
    }
    quant_s = 127.0f / max_val;
    float32x4_t vs = vdupq_n_f32(quant_s);
    k_idx = 0;
    for (; k_idx <= ffn_dim - 16; k_idx += 16) {
        float32x4_t f0 = vmulq_f32(vld1q_f32(ffn_sub_norm_out.data() + k_idx + 0), vs);
        float32x4_t f1 = vmulq_f32(vld1q_f32(ffn_sub_norm_out.data() + k_idx + 4), vs);
        float32x4_t f2 = vmulq_f32(vld1q_f32(ffn_sub_norm_out.data() + k_idx + 8), vs);
        float32x4_t f3 = vmulq_f32(vld1q_f32(ffn_sub_norm_out.data() + k_idx + 12), vs);

        int32x4_t s0 = vcvtnq_s32_f32(f0);
        int32x4_t s1 = vcvtnq_s32_f32(f1);
        int32x4_t s2 = vcvtnq_s32_f32(f2);
        int32x4_t s3 = vcvtnq_s32_f32(f3);

        int16x8_t q01 = vcombine_s16(vqmovn_s32(s0), vqmovn_s32(s1));
        int16x8_t q23 = vcombine_s16(vqmovn_s32(s2), vqmovn_s32(s3));
        int8x16_t q_all = vcombine_s8(vqmovn_s16(q01), vqmovn_s16(q23));

        vst1q_s8(x_q + k_idx, q_all);
    }
    auto nearest_int = [](float fval) -> int {
        float val = fval + 12582912.f;
        int iv; memcpy(&iv, &val, sizeof(int));
        return (iv & 0x007fffff) - 0x00400000;
    };
    for (; k_idx < ffn_dim; k_idx++) {
        int v = nearest_int(ffn_sub_norm_out[k_idx] * quant_s);
        if (v > 127) v = 127;
        if (v < -128) v = -128;
        x_q[k_idx] = (int8_t)v;
    }
#else
    double max_val = 0.00001;
    for (int j = 0; j < ffn_dim; j++) {
        double a = (double)std::fabs(ffn_sub_norm_out[j]);
        if (a > max_val) max_val = a;
    }
    quant_s = 127.0f / (float)max_val;
    auto nearest_int = [](float fval) -> int {
        float val = fval + 12582912.f;
        int iv; memcpy(&iv, &val, sizeof(int));
        return (iv & 0x007fffff) - 0x00400000;
    };
    for (int j = 0; j < ffn_dim; j++) {
        int v = nearest_int(ffn_sub_norm_out[j] * quant_s);
        if (v > 127) v = 127;
        if (v < -128) v = -128;
        x_q[j] = (int8_t)v;
    }
#endif

    // 8. Launch Down GEMV on FPGA
    v37_launch_linear_async(e, qact_sub_x_q.data(), down_slot, hidden_dim, ffn_dim, mod, 0, 0);

    // 9. Wait Down GEMV & Accumulate into residual
    v37_wait_linear_async(e, out, down_scales, quant_s, residual);

    return 0;
}

int v37_preload_fused_gate_up(v37_engine_t* e, int gate_slot, int up_slot, int fused_slot, int ffn_dim, int hidden_dim, v37_mode_t mod) {
    if (!e) return -1;
    if (gate_slot < 0 || gate_slot >= 512 || e->slot_offsets[gate_slot] == 0) return -2;
    if (up_slot < 0 || up_slot >= 512 || e->slot_offsets[up_slot] == 0) return -2;
    if (fused_slot < 0 || fused_slot >= 512) return -2;

    uint32_t kb = (hidden_dim + 63) / 64;
    kb = (kb + 3) & ~3;
    uint32_t pe_k_beats = kb;
    uint32_t m_blocks_half = (ffn_dim + 3) / 4; // 1728
    uint32_t m_blocks_fused = (ffn_dim * 2 + 3) / 4; // 3456

    uint32_t gate_payload_bytes;
    uint32_t up_payload_bytes;
    uint32_t gate_channel_sz;
    uint32_t up_channel_sz;

    uint32_t fused_matrix_beats;
    uint32_t fused_payload_bytes;
    uint32_t fused_channel_sz;

    v37_mode_t real_hw_mode = (v37_mode_t)(mod & 0xFFFF);

    if (real_hw_mode == V37_MODE_TL3_COMPACT) {
        uint32_t ch_stride = pe_k_beats * 13;
        uint32_t half_pld = m_blocks_half * ch_stride;
        uint32_t half_beats = (half_pld + 15) / 16;
        gate_channel_sz = (half_beats * 16 + 16 + 63) & ~63;
        up_channel_sz = gate_channel_sz;
        gate_payload_bytes = half_beats * 16;
        up_payload_bytes = gate_payload_bytes;

        uint32_t total_pld = m_blocks_fused * ch_stride;
        fused_matrix_beats = (total_pld + 15) / 16;
        fused_channel_sz = (fused_matrix_beats * 16 + 16 + 63) & ~63;
        fused_payload_bytes = fused_matrix_beats * 16;
    } else {
        uint32_t total_beats = pe_k_beats;
        if (real_hw_mode != V37_MODE_I2_S && real_hw_mode != V37_MODE_TL3) {
            total_beats = pe_k_beats + (pe_k_beats + 3) / 4;
        }
        uint32_t ch_stride = total_beats * 16;
        gate_payload_bytes = m_blocks_half * ch_stride;
        gate_channel_sz = (gate_payload_bytes + 16 + 63) & ~63;
        up_payload_bytes = gate_payload_bytes;
        up_channel_sz = gate_channel_sz;

        fused_matrix_beats = m_blocks_fused * total_beats;
        fused_payload_bytes = m_blocks_fused * ch_stride;
        fused_channel_sz = (fused_payload_bytes + 16 + 63) & ~63;
    }

    uint32_t alloc_sz = 4 * fused_channel_sz;
    uint32_t cma_offset = e->current_cma_offset;
    if (cma_offset + alloc_sz > e->udma_size) {
        printf("[V37-ERROR] CMA overflow during fused_gate_up! alloc_sz=%u offset=0x%X max=0x%lX\n",
               alloc_sz, cma_offset + alloc_sz, e->udma_size);
        return -1;
    }

    e->slot_offsets[fused_slot] = cma_offset;
    e->current_cma_offset = (e->current_cma_offset + alloc_sz + 4095) & ~4095;

    uint8_t* fused_base = (uint8_t*)e->cma_vbase + cma_offset;
    const uint8_t* gate_base = (const uint8_t*)e->cma_vbase + e->slot_offsets[gate_slot];
    const uint8_t* up_base = (const uint8_t*)e->cma_vbase + e->slot_offsets[up_slot];

    for (int i = 0; i < 4; i++) {
        uint8_t* f_ch = fused_base + i * fused_channel_sz;
        const uint8_t* g_ch = gate_base + i * gate_channel_sz;
        const uint8_t* u_ch = up_base + i * up_channel_sz;

        volatile uint32_t* hdr = (volatile uint32_t*)f_ch;
        if (i == 0) {
            uint32_t cfg_val = (pe_k_beats << 16) | real_hw_mode;
            uint32_t pe_out_reg = (m_blocks_fused & 0xFFFF) | (1 << 25);
            hdr[0] = cfg_val;
            hdr[1] = pe_out_reg;
            hdr[2] = 0x01000000;
            hdr[3] = fused_matrix_beats;
        } else {
            hdr[0] = pe_k_beats;
            hdr[1] = 0xDEADBEEF;
            hdr[2] = 0;
            hdr[3] = 0;
        }

        // Copy Gate payload
        memcpy(f_ch + 16, g_ch + 16, gate_payload_bytes);
        // Copy Up payload immediately after Gate
        memcpy(f_ch + 16 + gate_payload_bytes, u_ch + 16, up_payload_bytes);
    }

    sync_cma_for_device(e, cma_offset, alloc_sz);
    return 0;
}

int v37_preload_fused_gate_up_buffers(v37_engine_t* e, const uint8_t* gate_base, const uint8_t* up_base, int fused_slot, int ffn_dim, int hidden_dim, v37_mode_t mod) {
    if (!e || fused_slot < 0 || fused_slot >= 512 || !gate_base || !up_base) return -2;

    uint32_t k = hidden_dim;
    uint32_t m_half = ffn_dim;
    uint32_t m_fused = ffn_dim * 2;

    uint32_t k_beats = (k + 63) / 64;
    k_beats = (k_beats + 3) & ~3;
    uint32_t pe_k_beats = k_beats;

    uint32_t m_blocks_half = (m_half + 3) / 4;
    uint32_t m_blocks_fused = (m_fused + 3) / 4;

    uint32_t gate_channel_sz, up_channel_sz, fused_channel_sz;
    uint32_t gate_payload_bytes, up_payload_bytes, fused_payload_bytes;
    uint32_t fused_matrix_beats;

    v37_mode_t real_hw_mode = (v37_mode_t)(mod & 0xFFFF);

    if (real_hw_mode == V37_MODE_TL3_COMPACT) {
        uint32_t ch_stride = pe_k_beats * 13;
        uint32_t half_pld = m_blocks_half * ch_stride;
        uint32_t half_beats = (half_pld + 15) / 16;
        gate_channel_sz = (half_beats * 16 + 16 + 63) & ~63;
        up_channel_sz = gate_channel_sz;
        gate_payload_bytes = half_beats * 16;
        up_payload_bytes = gate_payload_bytes;

        uint32_t total_pld = m_blocks_fused * ch_stride;
        fused_matrix_beats = (total_pld + 15) / 16;
        fused_channel_sz = (fused_matrix_beats * 16 + 16 + 63) & ~63;
        fused_payload_bytes = fused_matrix_beats * 16;
    } else {
        uint32_t total_beats = pe_k_beats;
        if (real_hw_mode != V37_MODE_I2_S && real_hw_mode != V37_MODE_TL3) {
            total_beats = pe_k_beats + (pe_k_beats + 3) / 4;
        }
        uint32_t ch_stride = total_beats * 16;
        gate_payload_bytes = m_blocks_half * ch_stride;
        gate_channel_sz = (gate_payload_bytes + 16 + 63) & ~63;
        up_payload_bytes = gate_payload_bytes;
        up_channel_sz = gate_channel_sz;

        fused_matrix_beats = m_blocks_fused * total_beats;
        fused_payload_bytes = m_blocks_fused * ch_stride;
        fused_channel_sz = (fused_payload_bytes + 16 + 63) & ~63;
    }

    uint32_t alloc_sz = 4 * fused_channel_sz;
    uint32_t cma_offset = e->current_cma_offset;
    if (cma_offset + alloc_sz > e->udma_size) {
        printf("[V37-ERROR] CMA overflow during fused_gate_up! alloc_sz=%u offset=0x%X max=0x%lX\n",
               alloc_sz, cma_offset + alloc_sz, e->udma_size);
        return -1;
    }

    e->slot_offsets[fused_slot] = cma_offset;
    e->current_cma_offset = (e->current_cma_offset + alloc_sz + 4095) & ~4095;

    uint8_t* fused_base = (uint8_t*)e->cma_vbase + cma_offset;

    for (int i = 0; i < 4; i++) {
        uint8_t* f_ch = fused_base + i * fused_channel_sz;
        const uint8_t* g_ch = gate_base + i * gate_channel_sz;
        const uint8_t* u_ch = up_base + i * up_channel_sz;

        volatile uint32_t* hdr = (volatile uint32_t*)f_ch;
        if (i == 0) {
            uint32_t cfg_val = (pe_k_beats << 16) | real_hw_mode;
            uint32_t pe_out_reg = (m_blocks_fused & 0xFFFF) | (1 << 25);
            hdr[0] = cfg_val;
            hdr[1] = pe_out_reg;
            hdr[2] = 0x01000000;
            hdr[3] = fused_matrix_beats;
        } else {
            hdr[0] = pe_k_beats;
            hdr[1] = 0xDEADBEEF;
            hdr[2] = 0;
            hdr[3] = 0;
        }

        // Copy Gate payload
        memcpy(f_ch + 16, g_ch + 16, gate_payload_bytes);
        // Copy Up payload immediately after Gate
        memcpy(f_ch + 16 + gate_payload_bytes, u_ch + 16, up_payload_bytes);
    }

    sync_cma_for_device(e, cma_offset, alloc_sz);
    return 0;
}

int v37_preload_fused_qkv_buffers(v37_engine_t* e, const uint8_t* q_base, const uint8_t* k_base, const uint8_t* v_base, int fused_slot, int q_dim, int k_dim, int v_dim, int hidden_dim, v37_mode_t mod) {
    if (!e || fused_slot < 0 || fused_slot >= 512 || !q_base || !k_base || !v_base) return -2;

    uint32_t k = hidden_dim;
    uint32_t m_fused = q_dim + k_dim + v_dim;

    uint32_t k_beats = (k + 63) / 64;
    k_beats = (k_beats + 3) & ~3;
    uint32_t pe_k_beats = k_beats;

    uint32_t m_blocks_q = (q_dim + 3) / 4;
    uint32_t m_blocks_k = (k_dim + 3) / 4;
    uint32_t m_blocks_v = (v_dim + 3) / 4;
    uint32_t m_blocks_fused = (m_fused + 3) / 4;

    v37_mode_t real_hw_mode = (v37_mode_t)(mod & 0xFFFF);
    uint32_t total_beats = pe_k_beats;
    if (real_hw_mode != V37_MODE_I2_S && real_hw_mode != V37_MODE_TL3) {
        total_beats = pe_k_beats + (pe_k_beats + 3) / 4;
    }
    uint32_t ch_stride = total_beats * 16;

    uint32_t q_payload_bytes = m_blocks_q * ch_stride;
    uint32_t q_channel_sz = (q_payload_bytes + 16 + 63) & ~63;

    uint32_t k_payload_bytes = m_blocks_k * ch_stride;
    uint32_t k_channel_sz = (k_payload_bytes + 16 + 63) & ~63;

    uint32_t v_payload_bytes = m_blocks_v * ch_stride;
    uint32_t v_channel_sz = (v_payload_bytes + 16 + 63) & ~63;

    uint32_t fused_matrix_beats = m_blocks_fused * total_beats;
    uint32_t fused_payload_bytes = m_blocks_fused * ch_stride;
    uint32_t fused_channel_sz = (fused_payload_bytes + 16 + 63) & ~63;

    uint32_t alloc_sz = 4 * fused_channel_sz;
    uint32_t cma_offset = e->current_cma_offset;
    if (cma_offset + alloc_sz > e->udma_size) {
        printf("[V37-ERROR] CMA overflow during fused_qkv! alloc_sz=%u offset=0x%X max=0x%lX\n",
               alloc_sz, cma_offset + alloc_sz, e->udma_size);
        return -1;
    }

    e->slot_offsets[fused_slot] = cma_offset;
    e->current_cma_offset = (e->current_cma_offset + alloc_sz + 4095) & ~4095;

    uint8_t* fused_base = (uint8_t*)e->cma_vbase + cma_offset;

    for (int i = 0; i < 4; i++) {
        uint8_t* f_ch = fused_base + i * fused_channel_sz;
        const uint8_t* q_ch = q_base + i * q_channel_sz;
        const uint8_t* k_ch = k_base + i * k_channel_sz;
        const uint8_t* v_ch = v_base + i * v_channel_sz;

        volatile uint32_t* hdr = (volatile uint32_t*)f_ch;
        if (i == 0) {
            uint32_t cfg_val = (pe_k_beats << 16) | real_hw_mode;
            uint32_t pe_out_reg = (m_blocks_fused & 0xFFFF) | (1 << 25);
            hdr[0] = cfg_val;
            hdr[1] = pe_out_reg;
            hdr[2] = 0x01000000;
            hdr[3] = fused_matrix_beats;
        } else {
            hdr[0] = pe_k_beats;
            hdr[1] = 0xDEADBEEF;
            hdr[2] = 0;
            hdr[3] = 0;
        }

        memcpy(f_ch + 16, q_ch + 16, q_payload_bytes);
        memcpy(f_ch + 16 + q_payload_bytes, k_ch + 16, k_payload_bytes);
        memcpy(f_ch + 16 + q_payload_bytes + k_payload_bytes, v_ch + 16, v_payload_bytes);
    }

    sync_cma_for_device(e, cma_offset, alloc_sz);
    return 0;
}

int v37_preload_fused_direct(v37_engine_t* e, const uint8_t* fused_data, size_t size, int fused_slot) {
    if (!e || fused_slot < 0 || fused_slot >= 512 || !fused_data) return -2;

    uint32_t cma_offset = e->current_cma_offset;
    if (cma_offset + size > e->udma_size) {
        printf("[V37-ERROR] CMA overflow during fused_direct! alloc_sz=%zu offset=0x%X max=0x%lX\n",
               size, cma_offset + (uint32_t)size, e->udma_size);
        return -1;
    }

    e->slot_offsets[fused_slot] = cma_offset;
    e->current_cma_offset = (e->current_cma_offset + (uint32_t)size + 4095) & ~4095;

    uint8_t* fused_base = (uint8_t*)e->cma_vbase + cma_offset;
    memcpy(fused_base, fused_data, size);

    sync_cma_for_device(e, cma_offset, (uint32_t)size);
    return 0;
}

int v37_launch_linear_async_gate_up(v37_engine_t* e, const int8_t* x, int fused_slot, int m, int k, v37_mode_t mod) {
    if (!e || fused_slot < 0 || fused_slot >= 512 || e->slot_offsets[fused_slot] == 0) return -2;

    auto start_t = std::chrono::high_resolution_clock::now();
    uint64_t start_vct = arm64_get_cntvct();
    t_calls++;

    uint32_t k_beats = (k + 63) / 64;
    k_beats = (k_beats + 3) & ~3;
    uint32_t pe_k_beats = k_beats;
    uint32_t pulses = (m + 3) / 4;
    uint32_t pe_out_beats = pulses;
    uint32_t y_off = 0x600000u;
    uint32_t s2mm_bytes = pulses * 16;

    v37_mode_t real_hw_mode = (v37_mode_t)(mod & 0xFFFF);
    uint32_t hw_mode_to_prog = real_hw_mode;
    uint32_t cfg_val = (pe_k_beats << 16) | hw_mode_to_prog;

    // Send activation DMA (MM2S)
    uint32_t act_dma_sz = pe_k_beats * 64;
    uint8_t* act_ptr = (uint8_t*)e->cma_vbase + 0x400000u;
    memcpy(act_ptr, x, k);
    if (act_dma_sz > (uint32_t)k) {
        memset(act_ptr + k, 0, act_dma_sz - k);
    }
    cpu_clean_dcache(act_ptr, pe_k_beats * 64);

    uint32_t dma_out_bit = (1 << 25);
    uint32_t pe_out_reg = (pe_out_beats & 0xFFFF) | dma_out_bit;

    if (!e->ucode_mode) {
        wreg(e->vcore, 0x08, cfg_val);
        wreg(e->vcore, 0x0C, pe_out_reg);

        wreg(e->vcore, 0x00, 2);
        for (volatile int delay_idx = 0; delay_idx < 10; delay_idx++) {}
        wreg(e->vcore, 0x00, 0);
    }

    wreg(e->dma_x, 0x34, 0x7000);
    wreg(e->dma_x, 0x48, e->phys + y_off);
    wreg(e->dma_x, 0x58, s2mm_bytes);

    wreg(e->dma_x, 0x04, 0x7000);
    wreg(e->dma_x, 0x18, e->phys + 0x400000u);
    wreg(e->dma_x, 0x28, pe_k_beats * 64);
    // [Stage 1 Non-blocking MM2S]: Concurrent streaming with 4-channel weight DMAs

    uint32_t m_blocks = (m + 3) / 4;
    uint32_t ch_stride;
    uint32_t channel_sz;
    uint32_t current_matrix_beats;
    uint32_t hdr_bytes = e->ucode_mode ? 0 : 16;

    if (real_hw_mode == V37_MODE_TL3_COMPACT) {
        ch_stride = pe_k_beats * 13;
        uint32_t total_payload = m_blocks * ch_stride;
        current_matrix_beats = (total_payload + 15) / 16;
        channel_sz = current_matrix_beats * 16 + hdr_bytes;
        channel_sz = (channel_sz + 63) & ~63;
    } else {
        uint32_t total_beats = pe_k_beats;
        if (real_hw_mode != V37_MODE_I2_S && real_hw_mode != V37_MODE_TL3) {
            total_beats = pe_k_beats + (pe_k_beats + 3) / 4;
        }
        ch_stride = total_beats * 16;
        channel_sz = m_blocks * ch_stride + hdr_bytes;
        channel_sz = (channel_sz + 63) & ~63;
        current_matrix_beats = m_blocks * total_beats;
    }
    uint32_t w_len = current_matrix_beats * 16 + hdr_bytes;

    for (int i = 0; i < 4; i++) {
        uint32_t ch_base = i * channel_sz;
        wreg(e->dma_w[i], 0x18, e->phys + e->slot_offsets[fused_slot] + ch_base);
        wreg(e->dma_w[i], 0x28, w_len);
    }

    auto t_before_poll = std::chrono::high_resolution_clock::now();
    t_setup += std::chrono::duration_cast<std::chrono::microseconds>(t_before_poll - start_t).count();

    e->async_job.m = m;
    e->async_job.k = k;
    e->async_job.s_id = fused_slot;
    e->async_job.stage = 0;
    e->async_job.s2mm_out_beats = pulses;
    e->async_job.y_off = y_off;
    e->async_job.start_t = start_t;
    e->async_job.start_vct = start_vct;
    e->async_job.t_before_poll = t_before_poll;
    e->async_job.active = true;

    return 0;
}

int v37_wait_linear_async_gate_up(v37_engine_t* e, int8_t* qact_out, float* quant_s_out, const float* gate_scales, const float* up_scales, float act_scale, const float* sub_norm_weights, int ffn_dim, int act_type) {
    if (!e || !e->async_job.active) return -1;

    int m = e->async_job.m;
    uint32_t y_off = e->async_job.y_off;
    uint32_t s2mm_out_beats = e->async_job.s2mm_out_beats;
    auto t_before_poll = e->async_job.t_before_poll;
    auto start_t = e->async_job.start_t;

    hybrid_wait_s2mm(e, start_t, m, e->async_job.k, e->async_job.start_vct);

    int stuck_cnt = 0;
    uint32_t final_s2mm_stat = 0;
    while (!((final_s2mm_stat = rreg(e->dma_x, 0x34)) & 0x1002)) {
        cpu_relax();
        if (++stuck_cnt > 10000000) {
            printf("[DEBUG] Hung at S2MM Poll in wait_gate_up! stat=%08X\n", final_s2mm_stat);
            e->async_job.active = false;
            return -3;
        }
    }

    auto t_after_poll = std::chrono::high_resolution_clock::now();
    t_poll += std::chrono::duration_cast<std::chrono::microseconds>(t_after_poll - t_before_poll).count();

    static int cal_gu_cnt = 0;
    if (getenv("V37_BENCH_CALIBRATE") && cal_gu_cnt++ < 10) {
        printf("[CALIBRATE] FusedGateUp m=%d, k=%d -> actual_wait = %ld us\n", m, e->async_job.k,
               (long)std::chrono::duration_cast<std::chrono::microseconds>(t_after_poll - start_t).count());
    }

    uint32_t final_s2mm_bytes = s2mm_out_beats * 16;
    auto t1 = std::chrono::high_resolution_clock::now();
    sync_cma_for_cpu(e, y_off, final_s2mm_bytes);
    auto t_after_sync = std::chrono::high_resolution_clock::now();
    t_sync_cpu += std::chrono::duration_cast<std::chrono::microseconds>(t_after_sync - t1).count();

    const int32_t* raw_gate = (const int32_t*)((const uint8_t*)e->vbase + y_off);
    const int32_t* raw_up   = raw_gate + ffn_dim;

    static std::vector<float> ffn_act(ffn_dim);
    float inv_s = (act_scale > 0.0f) ? (1.0f / act_scale) : 1.0f;
    float quant_s = 1.0f;

    if (act_type == 1) { // SwiGLU (SiLU) for Falcon / LLaMA
        float max_val = 0.00001f;
        #pragma omp parallel for reduction(max:max_val) schedule(static)
        for (int i = 0; i < ffn_dim; i++) {
            float g = ((float)raw_gate[i] * gate_scales[i]) * inv_s;
            float u = ((float)raw_up[i] * up_scales[i]) * inv_s;
            float silu = g / (1.0f + expf(-g));
            float a = silu * u;
            ffn_act[i] = a;
            float aa = std::fabs(a);
            if (aa > max_val) max_val = aa;
        }
        quant_s = 127.0f / max_val;
        *quant_s_out = quant_s;

#if defined(__aarch64__) || defined(__arm__)
        float32x4_t vs = vdupq_n_f32(quant_s);
        int k_idx = 0;
        for (; k_idx <= ffn_dim - 16; k_idx += 16) {
            float32x4_t f0 = vmulq_f32(vld1q_f32(ffn_act.data() + k_idx + 0), vs);
            float32x4_t f1 = vmulq_f32(vld1q_f32(ffn_act.data() + k_idx + 4), vs);
            float32x4_t f2 = vmulq_f32(vld1q_f32(ffn_act.data() + k_idx + 8), vs);
            float32x4_t f3 = vmulq_f32(vld1q_f32(ffn_act.data() + k_idx + 12), vs);

            int32x4_t s0 = vcvtnq_s32_f32(f0);
            int32x4_t s1 = vcvtnq_s32_f32(f1);
            int32x4_t s2 = vcvtnq_s32_f32(f2);
            int32x4_t s3 = vcvtnq_s32_f32(f3);

            int16x8_t q01 = vcombine_s16(vqmovn_s32(s0), vqmovn_s32(s1));
            int16x8_t q23 = vcombine_s16(vqmovn_s32(s2), vqmovn_s32(s3));
            int8x16_t q_all = vcombine_s8(vqmovn_s16(q01), vqmovn_s16(q23));

            vst1q_s8(qact_out + k_idx, q_all);
        }
        for (; k_idx < ffn_dim; k_idx++) {
            int v = (int)std::lround(ffn_act[k_idx] * quant_s);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            qact_out[k_idx] = (int8_t)v;
        }
#else
        for (int k_idx = 0; k_idx < ffn_dim; k_idx++) {
            int v = (int)std::lround(ffn_act[k_idx] * quant_s);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            qact_out[k_idx] = (int8_t)v;
        }
#endif
        cpu_clean_dcache(qact_out, ffn_dim);
        auto t3 = std::chrono::high_resolution_clock::now();
        t_post += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t_after_sync).count();
        t_dma += std::chrono::duration_cast<std::chrono::microseconds>(t3 - start_t).count();
        e->async_job.active = false;
        return 0;
    }

#if defined(__aarch64__) || defined(__arm__)
    float32x4_t vinv_s = vdupq_n_f32(inv_s);
    float32x4_t zero   = vdupq_n_f32(0.0f);

    if (sub_norm_weights) {
        float32x4_t vinv_s = vdupq_n_f32(inv_s);
        float32x4_t zero   = vdupq_n_f32(0.0f);
        double th_sum[4] = {0};
        float th_max[4] = {0};

        #pragma omp parallel num_threads(4)
        {
            int tid = omp_get_thread_num();
            int chunk = ffn_dim / 4;
            int start = tid * chunk;
            int end = (tid == 3) ? ffn_dim : (start + chunk);

            float32x4_t sum_v0 = vdupq_n_f32(0.0f);
            float32x4_t sum_v1 = vdupq_n_f32(0.0f);
            float32x4_t vm_t   = vdupq_n_f32(0.00001f);

            int i = start;
            for (; i <= end - 16; i += 16) {
                float32x4_t g0 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(raw_gate + i + 0)),  vld1q_f32(gate_scales + i + 0)),  vinv_s);
                float32x4_t g1 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(raw_gate + i + 4)),  vld1q_f32(gate_scales + i + 4)),  vinv_s);
                float32x4_t g2 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(raw_gate + i + 8)),  vld1q_f32(gate_scales + i + 8)),  vinv_s);
                float32x4_t g3 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(raw_gate + i + 12)), vld1q_f32(gate_scales + i + 12)), vinv_s);

                float32x4_t rg0 = vmaxq_f32(g0, zero);
                float32x4_t rg1 = vmaxq_f32(g1, zero);
                float32x4_t rg2 = vmaxq_f32(g2, zero);
                float32x4_t rg3 = vmaxq_f32(g3, zero);

                float32x4_t sq0 = vmulq_f32(rg0, rg0);
                float32x4_t sq1 = vmulq_f32(rg1, rg1);
                float32x4_t sq2 = vmulq_f32(rg2, rg2);
                float32x4_t sq3 = vmulq_f32(rg3, rg3);

                float32x4_t u0 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(raw_up + i + 0)),  vld1q_f32(up_scales + i + 0)),  vinv_s);
                float32x4_t u1 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(raw_up + i + 4)),  vld1q_f32(up_scales + i + 4)),  vinv_s);
                float32x4_t u2 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(raw_up + i + 8)),  vld1q_f32(up_scales + i + 8)),  vinv_s);
                float32x4_t u3 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(raw_up + i + 12)), vld1q_f32(up_scales + i + 12)), vinv_s);

                float32x4_t a0 = vmulq_f32(sq0, u0);
                float32x4_t a1 = vmulq_f32(sq1, u1);
                float32x4_t a2 = vmulq_f32(sq2, u2);
                float32x4_t a3 = vmulq_f32(sq3, u3);

                sum_v0 = vmlaq_f32(sum_v0, a0, a0);
                sum_v1 = vmlaq_f32(sum_v1, a1, a1);
                sum_v0 = vmlaq_f32(sum_v0, a2, a2);
                sum_v1 = vmlaq_f32(sum_v1, a3, a3);

                float32x4_t w0 = vld1q_f32(sub_norm_weights + i + 0);
                float32x4_t w1 = vld1q_f32(sub_norm_weights + i + 4);
                float32x4_t w2 = vld1q_f32(sub_norm_weights + i + 8);
                float32x4_t w3 = vld1q_f32(sub_norm_weights + i + 12);

                float32x4_t t0 = vmulq_f32(a0, w0);
                float32x4_t t1 = vmulq_f32(a1, w1);
                float32x4_t t2 = vmulq_f32(a2, w2);
                float32x4_t t3 = vmulq_f32(a3, w3);

                vst1q_f32(ffn_act.data() + i + 0,  t0);
                vst1q_f32(ffn_act.data() + i + 4,  t1);
                vst1q_f32(ffn_act.data() + i + 8,  t2);
                vst1q_f32(ffn_act.data() + i + 12, t3);

                vm_t = vmaxnmq_f32(vm_t, vmaxnmq_f32(vmaxnmq_f32(vabsq_f32(t0), vabsq_f32(t1)),
                                                     vmaxnmq_f32(vabsq_f32(t2), vabsq_f32(t3))));
            }
            double th_s = (double)vaddvq_f32(vaddq_f32(sum_v0, sum_v1));
            float th_m = vmaxvq_f32(vm_t);
            for (; i < end; i++) {
                float g = ((float)raw_gate[i] * gate_scales[i]) * inv_s;
                float rg = (g > 0.0f) ? g : 0.0f;
                float u = ((float)raw_up[i] * up_scales[i]) * inv_s;
                float a = (rg * rg) * u;
                th_s += (double)(a * a);
                float t = a * sub_norm_weights[i];
                ffn_act[i] = t;
                float at = std::fabs(t);
                if (at > th_m) th_m = at;
            }
            th_sum[tid] = th_s;
            th_max[tid] = th_m;

            #pragma omp barrier

            #pragma omp single
            {
                double total_sum = th_sum[0] + th_sum[1] + th_sum[2] + th_sum[3];
                float total_max = std::max(std::max(th_max[0], th_max[1]), std::max(th_max[2], th_max[3]));
                float rms = 1.0f / std::sqrt((float)(total_sum / ffn_dim) + 1e-5f);
                float s_t = 127.0f / total_max;
                quant_s = s_t / rms;
                th_max[0] = s_t;
            }

            float s_t = th_max[0];
            float32x4_t vs_t = vdupq_n_f32(s_t);
            int k_idx = start;
            for (; k_idx <= end - 16; k_idx += 16) {
                float32x4_t f0 = vmulq_f32(vld1q_f32(ffn_act.data() + k_idx + 0), vs_t);
                float32x4_t f1 = vmulq_f32(vld1q_f32(ffn_act.data() + k_idx + 4), vs_t);
                float32x4_t f2 = vmulq_f32(vld1q_f32(ffn_act.data() + k_idx + 8), vs_t);
                float32x4_t f3 = vmulq_f32(vld1q_f32(ffn_act.data() + k_idx + 12), vs_t);

                int32x4_t s0 = vcvtnq_s32_f32(f0);
                int32x4_t s1 = vcvtnq_s32_f32(f1);
                int32x4_t s2 = vcvtnq_s32_f32(f2);
                int32x4_t s3 = vcvtnq_s32_f32(f3);

                int16x8_t q01 = vcombine_s16(vqmovn_s32(s0), vqmovn_s32(s1));
                int16x8_t q23 = vcombine_s16(vqmovn_s32(s2), vqmovn_s32(s3));
                int8x16_t q_all = vcombine_s8(vqmovn_s16(q01), vqmovn_s16(q23));

                vst1q_s8(qact_out + k_idx, q_all);
            }
            for (; k_idx < end; k_idx++) {
                int v = (int)std::lround(ffn_act[k_idx] * s_t);
                if (v > 127) v = 127;
                if (v < -128) v = -128;
                qact_out[k_idx] = (int8_t)v;
            }
            cpu_clean_dcache(qact_out + start, end - start);
        }
    } else {
        float32x4_t vm_t = vdupq_n_f32(0.00001f);
        int i = 0;
        for (; i <= ffn_dim - 16; i += 16) {
            float32x4_t g0 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(raw_gate + i + 0)),  vld1q_f32(gate_scales + i + 0)),  vinv_s);
            float32x4_t g1 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(raw_gate + i + 4)),  vld1q_f32(gate_scales + i + 4)),  vinv_s);
            float32x4_t g2 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(raw_gate + i + 8)),  vld1q_f32(gate_scales + i + 8)),  vinv_s);
            float32x4_t g3 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(raw_gate + i + 12)), vld1q_f32(gate_scales + i + 12)), vinv_s);

            float32x4_t rg0 = vmaxq_f32(g0, zero);
            float32x4_t rg1 = vmaxq_f32(g1, zero);
            float32x4_t rg2 = vmaxq_f32(g2, zero);
            float32x4_t rg3 = vmaxq_f32(g3, zero);

            float32x4_t sq0 = vmulq_f32(rg0, rg0);
            float32x4_t sq1 = vmulq_f32(rg1, rg1);
            float32x4_t sq2 = vmulq_f32(rg2, rg2);
            float32x4_t sq3 = vmulq_f32(rg3, rg3);

            float32x4_t u0 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(raw_up + i + 0)),  vld1q_f32(up_scales + i + 0)),  vinv_s);
            float32x4_t u1 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(raw_up + i + 4)),  vld1q_f32(up_scales + i + 4)),  vinv_s);
            float32x4_t u2 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(raw_up + i + 8)),  vld1q_f32(up_scales + i + 8)),  vinv_s);
            float32x4_t u3 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(raw_up + i + 12)), vld1q_f32(up_scales + i + 12)), vinv_s);

            float32x4_t a0 = vmulq_f32(sq0, u0);
            float32x4_t a1 = vmulq_f32(sq1, u1);
            float32x4_t a2 = vmulq_f32(sq2, u2);
            float32x4_t a3 = vmulq_f32(sq3, u3);

            vst1q_f32(ffn_act.data() + i + 0,  a0);
            vst1q_f32(ffn_act.data() + i + 4,  a1);
            vst1q_f32(ffn_act.data() + i + 8,  a2);
            vst1q_f32(ffn_act.data() + i + 12, a3);

            vm_t = vmaxnmq_f32(vm_t, vmaxnmq_f32(vmaxnmq_f32(vabsq_f32(a0), vabsq_f32(a1)),
                                                 vmaxnmq_f32(vabsq_f32(a2), vabsq_f32(a3))));
        }
        float max_val = vmaxvq_f32(vm_t);
        for (; i < ffn_dim; i++) {
            float g = ((float)raw_gate[i] * gate_scales[i]) * inv_s;
            float rg = (g > 0.0f) ? g : 0.0f;
            float u = ((float)raw_up[i] * up_scales[i]) * inv_s;
            float a = (rg * rg) * u;
            ffn_act[i] = a;
            float aa = std::fabs(a);
            if (aa > max_val) max_val = aa;
        }
        float quant_s = 127.0f / max_val;
        *quant_s_out = quant_s;

        float32x4_t vs = vdupq_n_f32(quant_s);
        int k_idx = 0;
        for (; k_idx <= ffn_dim - 16; k_idx += 16) {
            float32x4_t f0 = vmulq_f32(vld1q_f32(ffn_act.data() + k_idx + 0), vs);
            float32x4_t f1 = vmulq_f32(vld1q_f32(ffn_act.data() + k_idx + 4), vs);
            float32x4_t f2 = vmulq_f32(vld1q_f32(ffn_act.data() + k_idx + 8), vs);
            float32x4_t f3 = vmulq_f32(vld1q_f32(ffn_act.data() + k_idx + 12), vs);

            int32x4_t s0 = vcvtnq_s32_f32(f0);
            int32x4_t s1 = vcvtnq_s32_f32(f1);
            int32x4_t s2 = vcvtnq_s32_f32(f2);
            int32x4_t s3 = vcvtnq_s32_f32(f3);

            int16x8_t q01 = vcombine_s16(vqmovn_s32(s0), vqmovn_s32(s1));
            int16x8_t q23 = vcombine_s16(vqmovn_s32(s2), vqmovn_s32(s3));
            int8x16_t q_all = vcombine_s8(vqmovn_s16(q01), vqmovn_s16(q23));

            vst1q_s8(qact_out + k_idx, q_all);
        }
        for (; k_idx < ffn_dim; k_idx++) {
            int v = (int)std::lround(ffn_act[k_idx] * quant_s);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            qact_out[k_idx] = (int8_t)v;
        }
        cpu_clean_dcache(qact_out, ffn_dim);
    }
#else
    double sum = 0.0;
    for (int i = 0; i < ffn_dim; i++) {
        float g = ((float)raw_gate[i] * gate_scales[i]) * inv_s;
        float rg = (g > 0.0f) ? g : 0.0f;
        float u = ((float)raw_up[i] * up_scales[i]) * inv_s;
        float a = (rg * rg) * u;
        ffn_act[i] = a;
        if (sub_norm_weights) sum += (double)(a * a);
    }
    float max_val = 0.00001f;
    if (sub_norm_weights) {
        float rms = 1.0f / std::sqrt((float)(sum / ffn_dim) + 1e-5f);
        for (int j = 0; j < ffn_dim; j++) {
            float o = ffn_act[j] * rms * sub_norm_weights[j];
            ffn_act[j] = o;
            float a = std::fabs(o);
            if (a > max_val) max_val = a;
        }
    } else {
        for (int j = 0; j < ffn_dim; j++) {
            float a = std::fabs(ffn_act[j]);
            if (a > max_val) max_val = a;
        }
    }
    float quant_s = 127.0f / max_val;
    auto nearest_int = [](float fval) -> int {
        float val = fval + 12582912.f;
        int iv; memcpy(&iv, &val, sizeof(int));
        return (iv & 0x007fffff) - 0x00400000;
    };
    for (int j = 0; j < ffn_dim; j++) {
        int v = nearest_int(ffn_act[j] * quant_s);
        if (v > 127) v = 127;
        if (v < -128) v = -128;
        qact_out[j] = (int8_t)v;
    }
    cpu_clean_dcache(qact_out, ffn_dim);
#endif

    if (quant_s_out) *quant_s_out = quant_s;

    auto t3 = std::chrono::high_resolution_clock::now();
    t_post += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t_after_sync).count();
    t_dma += std::chrono::duration_cast<std::chrono::microseconds>(t3 - start_t).count();

    e->async_job.active = false;
    return 0;
}

static inline int v37_launch_down_direct(v37_engine_t* e, int down_slot, int hidden_dim, int ffn_dim, v37_mode_t mod) {
    if (!e || down_slot < 0 || down_slot >= 512 || e->slot_offsets[down_slot] == 0) return -2;

    auto start_t = std::chrono::high_resolution_clock::now();
    uint64_t start_vct = arm64_get_cntvct();
    t_calls++;

    uint32_t k_beats = (ffn_dim + 63) / 64;
    uint32_t pe_k_beats = (k_beats + 3) & ~3;
    uint32_t pulses = (hidden_dim + 3) / 4;
    uint32_t s2mm_bytes = pulses * 16;
    uint32_t y_off = 0x600000u;

    v37_mode_t real_hw_mode = (v37_mode_t)(mod & 0xFFFF);
    uint32_t cfg_val = (pe_k_beats << 16) | real_hw_mode;
    uint32_t pe_out_reg = (pulses & 0xFFFF) | (1 << 25);

    if (!e->ucode_mode) {
        wreg(e->vcore, 0x08, cfg_val);
        wreg(e->vcore, 0x0C, pe_out_reg);

        wreg(e->vcore, 0x00, 2);
        for (volatile int delay_idx = 0; delay_idx < 10; delay_idx++) {}
        wreg(e->vcore, 0x00, 0);
    }

    wreg(e->dma_x, 0x34, 0x7000);
    wreg(e->dma_x, 0x48, e->phys + y_off);
    wreg(e->dma_x, 0x58, s2mm_bytes);

    wreg(e->dma_x, 0x04, 0x7000);
    wreg(e->dma_x, 0x18, e->phys + 0x400000u);
    wreg(e->dma_x, 0x28, pe_k_beats * 64);

    uint32_t m_blocks = pulses;
    uint32_t ch_stride;
    uint32_t channel_sz;
    uint32_t current_matrix_beats;
    uint32_t hdr_bytes = e->ucode_mode ? 0 : 16;

    if (real_hw_mode == V37_MODE_TL3_COMPACT) {
        ch_stride = pe_k_beats * 13;
        uint32_t total_payload = m_blocks * ch_stride;
        current_matrix_beats = (total_payload + 15) / 16;
        channel_sz = current_matrix_beats * 16 + hdr_bytes;
        channel_sz = (channel_sz + 63) & ~63;
    } else {
        uint32_t total_beats = pe_k_beats;
        if (real_hw_mode != V37_MODE_I2_S && real_hw_mode != V37_MODE_TL3) {
            total_beats = pe_k_beats + (pe_k_beats + 3) / 4;
        }
        ch_stride = total_beats * 16;
        channel_sz = m_blocks * ch_stride + hdr_bytes;
        channel_sz = (channel_sz + 63) & ~63;
        current_matrix_beats = m_blocks * total_beats;
    }
    uint32_t w_len = current_matrix_beats * 16 + hdr_bytes;
    uint32_t w_base = e->phys + e->slot_offsets[down_slot];

    for (int i = 0; i < 4; i++) {
        wreg(e->dma_w[i], 0x18, w_base + i * channel_sz);
        wreg(e->dma_w[i], 0x28, w_len);
    }

    auto t_before_poll = std::chrono::high_resolution_clock::now();
    t_setup += std::chrono::duration_cast<std::chrono::microseconds>(t_before_poll - start_t).count();

    e->async_job.m = hidden_dim;
    e->async_job.k = ffn_dim;
    e->async_job.s_id = down_slot;
    e->async_job.stage = 0;
    e->async_job.s2mm_out_beats = pulses;
    e->async_job.y_off = y_off;
    e->async_job.start_t = start_t;
    e->async_job.start_vct = start_vct;
    e->async_job.t_before_poll = t_before_poll;
    e->async_job.active = true;

    return 0;
}

int v37_launch_ffn_chain(v37_engine_t* e, const int8_t* x, int gate_slot, int up_slot, int down_slot, int fused_gate_up_slot, int hidden_dim, int ffn_dim, const float* gate_scales, const float* up_scales, float act_scale, const float* sub_norm_weights, float* quant_s_out, v37_mode_t mod, int act_type) {
    if (!e) return -1;
    float quant_s = 1.0f;
    int8_t* cma_down_act = (int8_t*)((uint8_t*)e->cma_vbase + 0x400000u);

    // 1. Launch Fused Gate+Up in ONE continuous DMA stream!
    v37_launch_linear_async_gate_up(e, x, fused_gate_up_slot, ffn_dim * 2, hidden_dim, mod);

    // 2. Wait Gate+Up & execute activation, Sub-Norm, DynQuant directly into CMA activation buffer (zero-copy)
    v37_wait_linear_async_gate_up(e, cma_down_act, &quant_s, gate_scales, up_scales, act_scale, sub_norm_weights, ffn_dim, act_type);

    // 3. Launch Down GEMV on FPGA via specialized direct zero-overhead launcher
    v37_launch_down_direct(e, down_slot, hidden_dim, ffn_dim, mod);

    if (quant_s_out) *quant_s_out = quant_s;
    return 0;
}

int v37_wait_ffn_chain_down(v37_engine_t* e, float* out, const float* down_scales, float quant_s, float* residual) {
    return v37_wait_linear_async(e, out, down_scales, quant_s, residual);
}

int v37_compute_ffn_chain(v37_engine_t* e, const int8_t* x, int gate_slot, int up_slot, int down_slot, int fused_gate_up_slot, float* out, int hidden_dim, int ffn_dim, const float* gate_scales, const float* up_scales, const float* down_scales, float act_scale, const float* sub_norm_weights, float* residual, v37_mode_t mod) {
    if (!e) return -1;

    static int chain_mode = -1;
    if (chain_mode == -1) {
        const char* env_chain = getenv("V37_CHAIN_LAUNCH");
        chain_mode = env_chain ? atoi(env_chain) : 1;
    }

    if (chain_mode == 0 || fused_gate_up_slot < 0 || fused_gate_up_slot >= 512 || e->slot_offsets[fused_gate_up_slot] == 0) {
        return v37_compute_ffn_loopback(e, x, gate_slot, up_slot, down_slot, out, hidden_dim, ffn_dim, gate_scales, up_scales, down_scales, act_scale, sub_norm_weights, residual, mod);
    }

    float quant_s = 1.0f;
    v37_launch_ffn_chain(e, x, gate_slot, up_slot, down_slot, fused_gate_up_slot, hidden_dim, ffn_dim, gate_scales, up_scales, act_scale, sub_norm_weights, &quant_s, mod);
    return v37_wait_ffn_chain_down(e, out, down_scales, quant_s, residual);
}

int v37_compute_qkv_fused(v37_engine_t* e, const int8_t* x, int fused_slot, float* q_out, float* k_out, void* v_f16_out, int q_dim, int k_dim, int v_dim, int hidden_dim, const float* q_scales, const float* k_scales, const float* v_scales, float act_scale, int pos, void* k_f16_out, const float* rope_cos, const float* rope_sin, float inv_sqrt_head_dim, v37_mode_t mod) {
    if (!e || fused_slot < 0 || fused_slot >= 512 || e->slot_offsets[fused_slot] == 0) return -2;

    int m_fused = q_dim + k_dim + v_dim;

    // 1. Single DMA launch for Q, K, and V
    v37_launch_linear_async_gate_up(e, x, fused_slot, m_fused, hidden_dim, mod);

    // 2. Wait S2MM completion
    auto start_t = e->async_job.start_t;
    auto t_before_poll = e->async_job.t_before_poll;
    uint32_t y_off = e->async_job.y_off;
    uint32_t s2mm_out_beats = e->async_job.s2mm_out_beats;

    hybrid_wait_s2mm(e, start_t, m_fused, hidden_dim, e->async_job.start_vct);

    int stuck_cnt = 0;
    uint32_t final_s2mm_stat = 0;
    while (!((final_s2mm_stat = rreg(e->dma_x, 0x34)) & 0x1002)) {
        cpu_relax();
        if (++stuck_cnt > 10000000) {
            printf("[DEBUG] Hung at S2MM Poll in compute_qkv_fused! stat=%08X\n", final_s2mm_stat);
            e->async_job.active = false;
            return -3;
        }
    }

    auto t_after_poll = std::chrono::high_resolution_clock::now();
    t_poll += std::chrono::duration_cast<std::chrono::microseconds>(t_after_poll - t_before_poll).count();

    // 3. Cache Invalidate
    uint32_t final_s2mm_bytes = s2mm_out_beats * 16;
    auto t1 = std::chrono::high_resolution_clock::now();
    sync_cma_for_cpu(e, y_off, final_s2mm_bytes);
    auto t_after_sync = std::chrono::high_resolution_clock::now();
    t_sync_cpu += std::chrono::duration_cast<std::chrono::microseconds>(t_after_sync - t1).count();

    // 4. Raw pointers
    const int32_t* src_q = (const int32_t*)((const uint8_t*)e->vbase + y_off);
    const int32_t* src_k = src_q + q_dim;
    const int32_t* src_v = src_k + k_dim;

    float inv_s = (act_scale > 0.0f) ? (1.0f / act_scale) : 1.0f;

#if defined(__aarch64__) || defined(__arm__)
    float32x4_t vinv_s = vdupq_n_f32(inv_s);
    float32x4_t vinv = (inv_sqrt_head_dim > 0.0f) ? vdupq_n_f32(inv_sqrt_head_dim) : vdupq_n_f32(1.0f);
    bool use_rope = (rope_cos != nullptr && rope_sin != nullptr);
    const int head_dim = (inv_sqrt_head_dim > 0.0f) ? (int)std::round(1.0f / (inv_sqrt_head_dim * inv_sqrt_head_dim)) : 128;
    const int half_dim = head_dim / 2;
    int n_heads = q_dim / head_dim;
    int n_kv_heads = k_dim / head_dim;

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (q_out && q_scales) {
                int h_start = 0;
                int h_end = n_heads / 2;
                if (use_rope && inv_sqrt_head_dim > 0.0f) {
                    for (int h = h_start; h < h_end; h++) {
                        float* q_head = q_out + h * head_dim;
                        const int32_t* src_head = src_q + h * head_dim;
                        const float* sc_head = q_scales + h * head_dim;
                        for (int i = 0; i < half_dim; i += 4) {
                            float32x4_t v0 = vcvtq_f32_s32(vld1q_s32(src_head + i + 0));
                            float32x4_t v1 = vcvtq_f32_s32(vld1q_s32(src_head + i + half_dim));
                            float32x4_t sc0 = vmulq_f32(vld1q_f32(sc_head + i + 0), vinv_s);
                            float32x4_t sc1 = vmulq_f32(vld1q_f32(sc_head + i + half_dim), vinv_s);
                            float32x4_t q0 = vmulq_f32(v0, sc0);
                            float32x4_t q1 = vmulq_f32(v1, sc1);

                            float32x4_t cr = vld1q_f32(rope_cos + i);
                            float32x4_t ci = vld1q_f32(rope_sin + i);
                            float32x4_t out0 = vmulq_f32(vsubq_f32(vmulq_f32(q0, cr), vmulq_f32(q1, ci)), vinv);
                            float32x4_t out1 = vmulq_f32(vaddq_f32(vmulq_f32(q0, ci), vmulq_f32(q1, cr)), vinv);
                            vst1q_f32(q_head + i, out0);
                            vst1q_f32(q_head + i + half_dim, out1);
                        }
                    }
                } else {
                    int i = h_start * head_dim;
                    int i_end = h_end * head_dim;
                    for (; i <= i_end - 16; i += 16) {
                        float32x4_t v0 = vcvtq_f32_s32(vld1q_s32(src_q + i + 0));
                        float32x4_t v1 = vcvtq_f32_s32(vld1q_s32(src_q + i + 4));
                        float32x4_t v2 = vcvtq_f32_s32(vld1q_s32(src_q + i + 8));
                        float32x4_t v3 = vcvtq_f32_s32(vld1q_s32(src_q + i + 12));

                        float32x4_t sc0 = vmulq_f32(vld1q_f32(q_scales + i + 0), vinv_s);
                        float32x4_t sc1 = vmulq_f32(vld1q_f32(q_scales + i + 4), vinv_s);
                        float32x4_t sc2 = vmulq_f32(vld1q_f32(q_scales + i + 8), vinv_s);
                        float32x4_t sc3 = vmulq_f32(vld1q_f32(q_scales + i + 12), vinv_s);

                        vst1q_f32(q_out + i + 0,  vmulq_f32(v0, sc0));
                        vst1q_f32(q_out + i + 4,  vmulq_f32(v1, sc1));
                        vst1q_f32(q_out + i + 8,  vmulq_f32(v2, sc2));
                        vst1q_f32(q_out + i + 12, vmulq_f32(v3, sc3));
                    }
                }
            }
        }
        #pragma omp section
        {
            if (q_out && q_scales) {
                int h_start = n_heads / 2;
                int h_end = n_heads;
                if (use_rope && inv_sqrt_head_dim > 0.0f) {
                    for (int h = h_start; h < h_end; h++) {
                        float* q_head = q_out + h * head_dim;
                        const int32_t* src_head = src_q + h * head_dim;
                        const float* sc_head = q_scales + h * head_dim;
                        for (int i = 0; i < half_dim; i += 4) {
                            float32x4_t v0 = vcvtq_f32_s32(vld1q_s32(src_head + i + 0));
                            float32x4_t v1 = vcvtq_f32_s32(vld1q_s32(src_head + i + half_dim));
                            float32x4_t sc0 = vmulq_f32(vld1q_f32(sc_head + i + 0), vinv_s);
                            float32x4_t sc1 = vmulq_f32(vld1q_f32(sc_head + i + half_dim), vinv_s);
                            float32x4_t q0 = vmulq_f32(v0, sc0);
                            float32x4_t q1 = vmulq_f32(v1, sc1);

                            float32x4_t cr = vld1q_f32(rope_cos + i);
                            float32x4_t ci = vld1q_f32(rope_sin + i);
                            float32x4_t out0 = vmulq_f32(vsubq_f32(vmulq_f32(q0, cr), vmulq_f32(q1, ci)), vinv);
                            float32x4_t out1 = vmulq_f32(vaddq_f32(vmulq_f32(q0, ci), vmulq_f32(q1, cr)), vinv);
                            vst1q_f32(q_head + i, out0);
                            vst1q_f32(q_head + i + half_dim, out1);
                        }
                    }
                } else {
                    int i = h_start * head_dim;
                    int i_end = h_end * head_dim;
                    for (; i <= i_end - 16; i += 16) {
                        float32x4_t v0 = vcvtq_f32_s32(vld1q_s32(src_q + i + 0));
                        float32x4_t v1 = vcvtq_f32_s32(vld1q_s32(src_q + i + 4));
                        float32x4_t v2 = vcvtq_f32_s32(vld1q_s32(src_q + i + 8));
                        float32x4_t v3 = vcvtq_f32_s32(vld1q_s32(src_q + i + 12));

                        float32x4_t sc0 = vmulq_f32(vld1q_f32(q_scales + i + 0), vinv_s);
                        float32x4_t sc1 = vmulq_f32(vld1q_f32(q_scales + i + 4), vinv_s);
                        float32x4_t sc2 = vmulq_f32(vld1q_f32(q_scales + i + 8), vinv_s);
                        float32x4_t sc3 = vmulq_f32(vld1q_f32(q_scales + i + 12), vinv_s);

                        vst1q_f32(q_out + i + 0,  vmulq_f32(v0, sc0));
                        vst1q_f32(q_out + i + 4,  vmulq_f32(v1, sc1));
                        vst1q_f32(q_out + i + 8,  vmulq_f32(v2, sc2));
                        vst1q_f32(q_out + i + 12, vmulq_f32(v3, sc3));
                    }
                    for (; i < q_dim; i++) {
                        q_out[i] = ((float)src_q[i] * q_scales[i]) * inv_s;
                    }
                }
            }
        }
        #pragma omp section
        {
            if (k_scales) {
                if (k_f16_out && use_rope) {
                    __fp16* dst_f16 = (__fp16*)k_f16_out;
                    for (int h = 0; h < n_kv_heads; h++) {
                        __fp16* dst_head = dst_f16 + h * head_dim;
                        const int32_t* src_head = src_k + h * head_dim;
                        const float* sc_head = k_scales + h * head_dim;
                        for (int i = 0; i < half_dim; i += 4) {
                            float32x4_t v0 = vcvtq_f32_s32(vld1q_s32(src_head + i + 0));
                            float32x4_t v1 = vcvtq_f32_s32(vld1q_s32(src_head + i + half_dim));
                            float32x4_t sc0 = vmulq_f32(vld1q_f32(sc_head + i + 0), vinv_s);
                            float32x4_t sc1 = vmulq_f32(vld1q_f32(sc_head + i + half_dim), vinv_s);
                            float32x4_t k0 = vmulq_f32(v0, sc0);
                            float32x4_t k1 = vmulq_f32(v1, sc1);

                            float32x4_t cr = vld1q_f32(rope_cos + i);
                            float32x4_t ci = vld1q_f32(rope_sin + i);
                            float32x4_t out0 = vsubq_f32(vmulq_f32(k0, cr), vmulq_f32(k1, ci));
                            float32x4_t out1 = vaddq_f32(vmulq_f32(k0, ci), vmulq_f32(k1, cr));
                            vst1_f16(dst_head + i, vcvt_f16_f32(out0));
                            vst1_f16(dst_head + i + half_dim, vcvt_f16_f32(out1));
                        }
                    }
                } else if (k_out) {
                    int i = 0;
                    for (; i <= k_dim - 16; i += 16) {
                        float32x4_t v0 = vcvtq_f32_s32(vld1q_s32(src_k + i + 0));
                        float32x4_t v1 = vcvtq_f32_s32(vld1q_s32(src_k + i + 4));
                        float32x4_t v2 = vcvtq_f32_s32(vld1q_s32(src_k + i + 8));
                        float32x4_t v3 = vcvtq_f32_s32(vld1q_s32(src_k + i + 12));

                        float32x4_t sc0 = vmulq_f32(vld1q_f32(k_scales + i + 0), vinv_s);
                        float32x4_t sc1 = vmulq_f32(vld1q_f32(k_scales + i + 4), vinv_s);
                        float32x4_t sc2 = vmulq_f32(vld1q_f32(k_scales + i + 8), vinv_s);
                        float32x4_t sc3 = vmulq_f32(vld1q_f32(k_scales + i + 12), vinv_s);

                        vst1q_f32(k_out + i + 0,  vmulq_f32(v0, sc0));
                        vst1q_f32(k_out + i + 4,  vmulq_f32(v1, sc1));
                        vst1q_f32(k_out + i + 8,  vmulq_f32(v2, sc2));
                        vst1q_f32(k_out + i + 12, vmulq_f32(v3, sc3));
                    }
                    for (; i < k_dim; i++) {
                        k_out[i] = ((float)src_k[i] * k_scales[i]) * inv_s;
                    }
                }
            }
        }
        #pragma omp section
        {
            if (v_f16_out && v_scales) {
                __fp16* dst_f16 = (__fp16*)v_f16_out;
                int i = 0;
                for (; i <= v_dim - 16; i += 16) {
                    float32x4_t v0 = vcvtq_f32_s32(vld1q_s32(src_v + i + 0));
                    float32x4_t v1 = vcvtq_f32_s32(vld1q_s32(src_v + i + 4));
                    float32x4_t v2 = vcvtq_f32_s32(vld1q_s32(src_v + i + 8));
                    float32x4_t v3 = vcvtq_f32_s32(vld1q_s32(src_v + i + 12));

                    float32x4_t sc0 = vmulq_f32(vld1q_f32(v_scales + i + 0), vinv_s);
                    float32x4_t sc1 = vmulq_f32(vld1q_f32(v_scales + i + 4), vinv_s);
                    float32x4_t sc2 = vmulq_f32(vld1q_f32(v_scales + i + 8), vinv_s);
                    float32x4_t sc3 = vmulq_f32(vld1q_f32(v_scales + i + 12), vinv_s);

                    float32x4_t f0 = vmulq_f32(v0, sc0);
                    float32x4_t f1 = vmulq_f32(v1, sc1);
                    float32x4_t f2 = vmulq_f32(v2, sc2);
                    float32x4_t f3 = vmulq_f32(v3, sc3);

                    float16x8_t h0 = vcombine_f16(vcvt_f16_f32(f0), vcvt_f16_f32(f1));
                    float16x8_t h1 = vcombine_f16(vcvt_f16_f32(f2), vcvt_f16_f32(f3));

                    vst1q_f16(dst_f16 + i + 0, h0);
                    vst1q_f16(dst_f16 + i + 8, h1);
                }
                for (; i < v_dim; i++) {
                    float vf = ((float)src_v[i] * v_scales[i]) * inv_s;
                    dst_f16[i] = (__fp16)vf;
                }
            }
        }
    }
#else
    bool use_rope = (rope_cos != nullptr && rope_sin != nullptr);
    const int head_dim = (inv_sqrt_head_dim > 0.0f) ? (int)std::round(1.0f / (inv_sqrt_head_dim * inv_sqrt_head_dim)) : 128;
    const int half_dim = head_dim / 2;
    int n_heads = q_dim / head_dim;
    int n_kv_heads = k_dim / head_dim;
    if (q_out && q_scales) {
        if (use_rope && inv_sqrt_head_dim > 0.0f) {
            for (int h = 0; h < n_heads; h++) {
                float* q_head = q_out + h * head_dim;
                const int32_t* src_head = src_q + h * head_dim;
                const float* sc_head = q_scales + h * head_dim;
                for (int i = 0; i < half_dim; i++) {
                    float q0 = ((float)src_head[i] * sc_head[i]) * inv_s;
                    float q1 = ((float)src_head[i + half_dim] * sc_head[i + half_dim]) * inv_s;
                    float cr = rope_cos[i];
                    float ci = rope_sin[i];
                    q_head[i] = (q0 * cr - q1 * ci) * inv_sqrt_head_dim;
                    q_head[i + half_dim] = (q0 * ci + q1 * cr) * inv_sqrt_head_dim;
                }
            }
        } else {
            for (int i = 0; i < q_dim; i++) {
                q_out[i] = ((float)src_q[i] * q_scales[i]) * inv_s;
            }
        }
    }
    if (k_scales) {
        if (k_f16_out && use_rope) {
            uint16_t* dst_u16 = (uint16_t*)k_f16_out;
            for (int h = 0; h < n_kv_heads; h++) {
                uint16_t* dst_head = dst_u16 + h * head_dim;
                const int32_t* src_head = src_k + h * head_dim;
                const float* sc_head = k_scales + h * head_dim;
                for (int i = 0; i < half_dim; i++) {
                    float k0 = ((float)src_head[i] * sc_head[i]) * inv_s;
                    float k1 = ((float)src_head[i + half_dim] * sc_head[i + half_dim]) * inv_s;
                    float cr = rope_cos[i];
                    float ci = rope_sin[i];
                    float out0 = k0 * cr - k1 * ci;
                    float out1 = k0 * ci + k1 * cr;
                    dst_head[i] = ggml_compute_fp32_to_fp16(out0);
                    dst_head[i + half_dim] = ggml_compute_fp32_to_fp16(out1);
                }
            }
        } else if (k_out) {
            for (int i = 0; i < k_dim; i++) {
                k_out[i] = ((float)src_k[i] * k_scales[i]) * inv_s;
            }
        }
    }
    if (v_f16_out && v_scales) {
        uint16_t* dst_u16 = (uint16_t*)v_f16_out;
        for (int i = 0; i < v_dim; i++) {
            float vf = ((float)src_v[i] * v_scales[i]) * inv_s;
            dst_u16[i] = ggml_compute_fp32_to_fp16(vf);
        }
    }
#endif

    auto t3 = std::chrono::high_resolution_clock::now();
    t_post += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t_after_sync).count();
    t_dma += std::chrono::duration_cast<std::chrono::microseconds>(t3 - start_t).count();

    e->async_job.active = false;
    return 0;
}

