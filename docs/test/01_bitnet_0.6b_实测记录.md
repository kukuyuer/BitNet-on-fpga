# BitNet-0.6B KV260 板端硬件推理测试记录

> **测试平台**：Xilinx Kria KV260 Starter Kit (Zynq UltraScale+ MPSoC `xck26`)  
> **PL 核心主频**：250.00 MHz  
> **实测吞吐**：**48.22 TPS (稳态解码 20.74 ms/token)**  
> **端到端吞吐**：**32.75 TPS**  
> **测试状态**： 输出！！！！！，待确认原始模型格式

---

## 1. 模型文件与 MD5 完整性校验

| 类别           | 相对/绝对路径                                                                                        | 文件大小                     | MD5 校验码                         | 说明                                     |
|:-------------- |:---------------------------------------------------------------------------------------------------- |:----------------------------:|:----------------------------------:|:---------------------------------------- |
| **原始 GGUF**  | `/mnt/nvme_linux/V37_fpga_upload/model/bitnet-embeddings-0.6b/bitnet-embeddings-0.6b-bf16-i2_s.gguf` | 427,935,008 字节 (408.11 MB) | `f6e9290b5fe12b34dca13e825f784cfb` | 原始三值嵌入模型                         |
| **编译后容器** | `/mnt/nvme_linux/V37_fpga_upload/V37_software/models/bitnet-0.6b-unified.v37p`                       | 407,348,800 字节 (388.48 MB) | `35d9660b8247c50e4b9185f44e5f40cc` | Format v4 统一自包含容器 (Mode 6 紧凑流) |

---

## 2. 离线编译打包命令

使用 [`V37_compiler`](file:///mnt/nvme_linux/V37_fpga_upload/V37_compiler) 将原始 GGUF 模型离线编译为 Format v4 容器：

```bash
cd /mnt/nvme_linux/V37_fpga_upload/V37_compiler
./bin/build_v37p -f 4 -m tl3_compact -l x86 \
    /mnt/nvme_linux/V37_fpga_upload/model/bitnet-embeddings-0.6b/bitnet-embeddings-0.6b-bf16-i2_s.gguf \
    /mnt/nvme_linux/V37_fpga_upload/V37_software/models/bitnet-0.6b-unified.v37p
```

---

## 3. KV260 板端执行测试命令

```bash
# 1. 硬件初始化 (已加载 bitstream 与 650MB CMA)
cd /home/ubuntu/V37_deploy

# 2. 启动 FPGA 硬件加速自回归推理
sudo sysctl -w kernel.sched_rt_runtime_us=-1
sudo chrt -f 99 taskset -c 0-3 ./bin/semantic_engine \
    ./models/bitnet-0.6b-unified.v37p \
    -p "The hardware accelerator is" \
    -n 10 -t 4 -r 1.10
```

---

## 4. 板端实测终端完整日志

```text
================================================================================
[MAIN] Initializing llama backend...
[MAIN] Registering V37 FPGA Backend...
[MAIN] V37 Backend registered. I2_S MUL_MAT will be handled by V37.
[V37-LOADER] V37P container loaded (618 tensors, embedded GGUF vocab at offset 0xc400, size 5819 KB)
[V37-LOADER] 自定义物理扫描器完成。发现 618 个张量。
[MODEL-CONFIG] Dynamic Parameters:
  - n_layers   : 28
  - n_embd     : 1024
  - head_dim   : 128
  - n_heads    : 16 (Q)
  - n_kv_heads : 8 (KV, GQA ratio: 2)
  - n_ffn      : 3072
  - vocab_size : 151936
  - max_seq_len: 4096
  - rope_base  : 1000000.0
  - rms_eps    : 1.00e-06
  - sub_norm   : NO
  - activation : SwiGLU (SiLU)
[MAIN] Mode: Self-contained V37P unified container detected.
[MAIN] Successfully initialized tokenizer from embedded V37P vocab (5819 KB).
[V37 WARNING] Failed to open /dev/v37, will use fallback MMIO if possible.
[V37] Auto-detected PL0 clock: 250.00 MHz (div0=6, spin_scale=0.400)
[V37] Engine Initialized with Dynamic ModelConfig.
[MODEL-CONFIG] Dynamic Parameters:
  - n_layers   : 28
  - n_embd     : 1024
  - head_dim   : 128
  - n_heads    : 16 (Q)
  - n_kv_heads : 8 (KV, GQA ratio: 2)
  - n_ffn      : 3072
  - vocab_size : 151936
  - max_seq_len: 4096
  - rope_base  : 1000000.0
  - rms_eps    : 1.00e-06
  - sub_norm   : NO
  - activation : SwiGLU (SiLU)
[V37-UCODE] Loaded 112 microcode instructions into FPGA BRAM (status=0xA5000070)
[V37PACK] Loaded 618 pre-packed tensors from /home/ubuntu/V37_deploy/models/bitnet-0.6b-unified.v37p (388.5 MB, Format v4, Mode: TL3_COMPACT (Mode 6, 5-in-8 Gearbox Stream))
[V37-OPT] Starting ONE-TIME full model weight preloading into CMA (Zero-Copy Mode)...
[V37-OPT] Successfully preloaded all 196 tensors (85.31 MB) into CMA in 130.19 ms!
[V37-OPT] Zero-Copy CMA stream active: subsequent token generation will have ZERO memcpy overhead.
[V37-OPT] mlock SUCCESS: Pinned tables and weights in physical RAM!
[V37-OPT] Hierarchical LM_HEAD quantized tables (Tier1-Fine=28160, Coarse=151936, TwoStage=1 [S1=512, S2=384, K=512, TopCand=256]) built in 700.6 ms.
[V37-OPT] Pre-cached all 28 layers tensor metadata, weights, and norms into flat memory.
[MAIN] Using pre-packed weights from /home/ubuntu/V37_deploy/models/bitnet-0.6b-unified.v37p

[V37 FPGA] BitNet Inference Engine
PROMPT: The hardware accelerator is (add_bos: YES)
TOKEN_COUNT: 5
  token[0] = 785
  token[1] = 11773
  token[2] = 64356
  token[3] = 374
  token[4] = 151643
MODE: FPGA Accelerated
THREADS: 4
N_PREDICT: 10
REPEAT_PENALTY: 1.10 (last_n=64)
---
OUTPUT: !!!!!!!!!!

==== Performance Report ====
Mode: V37 FPGA
Prefill:  4 tokens in 97.94 ms (40.84 TPS)
Decode:   10 tokens in 0.207 sec (20.74 ms/token)
Speed:    48.22 TPS (Pure Decode), 32.75 TPS (End-to-End)
Forward:  avg 20.73 ms/token (48.25 TPS)
============================
FPGA TIMING: dma=205 ms (setup=0 ms, poll=157 ms), memcpy=0 ms, sync_cpu=0 ms, post=43 ms. calls=1568
```

---

## 待确定原始模型文件格式
