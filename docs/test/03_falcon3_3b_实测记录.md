# Falcon3-3B KV260 板端硬件推理测试记录

> **测试平台**：Xilinx Kria KV260 Starter Kit (Zynq UltraScale+ MPSoC `xck26`)  
> **PL 核心主频**：250.00 MHz  
> **实测吞吐**：**15.50 TPS (稳态解码 64.52 ms/token)**  
> **端到端吞吐**：**11.16 TPS**  
> **核心突破**：**模型体积精炼至 2.0GB，首次突破 KV260 板端 2.2GB 磁盘容量瓶颈并跑通全硬件加速！**

---

## 1. 模型文件与 MD5 完整性校验

| 类别 | 相对/绝对路径 | 文件大小 | MD5 校验码 | 说明 |
| :--- | :--- | :---: | :---: | :--- |
| **原始 GGUF** | `/mnt/nvme_linux/V37_fpga_upload/model/falcon3-3b-instruct/falcon3-3b-instruct-i2_s.gguf` | 2,221,465,632 字节 (2118.55 MB) | `d145f2eda29287551fc94dae236f8140` | 原始模型 |
| **编译后容器** | `/mnt/nvme_linux/V37_fpga_upload/V37_software/models/falcon3-3b-unified.v37p` | 2,107,954,112 字节 (2010.30 MB) | `1b17b364e12670b1ea7febcc1b49cbb0` | Format v4 统一自包含容器 (Mode 6 紧凑流) |

---

## 2. 离线编译打包命令

```bash
cd /mnt/nvme_linux/V37_fpga_upload/V37_compiler
./bin/build_v37p -f 4 -m tl3_compact -l x86 \
    /mnt/nvme_linux/V37_fpga_upload/model/falcon3-3b-instruct/falcon3-3b-instruct-i2_s.gguf \
    /mnt/nvme_linux/V37_fpga_upload/V37_software/models/falcon3-3b-unified.v37p
```

---

## 3. KV260 板端执行测试命令

```bash
cd /home/ubuntu/V37_deploy

sudo sysctl -w kernel.sched_rt_runtime_us=-1
sudo chrt -f 99 taskset -c 0-3 ./bin/semantic_engine \
    ./models/falcon3-3b-unified.v37p \
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
[V37-LOADER] V37P container loaded (289 tensors, embedded GGUF vocab at offset 0x5f00, size 4631 KB)
[V37-LOADER] 自定义物理扫描器完成。发现 289 个张量。
[MODEL-CONFIG] Dynamic Parameters:
  - n_layers   : 22
  - n_embd     : 3072
  - head_dim   : 256
  - n_heads    : 12 (Q)
  - n_kv_heads : 4 (KV, GQA ratio: 3)
  - n_ffn      : 9216
  - vocab_size : 131072
  - max_seq_len: 4096
  - rope_base  : 1000042.0
  - rms_eps    : 1.00e-06
  - sub_norm   : NO
  - activation : SwiGLU (SiLU)
[MAIN] Mode: Self-contained V37P unified container detected.
[MAIN] Successfully initialized tokenizer from embedded V37P vocab (4631 KB).
[V37 WARNING] Failed to open /dev/v37, will use fallback MMIO if possible.
[V37] Auto-detected PL0 clock: 250.00 MHz (div0=6, spin_scale=0.400)
[V37] Engine Initialized with Dynamic ModelConfig.
[MODEL-CONFIG] Dynamic Parameters:
  - n_layers   : 22
  - n_embd     : 3072
  - head_dim   : 256
  - n_heads    : 12 (Q)
  - n_kv_heads : 4 (KV, GQA ratio: 3)
  - n_ffn      : 9216
  - vocab_size : 131072
  - max_seq_len: 4096
  - rope_base  : 1000042.0
  - rms_eps    : 1.00e-06
  - sub_norm   : NO
  - activation : SwiGLU (SiLU)
[V37-UCODE] Loaded 88 microcode instructions into FPGA BRAM (status=0xA5000058)
[V37PACK] Loaded 289 pre-packed tensors from /home/ubuntu/V37_deploy/models/falcon3-3b-unified.v37p (2010.3 MB, Format v4, Mode: TL3_COMPACT (Mode 6, 5-in-8 Gearbox Stream))
[V37-OPT] Starting ONE-TIME full model weight preloading into CMA (Zero-Copy Mode)...
[V37-OPT] Successfully preloaded all 154 tensors (469.22 MB) into CMA in 21798.26 ms!
[V37-OPT] Zero-Copy CMA stream active: subsequent token generation will have ZERO memcpy overhead.
[V37-OPT] mlock SUCCESS: Pinned tables and weights in physical RAM!
[V37-OPT] Hierarchical LM_HEAD quantized tables (Tier1-Fine=28160, Coarse=131072, TwoStage=1 [S1=512, S2=384, K=512, TopCand=256]) built in 36769.6 ms.
[V37-OPT] Pre-cached all 22 layers tensor metadata, weights, and norms into flat memory.
[MAIN] Using pre-packed weights from /home/ubuntu/V37_deploy/models/falcon3-3b-unified.v37p

[V37 FPGA] BitNet Inference Engine
PROMPT: The hardware accelerator is (add_bos: NO)
TOKEN_COUNT: 4
  token[0] = 2530
  token[1] = 10472
  token[2] = 49722
  token[3] = 2340
MODE: FPGA Accelerated
THREADS: 4
N_PREDICT: 10
REPEAT_PENALTY: 1.10 (last_n=64)
---
OUTPUT:  a 10.5 times faster than the

==== Performance Report ====
Mode: V37 FPGA
Prefill:  3 tokens in 251.05 ms (11.95 TPS)
Decode:   10 tokens in 0.645 sec (64.52 ms/token)
Speed:    15.50 TPS (Pure Decode), 11.16 TPS (End-to-End)
Forward:  avg 64.51 ms/token (15.50 TPS)
============================
FPGA TIMING: dma=621 ms (setup=0 ms, poll=500 ms), memcpy=0 ms, sync_cpu=1 ms, post=116 ms. calls=1144
```

---

## 5. 关键指标深度解析

1. **突破性进展**：此前 Falcon3-3B 原始容器高达 3.28GB，远超 KV260 板端剩余磁盘容量（约 2.2GB），无法下发板端。通过 Format v4 剔除死矩阵与去重后，容器体积降至 **2.01 GB**，顺利传输并部署；
2. **硬件算力验证**：在 22 层、隐藏维度 3072 的规模下，跑出 **15.50 TPS** 的高稳态吞吐；
3. **CMA 容量契合**：纯权重占用 **469.22 MB**，严格落入系统的 650MB CMA 空间内，未引发任何溢出。
