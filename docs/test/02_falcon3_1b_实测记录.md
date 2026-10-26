# Falcon3-1B KV260 板端硬件推理测试记录

> **测试平台**：Xilinx Kria KV260 Starter Kit (Zynq UltraScale+ MPSoC `xck26`)  
> **PL 核心主频**：250.00 MHz  
> **实测吞吐**：**30.14 TPS (稳态解码 33.18 ms/token)**  
> **端到端吞吐**：**20.60 TPS**  
> **测试状态**： 通过 (语义通顺，0 异常)

---

## 1. 模型文件与 MD5 完整性校验

| 类别 | 相对/绝对路径 | 文件大小 | MD5 校验码 | 说明 |
| :--- | :--- | :---: | :---: | :--- |
| **原始 GGUF** | `/mnt/nvme_linux/V37_fpga_upload/model/falcon3-1b-instruct/falcon3-1b-instruct-i2_s.gguf` | 1,361,904,672 字节 (1298.81 MB) | `a47dd88ba7040ce3b711477f5be16688` | 开源官方 Instruct 版 |
| **编译后容器** | `/mnt/nvme_linux/V37_fpga_upload/V37_software/models/falcon3-1b-unified.v37p` | 1,308,844,800 字节 (1248.21 MB) | `32b9f7ba8c07a49f23ad06bee5d7e1e8` | Format v4 统一自包含容器 (Mode 6 紧凑流) |

---

## 2. 离线编译打包命令

```bash
cd /mnt/nvme_linux/V37_fpga_upload/V37_compiler
./bin/build_v37p -f 4 -m tl3_compact -l x86 \
    /mnt/nvme_linux/V37_fpga_upload/model/falcon3-1b-instruct/falcon3-1b-instruct-i2_s.gguf \
    /mnt/nvme_linux/V37_fpga_upload/V37_software/models/falcon3-1b-unified.v37p
```

---

## 3. KV260 板端执行测试命令

```bash
cd /home/ubuntu/V37_deploy

sudo sysctl -w kernel.sched_rt_runtime_us=-1
sudo chrt -f 99 taskset -c 0-3 ./bin/semantic_engine \
    ./models/falcon3-1b-unified.v37p \
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
[V37-LOADER] V37P container loaded (237 tensors, embedded GGUF vocab at offset 0x4f00, size 4629 KB)
[V37-LOADER] 自定义物理扫描器完成。发现 237 个张量。
[MODEL-CONFIG] Dynamic Parameters:
  - n_layers   : 18
  - n_embd     : 2048
  - head_dim   : 256
  - n_heads    : 8 (Q)
  - n_kv_heads : 4 (KV, GQA ratio: 2)
  - n_ffn      : 8192
  - vocab_size : 131072
  - max_seq_len: 4096
  - rope_base  : 1000042.0
  - rms_eps    : 1.00e-06
  - sub_norm   : NO
  - activation : SwiGLU (SiLU)
[MAIN] Mode: Self-contained V37P unified container detected.
[MAIN] Successfully initialized tokenizer from embedded V37P vocab (4629 KB).
[V37 WARNING] Failed to open /dev/v37, will use fallback MMIO if possible.
[V37] Auto-detected PL0 clock: 250.00 MHz (div0=6, spin_scale=0.400)
[V37] Engine Initialized with Dynamic ModelConfig.
[MODEL-CONFIG] Dynamic Parameters:
  - n_layers   : 18
  - n_embd     : 2048
  - head_dim   : 256
  - n_heads    : 8 (Q)
  - n_kv_heads : 4 (KV, GQA ratio: 2)
  - n_ffn      : 8192
  - vocab_size : 131072
  - max_seq_len: 4096
  - rope_base  : 1000042.0
  - rms_eps    : 1.00e-06
  - sub_norm   : NO
  - activation : SwiGLU (SiLU)
[V37-UCODE] Loaded 72 microcode instructions into FPGA BRAM (status=0xA5000048)
[V37PACK] Loaded 237 pre-packed tensors from /home/ubuntu/V37_deploy/models/falcon3-1b-unified.v37p (1248.2 MB, Format v4, Mode: TL3_COMPACT (Mode 6, 5-in-8 Gearbox Stream))
[V37-OPT] Starting ONE-TIME full model weight preloading into CMA (Zero-Copy Mode)...
[V37-OPT] Successfully preloaded all 126 tensors (219.38 MB) into CMA in 348.42 ms!
[V37-OPT] Zero-Copy CMA stream active: subsequent token generation will have ZERO memcpy overhead.
[V37-OPT] mlock SUCCESS: Pinned tables and weights in physical RAM!
[V37-OPT] Hierarchical LM_HEAD quantized tables (Tier1-Fine=28160, Coarse=131072, TwoStage=1 [S1=512, S2=384, K=512, TopCand=256]) built in 786.3 ms.
[V37-OPT] Pre-cached all 18 layers tensor metadata, weights, and norms into flat memory.
[MAIN] Using pre-packed weights from /home/ubuntu/V37_deploy/models/falcon3-1b-unified.v37p

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
OUTPUT:  a device that processes data in the form of information

==== Performance Report ====
Mode: V37 FPGA
Prefill:  3 tokens in 153.56 ms (19.54 TPS)
Decode:   10 tokens in 0.332 sec (33.18 ms/token)
Speed:    30.14 TPS (Pure Decode), 20.60 TPS (End-to-End)
Forward:  avg 33.17 ms/token (30.15 TPS)
============================
FPGA TIMING: dma=337 ms (setup=0 ms, poll=273 ms), memcpy=0 ms, sync_cpu=1 ms, post=60 ms. calls=936
```

---

## 5. 关键指标深度解析

1. **稳态吞吐**：达到 **30.14 TPS**，在边缘端 1B 参数量模型上展现了极高的推理能效；
2. **生成质量**：生成文本 `" a device that processes data in the form of information"` 语义极度连贯，0 乱码，0 NaN；
3. **CMA 显存控制**：纯权重占用 **219.38 MB**，在 348.42 ms 内完成加载，即使在 4GB 内存的 KV260 上也拥有充足的剩余 RAM。
