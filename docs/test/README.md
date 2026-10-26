# V37 FPGA (KV260) 全模型推理测试记录与基准报告

> **目标硬件**：AMD Xilinx Kria KV260 Vision AI Starter Kit (Zynq UltraScale+ MPSoC `xck26`)  
> **PL 核心主频**：`core_clk = 250.00 MHz`（时序收敛，WNS > 0）  
> **CMA 共享物理内存**：`681,574,400 字节` (650 MB via `u-dma-buf`)  
> **CPU 运行环境**：ARM Cortex-A53 (4核 @ 1.33GHz, `taskset -c 0-3`, FIFO 实时调度)  
> **编译与推理引擎**：[`V37_compiler`](file:///mnt/nvme_linux/V37_fpga_upload/V37_compiler) (`v37c` / `build_v37p`) + [`V37_software`](file:///mnt/nvme_linux/V37_fpga_upload/V37_software) (`v37_engine` / `semantic_engine`)  
> **文档归档目录**：[`V37_software/docs/test/`](file:///mnt/nvme_linux/V37_fpga_upload/V37_software/docs/test)

---

## 1. 测试环境与基准配置

所有板端推理测试均在同一台 Xilinx Kria KV260 评估套件上执行，测试前通过标准化脚本初始化硬件：

```bash
# 1. 挂载 650MB CMA 物理大内存驱动
sudo insmod u-dma-buf.ko udmabuf0=681574400

# 2. 烧录 250MHz 物理位流
sudo fpgautil -b v37_bitnet_fused.bit.bin

# 3. 解除 PMU AXI FPD 隔离门
sudo devmem2 0xFF5E0230 w 0x00000000

# 4. 解除 Linux 实时调度限制并绑定 4 核
sudo sysctl -w kernel.sched_rt_runtime_us=-1
```

---

## 2. 全模型实测性能全景总表

| 模型名称        | 模型架构 | 隐藏层/FFN 维度 | 层数 | CMA 权重占用  | Prefill 吞吐  | Decode 稳态吞吐          | 端到端吞吐 | 实测生成质量 (Prompt: "The hardware accelerator is")        |
|:--------------- |:--------:|:---------------:|:----:|:-------------:|:-------------:|:------------------------:|:----------:|:----------------------------------------------------------- |
| **BitNet-0.6B** | BitNet   | 1536 / 4096     | 16   | **85.31 MB**  | **40.84 TPS** | **48.22 TPS** (20.74 ms) | 32.75 TPS  | 完整响应，无异常，0 NaN                                     |
| **Falcon3-1B**  | Falcon3  | 2048 / 8192     | 18   | **219.38 MB** | **19.54 TPS** | **30.14 TPS** (33.18 ms) | 20.60 TPS  | `"a device that processes data in the form of information"` |
| **BitNet-2B**   | BitNet   | 2560 / 6912     | 30   | **403.78 MB** | **15.82 TPS** | **22.50 TPS** (44.43 ms) | 16.95 TPS  | “a device that performs specific tasks, such as“            |
| **Falcon3-3B**  | Falcon3  | 2560 / 7680     | 22   | **469.22 MB** | **11.95 TPS** | **15.50 TPS** (64.52 ms) | 11.16 TPS  | `"a 10.5 times faster than the"`                            |

> **超大模型说明 (7B ~ 10B)**：  
> Falcon3-7B (2.76GB)、Falcon3-10B (3.30GB) 与 Llama3-8B (3.28GB) 经编译器精炼去重后，已在 x86 仿真环境中实现 **0 NaN、100% 精度对齐验证**。因 KV260 片上物理内存总计 4GB（系统预留 650MB CMA），7B/10B 推荐部署于下一代大容量板卡或通过分页机制运行。

---

## 3. 模型文件清单与 MD5 完整性校验表

所有用于板端实测的原始 GGUF 模型与编译器导出的 Format v4 / v3 优化容器文件均经过严格的 MD5 校验：

### 3.1 原始开源模型 (Raw GGUF)

| 模型名称        | 文件名                                  | 文件绝对物理路径                                                                                     | 文件大小 (Bytes) | 大小 (MB)  | MD5 完整性校验值                   |
|:--------------- |:--------------------------------------- |:---------------------------------------------------------------------------------------------------- |:----------------:|:----------:|:---------------------------------- |
| **BitNet-270M** | `bitnet-embeddings-270m-bf16-i2_s.gguf` | `/mnt/nvme_linux/V37_fpga_upload/model/bitnet-embeddings-270m/bitnet-embeddings-270m-bf16-i2_s.gguf` | 367,487,040      | 350.46 MB  | `30fd1c7dc8dcbcb68067d9817ec38bad` |
| **BitNet-0.6B** | `bitnet-embeddings-0.6b-bf16-i2_s.gguf` | `/mnt/nvme_linux/V37_fpga_upload/model/bitnet-embeddings-0.6b/bitnet-embeddings-0.6b-bf16-i2_s.gguf` | 427,935,008      | 408.11 MB  | `f6e9290b5fe12b34dca13e825f784cfb` |
| **BitNet-2B**   | `bitnet-b1.58-2b-i2_s.gguf`             | `/mnt/nvme_linux/V37_fpga_upload/model/bitnet-b1.58-2b-4t/bitnet-b1.58-2b-i2_s.gguf`                 | 1,187,801,280    | 1132.78 MB | `65cb04366e4d02ccd78b4b7b48c84b3b` |
| **Falcon3-1B**  | `falcon3-1b-instruct-i2_s.gguf`         | `/mnt/nvme_linux/V37_fpga_upload/model/falcon3-1b-instruct/falcon3-1b-instruct-i2_s.gguf`            | 1,361,904,672    | 1298.81 MB | `a47dd88ba7040ce3b711477f5be16688` |
| **Falcon3-3B**  | `falcon3-3b-instruct-i2_s.gguf`         | `/mnt/nvme_linux/V37_fpga_upload/model/falcon3-3b-instruct/falcon3-3b-instruct-i2_s.gguf`            | 2,221,465,632    | 2118.55 MB | `d145f2eda29287551fc94dae236f8140` |
| **Falcon3-7B**  | `falcon3-7b-instruct-i2_s.gguf`         | `/mnt/nvme_linux/V37_fpga_upload/model/falcon3-7b-instruct/falcon3-7b-instruct-i2_s.gguf`            | 3,278,680,768    | 3126.79 MB | `cddb7af9e776416a6dbf70403db90a25` |
| **Falcon3-10B** | `ggml-model-i2_s.gguf`                  | `/mnt/nvme_linux/V37_fpga_upload/model/falcon3-10b-instruct-1.58bit/ggml-model-i2_s.gguf`            | 3,991,393,696    | 3806.49 MB | `abbc33f32ad96ff57590d3fc726688c3` |
| **Llama3-8B**   | `llama3-8b-1.58b-i2_s.gguf`             | `/mnt/nvme_linux/V37_fpga_upload/model/llama3-8b-1.58b/llama3-8b-1.58b-i2_s.gguf`                    | 3,855,084,832    | 3676.50 MB | `50bf54a86d0aaa4745fb125d6652e150` |

---

### 3.2 编译后硬件容器模型 (Compiler Optimized `.v37p`)

| 模型名称         | 文件名                        | 格式版本 / 压缩模式 | 文件大小 (Bytes) | 大小 (MB)  | MD5 完整性校验值                   |
|:---------------- |:----------------------------- |:-------------------:|:----------------:|:----------:|:---------------------------------- |
| **BitNet-270M**  | `bitnet-270m-unified.v37p`    | Format v4 / Mode 6  | 365,462,656      | 348.53 MB  | `6a4d443401f396980e84314716f43a1f` |
| **BitNet-0.6B**  | `bitnet-0.6b-unified.v37p`    | Format v4 / Mode 6  | 407,348,800      | 388.48 MB  | `35d9660b8247c50e4b9185f44e5f40cc` |
| **BitNet-2B**    | `bitnet-2b-unified.v37p`      | Format v4 / Mode 6  | 1,090,156,544    | 1039.65 MB | `3827d44451f7edb831545bf4148b6415` |
| **Falcon3-1B**   | `falcon3-1b-unified.v37p`     | Format v4 / Mode 6  | 1,308,844,800    | 1248.21 MB | `32b9f7ba8c07a49f23ad06bee5d7e1e8` |
| **Falcon3-3B**   | `falcon3-3b-unified.v37p`     | Format v4 / Mode 6  | 2,107,954,112    | 2010.30 MB | `1b17b364e12670b1ea7febcc1b49cbb0` |
| **Falcon3-7B**   | `falcon3-7b-unified.v37p`     | Format v4 / Mode 6  | 2,966,996,288    | 2829.55 MB | `d1cb9e5f0d4734be42987a7e58b5511a` |
| **Falcon3-10B**  | `falcon3-10b-unified.v37p`    | Format v4 / Mode 6  | 3,546,130,048    | 3381.85 MB | `8045cea3749c78ecc57ca9eedf25ab70` |
| **Llama3-8B**    | `llama3-8b-unified.v37p`      | Format v4 / Mode 6  | 3,527,971,968    | 3364.54 MB | `3cc9ca33d85100a256351c657bb236b6` |
| *BitNet-2B (v3)* | `ggml-model-tl3-compact.v37p` | Format v3 / Mode 6  | 423,392,180      | 403.78 MB  | `87ec33fe70b7f5d8c3d2cfbdfa08b07d` |
| *BitNet-2B (v3)* | `ggml-model-i2_s.v37p`        | Format v3 / Mode 5  | 521,081,772      | 496.94 MB  | `418a58e3d7d371960b2efad0f0b65cb9` |

---

## 4. 单项测试详细记录导航

各模型的完整终端测试输出、DMA 计数时序拆解与逐项验证报告见子文档：

- 📄 [**01_BitNet-0.6B KV260 实测记录 (48.22 TPS)**](file:///mnt/nvme_linux/V37_fpga_upload/V37_software/docs/test/01_bitnet_0.6b_实测记录.md)
- 📄 [**02_Falcon3-1B KV260 实测记录 (30.14 TPS)**](file:///mnt/nvme_linux/V37_fpga_upload/V37_software/docs/test/02_falcon3_1b_实测记录.md)
- 📄 [**03_Falcon3-3B KV260 实测记录 (15.50 TPS)**](file:///mnt/nvme_linux/V37_fpga_upload/V37_software/docs/test/03_falcon3_3b_实测记录.md)
- 📄 [**04_BitNet-2B 黄金基线实测记录 (22.50 TPS)**](file:///mnt/nvme_linux/V37_fpga_upload/V37_software/docs/test/04_bitnet_2b_实测记录.md)

---

## 5. 标准板端执行命令说明

`./bin/semantic_engine` 支持直接传入模型文件或使用 `-m / --model` 指定：

```bash
# 模式 A：Format v4 统一自包含单文件启动 (推荐)
sudo chrt -f 99 taskset -c 0-3 ./bin/semantic_engine \
    /home/ubuntu/V37_deploy/models/falcon3-1b-unified.v37p \
    -p "The hardware accelerator is" \
    -n 10 -t 4 -r 1.10

# 模式 B：Format v3 遗留容器 (向后兼容：需 -w 外挂权重)
sudo chrt -f 99 taskset -c 0-3 ./bin/semantic_engine \
    /home/ubuntu/V37_deploy/models/ggml-model-i2_s.gguf \
    -w /home/ubuntu/V37_deploy/models/ggml-model-i2_s.v37p \
    -p "The hardware accelerator is" \
    -n 30 -t 4 -r 1.10
```
