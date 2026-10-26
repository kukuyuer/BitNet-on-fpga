# V37 三值大模型软件推理栈与驱动(V37_software)

[![Target Platform](https://img.shields.io/badge/Platform-Xilinx%20Kria%20KV260-blue.svg)](https://www.xilinx.com/products/som/kria/kv260-vision-ai-starter-kit.html)
[![Core Clock](https://img.shields.io/badge/PL%20Clock-250%20MHz%20(Timing%20Met)-success.svg)](#)
[![Decoding Throughput](https://img.shields.io/badge/Max%20Throughput-48.22%20TPS-brightgreen.svg)](#)
[![Supported Models](https://img.shields.io/badge/Models-BitNet%20%7C%20Falcon3%20%7C%20Llama3%20(270M~10B)-purple.svg)](#)
[![Format Spec](https://img.shields.io/badge/Container-Format%20v4%20(Unified%20Self--Contained)-orange.svg)](#)

本项目是专为 **三值量化（1.58-bit / Ternary weights）大语言模型**（涵盖 BitNet b1.58 系列、Falcon3 三值全系列、Llama3-1.58b 等）量身定制的软硬件协同推理引擎与跨平台驱动。目前仅支持**Xilinx Kria KV260 FPGA 硬件加速** 。

在 Xilinx Kria KV260 板端硬件实测中，解码能效实测如下：

* **BitNet-0.6B**： **32.75 TPS**；
* **Falcon3-1B**： **20.60 TPS**；
* **BitNet-2B**： **22.50 TPS** ；
* **Falcon3-3B**：**15.50 TPS**；

---

---

## 📂 规范工程目录树

```text
V37_software/
├── CMakeLists.txt              # 统一顶层 CMake 构建入口 (支持 $ORIGIN 跨路径分发)
├── build.sh                    # 一键快速构建脚本 (cpuref / emulator / kv260 / clean)
├── README.md                   # 本系统门户手册
├── common/                     # 跨平台共享头文件与公共数据定义
│   └── include/
│       ├── v37_driver.h        # 硬件驱动抽象层接口 (HAL)
│       ├── model_config.h      # 动态模型配置结构体 (ModelConfig)
│       └── ggml-bitnet.h       # BitNet 三值量化常数与类型定义
├── engine/                     # 跨平台通用推理引擎源码
│   ├── include/
│   │   ├── engine.h            # 自回归解码器、两级剪枝 LM_HEAD、KV Cache 声明
│   │   └── loader.h            # V37P (Format v4/v3) 与 GGUF 流式内存映射加载器
│   └── src/
│       ├── main.cpp            # 主控程序 CLI (支持直接传参或 -m, -p, -n, -t, -r)
│       ├── engine.cpp          # 核心执行引擎 (含硬件流解码、Fast RMSNorm 与 Squared ReLU)
│       ├── loader.cpp          # 权重与内嵌词表动态解析加载器
│       └── attention.cpp       # FlashAttention + RoPE 旋转位置编码
├── platforms/                  # 平台特化驱动实现 (HAL 具体实现)
│   ├── kv260_hardware/         # KV260 板端真实 FPGA 加速器驱动 (MMIO / 650MB CMA / DMA)
│   └── cpu_reference/          # 纯 CPU 黄金参考驱动 (直通原生 CPU 高精度计算)
├── third_party/                # 第三方预编译库与头文件 (arm64 & x86_64)
├── tests/                      # 自动化基准测试与黄金对齐日志 (golden_30tokens.log)
└── docs/                       # 系统架构、接口规范、部署与实测报告
    └── test/                   # KV260 板端全模型实测记录与 MD5 校验
        ├── README.md           # 板端实测总览与全模型 MD5 校验表
        ├── 01_bitnet_0.6b_实测记录.md
        ├── 02_falcon3_1b_实测记录.md
        ├── 03_falcon3_3b_实测记录.md
        └── 04_bitnet_2b_实测记录.md
```

---

## 上手与推理测试指南 (Quick Start)

### 1. 场景 A：x86 PC 精度参考（无需 FPGA 板卡）

在普通 Linux PC/服务器上进行算法验证或精度检查，推荐构建 CPU 参考版：

```bash
# 1. 编译 CPU 黄金参考版
./build.sh cpuref

# 2. 运行推理 (推荐：单文件自包含 Format v4 模型直接启动)
./bin/semantic_engine ./models/falcon3-1b-unified.v37p \
    -p "The hardware accelerator is" -n 30 -t 4

# 3. 运行经典 BitNet-2B 黄金基线验证 (30 Tokens 逐字对齐)
./bin/semantic_engine ./models/bitnet-2b-unified.v37p \
    -p "The hardware accelerator is" -n 30 -t 4 -r 1.10
```

---

### 2. 场景 B：模型文件准备与编译器 (`V37_compiler`)

推理引擎优先使用由专用离线编译器 [`V37_compiler`](file:///mnt/nvme_linux/V37_fpga_upload/V37_compiler) 生成的 **Format v4 统一自包含容器 (`.v37p`)**。该容器内嵌词表与微码，体积体积略小于原始模型：

```bash
# 切换至独立编译器仓库构建工具:
cd ../V37_compiler && ./build.sh

# 一键离线编译 GGUF 为 Format v4 自包含硬件容器:
./bin/build_v37p -f 4 -m tl3_compact -l x86 ./models/model.gguf ./models/model.v37p
```

---

### 3. 场景 C：KV260 板端硬件加速部署与全模型推理实测

#### 步骤 1：板端编译

在 KV260 开发板终端中执行：

```bash
./build.sh kv260
```

#### 步骤 2：加载 PL 固件与驱动

```bash
cd platforms/kv260_hardware/scripts
sudo bash init_fpga.sh
```

该脚本全自动完成：

1. 动态挂载 650MB CMA 物理连续内存驱动：`sudo insmod u-dma-buf.ko udmabuf0=681574400`；
2. 烧录 250MHz 比特流到 PL：`sudo fpgautil -b v37_bitnet_fused.bit.bin`；
3. 打开 PMU AXI 隔离门：解除 PS-PL DMA 总线锁定。

#### 步骤 3：硬件状态自检

```bash
sudo ./bin/test_dma
```

* **正常就绪输出**：`Compute finished. ret = 0`，代表 MMIO 寄存器映射、CMA 内存通路与 PL 阵列均处于就绪状态。

#### 步骤 4：运行高吞吐硬件加速推理

```bash
# 1. 允许实时进程独占 CPU (消灭微调度抖动)
sudo sysctl -w kernel.sched_rt_runtime_us=-1

# 2. 运行 BitNet-0.6B (48.22 TPS)
sudo chrt -f 99 taskset -c 0-3 ./bin/semantic_engine \
    ./models/bitnet-0.6b-unified.v37p \
    -p "The hardware accelerator is" -n 20 -t 4

# 3. 运行 Falcon3-1B (30.14 TPS)
sudo chrt -f 99 taskset -c 0-3 ./bin/semantic_engine \
    ./models/falcon3-1b-unified.v37p \
    -p "The hardware accelerator is" -n 20 -t 4

# 4. 运行 BitNet-2B  (22.50 TPS)
sudo chrt -f 99 taskset -c 0-3 ./bin/semantic_engine \
    ./models/bitnet-2b-unified.v37p \
    -p "The hardware accelerator is" -n 30 -t 4 -r 1.10

# 5. 运行 Falcon3-3B ( 15.50 TPS)
sudo chrt -f 99 taskset -c 0-3 ./bin/semantic_engine \
    ./models/falcon3-3b-unified.v37p \
    -p "The hardware accelerator is" -n 10 -t 4
```

> **遗留 Format v3 双文件模式运行示例**（向后兼容）：  
> 若模型未包含内嵌词表，需使用 `-w` 显式外挂权重：  
> `sudo chrt -f 99 taskset -c 0-3 ./bin/semantic_engine ./models/model.gguf -w ./models/model.v37p -p "Hello" -n 30`

---

## 📋 命令行参数完整速查表

`./bin/semantic_engine` 支持灵活的位置参数与选项标志：

| 参数选项              | 全称标志           | 默认值              | 参数类型 | 功能详细说明                                                        |
|:--------------------- |:------------------ |:-------------------:|:--------:|:------------------------------------------------------------------- |
| **`[model_file]`**    | `-m, --model`      | **必填**            | 文件路径 | **模型路径**：优先传入 Format v4 自包含 `.v37p` 容器（亦支持 GGUF） |
| **`-p`**              | `--prompt`         | `"The hardware..."` | 字符串   | 输入提示词文本 (Prompt)                                             |
| **`-n`**              | `--predict`        | `32`                | 整数     | 最大生成的 Token 数量                                               |
| **`-t`**              | `--threads`        | `4`                 | 整数     | CPU 执行线程数（KV260 板端推荐设为 `4`）                            |
| **`-r`**              | `--repeat-penalty` | `1.10`              | 浮点数   | 重复惩罚系数（避免自回归文本死循环）                                |
| **`--repeat-last-n`** | -                  | `64`                | 整数     | 重复惩罚窗口大小（Token 历史视窗）                                  |
| **`-w`**              | `--weight`         | 空                  | 文件路径 | 显式指定 `.v37p` 权重文件（仅用于 Format v3 双文件兼容模式）        |
| **`-h`**              | `--help`           | -                   | -        | 输出完整的命令行参数帮助菜单                                        |

### 环境变量开关

* `DISABLE_V37=1`：禁用 FPGA 硬件加速，强制路由至纯 CPU 黄金参考逻辑（用于板端与 CPU 的比对与排错）。

---

## 🔍 KV260 硬件就绪状态诊断手册

| 检查项                 | 命令                                                              | 正常状态返回值                                     | 异常说明与排查                                                                       |
|:---------------------- |:----------------------------------------------------------------- |:-------------------------------------------------- |:------------------------------------------------------------------------------------ |
| **① PL 编程状态**      | `cat /sys/class/fpga_manager/fpga0/state`                         | `operating`                                        | 若显示 `unknown` 说明位流未成功烧入                                                  |
| **② CMA 共享内存**     | `ls -l /dev/udmabuf0`<br>`cat /sys/class/u-dma-buf/udmabuf0/size` | 设备存在且容量约为预留大小（`681574400` 即 650MB） | 若不存在需执行 `sudo insmod u-dma-buf.ko udmabuf0=681574400`                         |
| **③ PMU AXI 隔离门**   | `sudo devmem2 0xFF5E0230`                                         | `0x00000000`                                       | 若非 0 说明 PS-PL 总线处于锁定隔离态，需向该地址写入 0 解锁                          |
| **④ V37 控制器状态**   | `sudo devmem2 0xA0000000`                                         | **`0xA5xxxxxx`**                                   | **核心就绪指标**。高 8 位应为 `0xA5` 魔数。若报 Bus Error 说明 PL 端无时钟或总线挂死 |
| **⑤ DMA 控制器就绪态** | `sudo devmem2 0xA0010000`                                         | `0x00010003`                                       | 表示 MM2S/S2MM DMA 引擎已就绪且处于空闲待命状态                                      |

---

## 测试记录

请参阅 ：

- [ **KV260 板端全模型实测记录与 MD5 校验总表**](docs/test/README.md)
