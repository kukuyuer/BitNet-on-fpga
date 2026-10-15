#!/bin/bash
# ==============================================================================
# V37 BitNet FPGA 初始化脚本 
# 用法: sudo bash init_fpga.sh
# ==============================================================================

set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# 优先在当前目录查找固件与驱动，若无则回退至相对路径
if [ -f "$DIR/v37_bitnet_fused.bit.bin" ]; then
    BITSTREAM="$DIR/v37_bitnet_fused.bit.bin"
elif [ -f "$DIR/../firmware/v37_bitnet_fused.bit.bin" ]; then
    BITSTREAM="$DIR/../firmware/v37_bitnet_fused.bit.bin"
else
    BITSTREAM="$DIR/v37_bitnet_fused.bit.bin"
fi

if [ -f "$DIR/u-dma-buf.ko" ]; then
    UDMABUF_KO="$DIR/u-dma-buf.ko"
elif [ -f "$DIR/../firmware/u-dma-buf.ko" ]; then
    UDMABUF_KO="$DIR/../firmware/u-dma-buf.ko"
else
    UDMABUF_KO="$DIR/u-dma-buf.ko"
fi

UDMABUF_SIZE=681574400  # 650MB

echo "=================================================="
echo "    [STEP 1] 挂载 650MB CMA 物理内存底座 (udmabuf)"
echo "=================================================="
if [ -d "/sys/class/u-dma-buf/udmabuf0" ]; then
    echo "⚠️  udmabuf0 已加载，跳过。"
else
    sudo rmmod u_dma_buf 2>/dev/null || true
    sudo insmod "$UDMABUF_KO" udmabuf0=$UDMABUF_SIZE
    if [ $? -ne 0 ]; then
        echo "❌ [ERROR] 无法加载 u-dma-buf.ko，尝试更小的 512MB..."
        sudo insmod "$UDMABUF_KO" udmabuf0=536870912
        if [ $? -ne 0 ]; then
            echo "❌ [ERROR] u-dma-buf 加载失败。请检查 CMA 预留。"
            exit 1
        fi
    fi
fi
RAW_ADDR=$(cat /sys/class/u-dma-buf/udmabuf0/phys_addr)
SIZE_RAW=$(cat /sys/class/u-dma-buf/udmabuf0/size)
SIZE_MB=$(( SIZE_RAW / 1024 / 1024 ))
echo "✅ 物理内存分配成功，基地址: $RAW_ADDR，容量: ${SIZE_MB}MB"

echo ""
echo "=================================================="
echo "    [STEP 2] 用 fpgautil 烧录 bitstream"
echo "=================================================="
# 释放之前可能被锁定的 FPGA 资源
sudo fpgautil -R 2>/dev/null || true
# 烧录 bitstream
sudo fpgautil -b "$BITSTREAM"
if [ $? -ne 0 ]; then
    echo "❌ [ERROR] FPGA bitstream 烧录失败。"
    exit 1
fi
echo "✅ Bitstream 烧录完成。"

echo ""
echo "=================================================="
echo "    [STEP 3] 开启 AXI 隔离门 (PMU FPD)"
echo "=================================================="
sudo python3 -c '
import os, struct, mmap
fd = os.open("/dev/mem", os.O_RDWR)
m = mmap.mmap(fd, 0x1000, offset=0xFF5E0000)
m[0x230:0x234] = struct.pack("<I", 0)
os.close(fd)
'
echo "✅ PMU AXI FPD 隔离门已开启！"

echo ""
echo "=================================================="
echo " ✅ FPGA 初始化完成！可以运行推理了。"
echo "=================================================="
