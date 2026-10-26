#!/bin/bash
# ==============================================================================
# V37 Software 黄金标准逐字自动对齐测试脚本
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIN_PATH="${ROOT_DIR}/bin/semantic_engine"

# 默认模型选择：优先检测 Format v4 统一自包含模型，若无则回退至 Format v3 遗留模型
DEFAULT_MODEL="${ROOT_DIR}/models/bitnet-2b-unified.v37p"
if [ ! -f "${DEFAULT_MODEL}" ]; then
    DEFAULT_MODEL="${ROOT_DIR}/models/ggml-model-i2_s.gguf"
fi

MODEL_ARG="${1:-${DEFAULT_MODEL}}"
EXTRA_WEIGHT_ARG=""

if [ ! -f "${BIN_PATH}" ]; then
    echo "❌ 未找到可执行文件: ${BIN_PATH}，请先执行 ./build.sh 编译！"
    exit 1
fi

if [ ! -f "${MODEL_ARG}" ]; then
    echo "❌ 未找到模型文件: ${MODEL_ARG}"
    exit 1
fi

if [[ "${MODEL_ARG}" == *.gguf ]]; then
    # Format v3 双文件模式
    MODEL_V37P="${2:-${MODEL_ARG%.gguf}.v37p}"
    if [ ! -f "${MODEL_V37P}" ]; then
        MODEL_V37P="${ROOT_DIR}/models/ggml-model-i2_s.v37p"
    fi
    EXTRA_WEIGHT_ARG="-w ${MODEL_V37P}"
    echo "📦 模式: Format v3 双文件兼容 (结构: $(basename ${MODEL_ARG}), 权重: $(basename ${MODEL_V37P}))"
else
    # Format v4 单文件统一容器模式
    echo "📦 模式: Format v4 统一自包含容器 ($(basename ${MODEL_ARG}))"
fi

LOG_OUT="/tmp/v37_test_output.log"
GOLDEN_TEXT="a device that performs specific tasks, such as encryption or decryption, and can be used to speed up the processing of data. This type of processor is"
EMU_GOLDEN_TEXT="a device that performs specific tasks, such as encryption or decryption, and can be used to speed up the processing of data. It is often used in"
CPUREF_GOLDEN_TEXT="a device that performs specific tasks, such as encryption or compression. It can be integrated into the system or used separately."

echo "正在执行 30 Tokens 推理..."
"${BIN_PATH}" "${MODEL_ARG}" ${EXTRA_WEIGHT_ARG} -p "The hardware accelerator is" -n 30 -t 4 -r 1.10 | tee "${LOG_OUT}"

ACTUAL_TEXT=$(grep "OUTPUT:" "${LOG_OUT}" | sed 's/OUTPUT:[ ]*//')

echo ""
echo "=================================================================="
echo "🎯 官方硬件/CPU 黄金基线 (KV260 FPGA):"
echo "   \"${GOLDEN_TEXT}\""
echo "🎯 x86 仿真器黄金基线 (env2):"
echo "   \"${EMU_GOLDEN_TEXT}\""
echo "🎯 x86 CPU Reference 黄金基线:"
echo "   \"${CPUREF_GOLDEN_TEXT}\""
echo "🎯 实际推理输出:"
echo "   \"${ACTUAL_TEXT}\""
echo "=================================================================="

if [ "${ACTUAL_TEXT}" = "${GOLDEN_TEXT}" ] || [ "${ACTUAL_TEXT}" = "${EMU_GOLDEN_TEXT}" ] || [ "${ACTUAL_TEXT}" = "${CPUREF_GOLDEN_TEXT}" ]; then
    echo "🎉 [PASSED] 30 个 Token 逐字 100.00% 绝对吻合！完全通过黄金标准测试！"
    exit 0
else
    echo "❌ [FAILED] 输出与黄金基线不一致，请检查差异！"
    exit 1
fi
