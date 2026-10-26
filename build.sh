#!/bin/bash
# ==============================================================================
# V37 Software 统一构建管理脚本
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BIN_DIR="${SCRIPT_DIR}/bin"

print_header() {
    echo "=================================================================="
    echo "  🚀 V37 BitNet 软件推理栈构建管理系统"
    echo "=================================================================="
}

print_guide() {
    echo ""
    echo "💡 【首次使用与开发指引】"
    echo "  • 场景 A（x86 电脑快速对齐验证）:"
    echo "    若您刚拉取代码并在普通 Linux PC/服务器上测试，推荐选择 [1] CPU_REF。"
    echo "    无需连接 FPGA 板卡，直接运行验证 30 Tokens 黄金标准逐字对齐。"
    echo "  • 场景 B（x86 脉动阵列纯算仿真）:"
    echo "    若需要验证软件驱动对 FPGA 内部数据流与量化行为的软件仿真，请选 [2] EMULATOR。"
    echo "  • 场景 C（KV260 板端硬件加速部署）:"
    echo "    若在 Kria KV260 板端终端本地编译，请选 [3] KV260。"
    echo "=================================================================="
}

show_menu() {
    print_header
    print_guide
    echo ""
    echo "请选择构建目标平台："
    echo "  1) CPU_REF   - 纯 CPU 黄金参考模式 (x86/ARM 通用，推荐首次验证)"
    echo "  2) EMULATOR  - x86 脉动阵列仿真模式 (PC 端驱动逻辑模拟)"
    echo "  3) KV260     - KV260 板端真实 FPGA 加速模式 (含 DMA/CMA 驱动)"
    echo "  4) clean     - 清空 build/ 和 bin/ 重新构建"
    echo "  q) 退出"
    echo ""
    read -p "请输入选项 [1-4 或 q]: " choice
    case "$choice" in
        1) do_build "CPU_REF" ;;
        2) do_build "EMULATOR" ;;
        3) do_build "KV260" ;;
        4) do_clean ;;
        q|Q) echo "退出构建。"; exit 0 ;;
        *) echo "无效输入，退出。"; exit 1 ;;
    esac
}

do_clean() {
    echo "🧹 正在清理构建产物..."
    rm -rf "${SCRIPT_DIR}/build" "${SCRIPT_DIR}"/build_* "${BIN_DIR}"
    echo "✨ 清理完成！"
}

do_build() {
    local platform="$1"
    local p_lower=$(echo "${platform}" | tr '[:upper:]' '[:lower:]')
    local cur_build_dir="${SCRIPT_DIR}/build_${p_lower}"

    echo ""
    echo "🔨 正在开始构建平台目标: [${platform}] ..."
    mkdir -p "${cur_build_dir}"
    mkdir -p "${BIN_DIR}"

    local extra_cmake_flags=()
    if [ "${platform}" = "KV260" ] && [ "$(uname -m)" != "aarch64" ]; then
        echo "🔍 检测到宿主机为 $(uname -m)，正在启用 aarch64-linux-gnu 交叉编译工具链..."
        extra_cmake_flags+=(
            "-DCMAKE_SYSTEM_NAME=Linux"
            "-DCMAKE_SYSTEM_PROCESSOR=aarch64"
            "-DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc"
            "-DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++"
        )
    fi

    cd "${cur_build_dir}"
    cmake .. -DPLATFORM="${platform}" -DCMAKE_BUILD_TYPE=Release "${extra_cmake_flags[@]}"
    cmake --build . -- -j"$(nproc)"

    echo ""
    echo "=================================================================="
    if [ "${platform}" = "KV260" ] && [ -d "${BIN_DIR}/arm64" ]; then
        echo "✅ 构建成功！ARM64 可执行文件已生成在: ${BIN_DIR}/arm64/"
        ls -lh "${BIN_DIR}/arm64/"
    else
        echo "✅ 构建成功！可执行文件已生成在: ${BIN_DIR}/"
        ls -lh "${BIN_DIR}/"
    fi
    echo "=================================================================="
}

# 命令行非交互式快捷支持
if [ $# -gt 0 ]; then
    case "$1" in
        1|cpuref|CPU_REF)
            do_build "CPU_REF"
            ;;
        2|emu|emulator|EMULATOR)
            do_build "EMULATOR"
            ;;
        3|kv260|KV260)
            do_build "KV260"
            ;;
        4|clean)
            do_clean
            ;;
        help|-h|--help)
            print_header
            print_guide
            echo "用法: ./build.sh [cpuref | emulator | kv260 | clean]"
            ;;
        *)
            echo "❌ 未知参数: $1"
            echo "可用参数: cpuref, emulator, kv260, clean"
            exit 1
            ;;
    esac
else
    show_menu
fi
