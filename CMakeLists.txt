cmake_minimum_required(VERSION 3.12)
project(v37_software LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

# ==============================================================================
# 平台选择: KV260 (板端硬件) | EMULATOR (x86仿真) | CPU_REF (纯CPU黄金参考)
# ==============================================================================
set(PLATFORM "CPU_REF" CACHE STRING "Target platform: KV260, EMULATOR, or CPU_REF")
message(STATUS "Selected PLATFORM: ${PLATFORM}")

# 架构判定
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(TARGET_ARCH "arm64")
else()
    set(TARGET_ARCH "x86_64")
endif()
message(STATUS "Detected Architecture: ${TARGET_ARCH}")

# 编译参数
add_compile_options(-O3 -fPIC -Wall -Wextra)
if (TARGET_ARCH STREQUAL "arm64")
    add_compile_options(-mcpu=cortex-a53+crypto -mtune=cortex-a53 -ftree-vectorize -funroll-loops)
endif()

# OpenMP
find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    add_compile_options(${OpenMP_CXX_FLAGS})
endif()

# 包含路径
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/common/include
    ${CMAKE_CURRENT_SOURCE_DIR}/engine/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/include
)

# 依赖库路径与特化源码
if (PLATFORM STREQUAL "KV260")
    set(LIB_ARCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lib/arm64)
    set(DRIVER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/platforms/kv260_hardware/src/v37_driver_hw.cpp)
    add_definitions(-DV37_PLATFORM_KV260)
elseif (PLATFORM STREQUAL "EMULATOR")
    set(LIB_ARCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lib/x86_64)
    set(DRIVER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/platforms/x86_emulator/src/v37_driver_emu.cpp)
    add_definitions(-DV37_PLATFORM_EMULATOR)
else()
    # 纯 CPU 黄金参考模式
    if (TARGET_ARCH STREQUAL "arm64")
        set(LIB_ARCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lib/arm64)
    else()
        set(LIB_ARCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lib/x86_64)
    endif()
    set(DRIVER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/platforms/cpu_reference/src/v37_driver_cpu.cpp)
    add_definitions(-DV37_PLATFORM_CPUREF -DDISABLE_V37=1)
endif()

link_directories(${LIB_ARCH_DIR})

# RPATH 设置: 优先使用 $ORIGIN 相对定位，支持随处移动
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN/../third_party/lib/${TARGET_ARCH}:${LIB_ARCH_DIR}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags -Wl,-rpath,'$ORIGIN/../third_party/lib/${TARGET_ARCH}' -Wl,-rpath-link,${LIB_ARCH_DIR}")

# ==============================================================================
# 目标 1: semantic_engine (核心自回归推理程序)
# ==============================================================================
set(ENGINE_SOURCES
    engine/src/main.cpp
    engine/src/engine.cpp
    engine/src/loader.cpp
    engine/src/attention.cpp
    engine/src/ggml_type_i2s_override.c
    engine/src/ggml-backend-v37.c
    ${DRIVER_SRC}
)

add_executable(semantic_engine ${ENGINE_SOURCES})

target_link_libraries(semantic_engine
    PRIVATE
    -Wl,--no-as-needed
    ${LIB_ARCH_DIR}/libllama.so
    ${LIB_ARCH_DIR}/libggml.so
    ${LIB_ARCH_DIR}/libggml-cpu.so
    ${LIB_ARCH_DIR}/libggml-base.so
    ${LIB_ARCH_DIR}/libllama-common.so
    -Wl,--as-needed
    pthread
    OpenMP::OpenMP_CXX
)

# ==============================================================================
# 说明: build_v37p (AOT 模型与微码离线编译器) 已完全剥离至独立仓库 V37_compiler
# 源码与构建工具链详见: /mnt/nvme_linux/V37_fpga_upload/V37_compiler
# ==============================================================================

# ==============================================================================
# 目标 2: test_dma (硬件冒烟测试工具，仅限 KV260)
# ==============================================================================
if (PLATFORM STREQUAL "KV260")
    add_executable(test_dma
        platforms/kv260_hardware/src/test_dma.cpp
        ${DRIVER_SRC}
    )
    target_link_libraries(test_dma
        PRIVATE
        pthread
        OpenMP::OpenMP_CXX
    )
endif()

# 输出路径规范到 bin 目录 (arm64 独立输出至 bin/arm64，避免与 x86 产物相互覆盖)
if (TARGET_ARCH STREQUAL "arm64")
    set(DEFAULT_BIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bin/arm64")
else()
    set(DEFAULT_BIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bin")
endif()

set_target_properties(semantic_engine PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${DEFAULT_BIN_DIR}")
if (TARGET test_dma)
    set_target_properties(test_dma PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${DEFAULT_BIN_DIR}")
endif()
