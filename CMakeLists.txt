# qw: inference engine for Qwen3.8-Flash-Next on 4x AMD Radeon Pro V620 (gfx1030).
cmake_minimum_required(VERSION 3.21)
project(qw LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")
set(CMAKE_HIP_FLAGS_RELWITHDEBINFO "-O3 -g")

option(QW_GPU "Build the GPU engine (needs ROCm)" ON)
option(QW_DEVICE_CHECKS "Device-side bounds assertions in the kernels (debugging)" OFF)
find_package(Threads REQUIRED)

# ---------------------------------------------------------------- host libraries
add_library(qw_core STATIC
  src/core/config.cpp
  src/core/json.cpp
  src/core/ple.cpp
  src/core/safetensors.cpp)
target_include_directories(qw_core PUBLIC src)
target_compile_options(qw_core PRIVATE -march=native -Wall -Wextra)
target_link_libraries(qw_core PUBLIC Threads::Threads)

# fp32 CPU reference of the model (validation)
add_library(qw_ref STATIC src/ref/ref_model.cpp)
target_compile_options(qw_ref PRIVATE -march=native -fopenmp-simd -Wall -Wextra)
target_link_libraries(qw_ref PUBLIC qw_core)

add_executable(qw_ref_cli tools/qw_ref.cpp)
set_target_properties(qw_ref_cli PROPERTIES OUTPUT_NAME qw_ref)
target_link_libraries(qw_ref_cli PRIVATE qw_ref)

enable_testing()
add_executable(test_ngram tests/unit/test_ngram.cpp)
target_link_libraries(test_ngram PRIVATE qw_core)
add_test(NAME ngram COMMAND test_ngram)
add_executable(test_chunker tests/unit/test_chunker.cpp src/session/chunker.cpp)
target_include_directories(test_chunker PRIVATE src)
add_test(NAME chunker COMMAND test_chunker)

if(NOT QW_GPU)
  return()
endif()

# ---------------------------------------------------------------- GPU engine
set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "ROCm install prefix")
if(NOT ROCM_PATH)
  set(ROCM_PATH /opt/rocm CACHE PATH "ROCm install prefix" FORCE)
endif()
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
set(CMAKE_HIP_ARCHITECTURES gfx1030)
set(CMAKE_HIP_STANDARD 17)
enable_language(HIP)
find_package(hip REQUIRED)
find_package(rocblas REQUIRED)

set(QW_HIP_SOURCES
  src/kernels/gemm.hip
  src/kernels/gemv.hip
  src/kernels/hc.hip
  src/kernels/gdn.hip
  src/kernels/qsa.hip
  src/kernels/moe_route.hip
  src/kernels/moe_experts.hip
  src/kernels/moe_w4a8.hip
  src/kernels/ple.hip
  src/kernels/mtp.hip
  src/kernels/logits.hip
  src/kernels/blend.hip
  src/kernels/vision.hip
  src/vision/vision_encoder.hip
  src/comm/comm.hip
  src/engine/weights.hip
  src/engine/buffers.hip
  src/engine/engine.hip
  src/engine/decode.hip
  src/engine/speculative.hip
  src/engine/prefill.hip
  src/engine/snapshot.hip
  src/engine/host_tier.hip
  src/engine/blend.hip
  src/engine/vision_job.hip)
# host code that includes HIP headers
set(QW_HIP_HOST_SOURCES
  src/engine/device_mem.cpp
  src/engine/pinned_pool.cpp
  src/engine/weight_convert.cpp
  src/vision/vision_weights.cpp
  src/session/session.cpp
  src/session/generate.cpp
  src/session/prefix_cache.cpp
  src/session/block_store.cpp
  src/session/vision_cache.cpp
  src/session/disk_tier.cpp
  src/session/sampling.cpp)
set_source_files_properties(${QW_HIP_HOST_SOURCES} src/capi/capi.cpp PROPERTIES LANGUAGE HIP)

add_library(qw_gpu STATIC ${QW_HIP_SOURCES} ${QW_HIP_HOST_SOURCES} src/session/chunker.cpp)
target_include_directories(qw_gpu PUBLIC src include)
target_compile_options(qw_gpu PRIVATE -Wall)
if(QW_DEVICE_CHECKS)
  target_compile_definitions(qw_gpu PRIVATE QW_DEVICE_CHECKS)
endif()
target_link_libraries(qw_gpu PUBLIC qw_core roc::rocblas hip::host)

# The engine as a shared library with a C API (include/qw/capi.h), for server/.
add_library(qw_engine SHARED src/capi/capi.cpp)
target_include_directories(qw_engine PUBLIC include)
target_link_libraries(qw_engine PRIVATE qw_gpu)

function(qw_gpu_program name src)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE qw_gpu)
endfunction()

qw_gpu_program(qw_gpu_cli tools/qw_gpu.hip)
qw_gpu_program(qw_layer_diff tools/qw_layer_diff.hip)
qw_gpu_program(qw_blend_eval tools/qw_blend_eval.hip)
set_target_properties(qw_gpu_cli PROPERTIES OUTPUT_NAME qw_gpu)

foreach(b p2p_bench gemm_bench uncached_bench prefix_cache_bench)
  qw_gpu_program(${b} bench/${b}.hip)
endforeach()
qw_gpu_program(stress_comm tests/gpu/stress_comm.hip)

foreach(t test_comm test_batch_decode test_speculative test_host_tier test_snapshot_capture test_block_store test_vision_encoder test_moe test_gemv)
  qw_gpu_program(${t} tests/gpu/${t}.hip)
endforeach()
add_test(NAME comm COMMAND test_comm)
set_tests_properties(comm PROPERTIES LABELS gpu)
