cmake_minimum_required(VERSION 3.20)
project(ppo_ranker_onnx_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "ONNXRUNTIME_ROOT is not set. Point it to the extracted onnxruntime-win-x64 directory.")
endif()

set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
set(ONNXRUNTIME_DLL "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")
set(ONNXRUNTIME_SHARED_PROVIDERS_DLL "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll")

if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "onnxruntime_cxx_api.h not found under ${ONNXRUNTIME_INCLUDE_DIR}")
endif()

if(NOT EXISTS "${ONNXRUNTIME_LIBRARY}")
    message(FATAL_ERROR "onnxruntime.lib not found under ${ONNXRUNTIME_LIBRARY}")
endif()

add_library(ppo_ranker_onnx_runtime SHARED
    ppo_ranker_onnx_runtime.cpp
    ppo_ranker_onnx_c_api.cpp
)

target_include_directories(ppo_ranker_onnx_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    "${ONNXRUNTIME_INCLUDE_DIR}"
)

target_link_libraries(ppo_ranker_onnx_runtime PRIVATE "${ONNXRUNTIME_LIBRARY}")
target_compile_features(ppo_ranker_onnx_runtime PRIVATE cxx_std_17)
target_compile_definitions(ppo_ranker_onnx_runtime PRIVATE PPO_RANKER_ONNX_RUNTIME_EXPORTS)

if(MSVC)
    target_compile_options(ppo_ranker_onnx_runtime PRIVATE /W4 /EHsc)
else()
    target_compile_options(ppo_ranker_onnx_runtime PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(EXISTS "${ONNXRUNTIME_DLL}")
    add_custom_command(
        TARGET ppo_ranker_onnx_runtime POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DLL}"
            "$<TARGET_FILE_DIR:ppo_ranker_onnx_runtime>/onnxruntime.dll"
    )
endif()

if(EXISTS "${ONNXRUNTIME_SHARED_PROVIDERS_DLL}")
    add_custom_command(
        TARGET ppo_ranker_onnx_runtime POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_SHARED_PROVIDERS_DLL}"
            "$<TARGET_FILE_DIR:ppo_ranker_onnx_runtime>/onnxruntime_providers_shared.dll"
    )
endif()
