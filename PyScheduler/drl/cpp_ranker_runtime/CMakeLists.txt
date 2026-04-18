cmake_minimum_required(VERSION 3.20)
project(ppo_ranker_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED Torch_DIR)
    message(FATAL_ERROR "Torch_DIR is not set. Point it to <libtorch>/share/cmake/Torch")
endif()

find_package(Torch REQUIRED)

add_library(ppo_ranker_runtime SHARED
    ppo_ranker_runtime.cpp
    ppo_ranker_c_api.cpp
)

target_include_directories(ppo_ranker_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ppo_ranker_runtime PRIVATE "${TORCH_LIBRARIES}")
target_compile_features(ppo_ranker_runtime PRIVATE cxx_std_17)
target_compile_definitions(ppo_ranker_runtime PRIVATE PPO_RANKER_RUNTIME_EXPORTS)

if(MSVC)
    target_compile_options(ppo_ranker_runtime PRIVATE /W4 /EHsc)
else()
    target_compile_options(ppo_ranker_runtime PRIVATE -Wall -Wextra -Wpedantic)
endif()
