cmake_minimum_required(VERSION 3.20)
project(ppo_ranker_libtorch_smoke LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED Torch_DIR)
    message(FATAL_ERROR "Torch_DIR is not set. Point it to <python-env>/Lib/site-packages/torch/share/cmake/Torch")
endif()

find_package(Torch REQUIRED)

add_executable(ppo_ranker_libtorch_smoke
    main.cpp
)

target_link_libraries(ppo_ranker_libtorch_smoke PRIVATE "${TORCH_LIBRARIES}")
target_compile_features(ppo_ranker_libtorch_smoke PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(ppo_ranker_libtorch_smoke PRIVATE /W4 /EHsc)
else()
    target_compile_options(ppo_ranker_libtorch_smoke PRIVATE -Wall -Wextra -Wpedantic)
endif()
