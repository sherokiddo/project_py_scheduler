#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

class PpoRankerONNXRuntime {
public:
    explicit PpoRankerONNXRuntime(const std::string& model_path);

    std::vector<float> predict_scores(
        const float* obs_data,
        std::size_t obs_len,
        const int32_t* action_mask_data,
        std::size_t action_mask_len
    );

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
    Ort::MemoryInfo memory_info_;
};
