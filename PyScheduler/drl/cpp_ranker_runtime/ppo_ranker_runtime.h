#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <torch/script.h>

class PpoRankerRuntime {
public:
    explicit PpoRankerRuntime(const std::string& model_path);

    std::vector<float> predict_scores(
        const float* obs_data,
        std::size_t obs_len,
        const int32_t* action_mask_data,
        std::size_t action_mask_len
    );

private:
    torch::jit::script::Module module_;
};
