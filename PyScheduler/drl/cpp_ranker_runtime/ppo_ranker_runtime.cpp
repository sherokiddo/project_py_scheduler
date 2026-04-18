#include "ppo_ranker_runtime.h"

#include <cstring>
#include <stdexcept>

namespace {

torch::Tensor make_obs_tensor(const float* obs_data, const std::size_t obs_len) {
    return torch::from_blob(
        const_cast<float*>(obs_data),
        {1, static_cast<long long>(obs_len)},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone();
}

torch::Tensor make_action_mask_tensor(
    const int32_t* action_mask_data,
    const std::size_t action_mask_len
) {
    std::vector<uint8_t> bool_mask(action_mask_len, 0);
    for (std::size_t idx = 0; idx < action_mask_len; ++idx) {
        bool_mask[idx] = static_cast<uint8_t>(action_mask_data[idx] != 0);
    }

    return torch::from_blob(
        bool_mask.data(),
        {1, static_cast<long long>(action_mask_len)},
        torch::TensorOptions().dtype(torch::kBool)
    ).clone();
}

}  // namespace

PpoRankerRuntime::PpoRankerRuntime(const std::string& model_path)
    : module_(torch::jit::load(model_path, torch::kCPU))
{
    module_.eval();
}

std::vector<float> PpoRankerRuntime::predict_scores(
    const float* obs_data,
    const std::size_t obs_len,
    const int32_t* action_mask_data,
    const std::size_t action_mask_len
) {
    if (obs_data == nullptr) {
        throw std::invalid_argument("obs_data must not be null.");
    }
    if (action_mask_data == nullptr) {
        throw std::invalid_argument("action_mask_data must not be null.");
    }
    if (obs_len == 0) {
        throw std::invalid_argument("obs_len must be greater than zero.");
    }
    if (action_mask_len == 0) {
        throw std::invalid_argument("action_mask_len must be greater than zero.");
    }

    torch::InferenceMode inference_guard;

    torch::Tensor obs_tensor = make_obs_tensor(obs_data, obs_len);
    torch::Tensor action_mask_tensor = make_action_mask_tensor(action_mask_data, action_mask_len);

    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(obs_tensor);
    inputs.emplace_back(action_mask_tensor);

    torch::Tensor output = module_.forward(inputs).toTensor().to(torch::kCPU).contiguous().squeeze(0);
    if (output.numel() != static_cast<long long>(action_mask_len)) {
        throw std::runtime_error("TorchScript output size does not match action_mask_len.");
    }

    std::vector<float> score_vector(static_cast<std::size_t>(output.numel()), 0.0f);
    std::memcpy(score_vector.data(), output.data_ptr<float>(), score_vector.size() * sizeof(float));
    return score_vector;
}
