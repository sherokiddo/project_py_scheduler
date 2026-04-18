#include "ppo_ranker_onnx_runtime.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace {

std::vector<std::uint8_t> make_action_mask_buffer(
    const int32_t* action_mask_data,
    const std::size_t action_mask_len
) {
    std::vector<std::uint8_t> bool_mask(action_mask_len, 0);
    for (std::size_t idx = 0; idx < action_mask_len; ++idx) {
        bool_mask[idx] = static_cast<std::uint8_t>(action_mask_data[idx] != 0);
    }
    return bool_mask;
}

Ort::SessionOptions make_session_options() {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED
    );
    return session_options;
}

}  // namespace

PpoRankerONNXRuntime::PpoRankerONNXRuntime(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "ppo_ranker_onnx_runtime"),
      session_options_(make_session_options()),
      session_(
          env_,
          std::filesystem::path(model_path).c_str(),
          session_options_
      ),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
}

std::vector<float> PpoRankerONNXRuntime::predict_scores(
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

    const std::array<int64_t, 2> obs_shape = {
        1,
        static_cast<int64_t>(obs_len),
    };
    const std::array<int64_t, 2> mask_shape = {
        1,
        static_cast<int64_t>(action_mask_len),
    };

    Ort::Value obs_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(obs_data),
        obs_len,
        obs_shape.data(),
        obs_shape.size()
    );

    std::vector<std::uint8_t> bool_mask = make_action_mask_buffer(
        action_mask_data,
        action_mask_len
    );
    Ort::Value action_mask_tensor = Ort::Value::CreateTensor<bool>(
        memory_info_,
        reinterpret_cast<bool*>(bool_mask.data()),
        bool_mask.size(),
        mask_shape.data(),
        mask_shape.size()
    );

    std::array<const char*, 2> input_names = {"obs", "action_mask"};
    std::array<const char*, 1> output_names = {"score_vector"};
    std::array<Ort::Value, 2> input_tensors = {
        std::move(obs_tensor),
        std::move(action_mask_tensor),
    };

    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        input_tensors.data(),
        input_tensors.size(),
        output_names.data(),
        output_names.size()
    );

    if (output_tensors.size() != 1) {
        throw std::runtime_error("Unexpected number of outputs from ONNX session.");
    }

    Ort::Value& output_tensor = output_tensors.front();
    auto shape_info = output_tensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = shape_info.GetShape();

    std::size_t output_size = 1;
    for (const int64_t dim : output_shape) {
        if (dim < 0) {
            throw std::runtime_error("Dynamic output shape is not supported in runtime output materialization.");
        }
        output_size *= static_cast<std::size_t>(dim);
    }
    if (output_size != action_mask_len) {
        throw std::runtime_error("ONNX output size does not match action_mask_len.");
    }

    const float* output_data = output_tensor.GetTensorData<float>();
    return std::vector<float>(output_data, output_data + output_size);
}
