#include "ppo_ranker_onnx_c_api.h"

#include "ppo_ranker_onnx_runtime.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

thread_local std::string g_last_error;

void set_last_error(const std::string& error_message) {
    g_last_error = error_message;
}

}  // namespace

extern "C" {

int ppo_ranker_onnx_create(const char* model_path, void** out_handle) {
    try {
        if (model_path == nullptr) {
            throw std::invalid_argument("model_path must not be null.");
        }
        if (out_handle == nullptr) {
            throw std::invalid_argument("out_handle must not be null.");
        }

        auto* runtime = new PpoRankerONNXRuntime(std::string{model_path});
        *out_handle = runtime;
        g_last_error.clear();
        return 0;
    } catch (const std::exception& error) {
        set_last_error(error.what());
        return 1;
    }
}

int ppo_ranker_onnx_predict(
    void* handle,
    const float* obs,
    int obs_len,
    const int32_t* action_mask,
    int action_mask_len,
    float* out_scores,
    int out_scores_len
) {
    try {
        if (handle == nullptr) {
            throw std::invalid_argument("handle must not be null.");
        }
        if (obs == nullptr) {
            throw std::invalid_argument("obs must not be null.");
        }
        if (action_mask == nullptr) {
            throw std::invalid_argument("action_mask must not be null.");
        }
        if (out_scores == nullptr) {
            throw std::invalid_argument("out_scores must not be null.");
        }
        if (obs_len <= 0) {
            throw std::invalid_argument("obs_len must be greater than zero.");
        }
        if (action_mask_len <= 0) {
            throw std::invalid_argument("action_mask_len must be greater than zero.");
        }
        if (out_scores_len != action_mask_len) {
            throw std::invalid_argument("out_scores_len must match action_mask_len.");
        }

        auto* runtime = static_cast<PpoRankerONNXRuntime*>(handle);
        std::vector<float> score_vector = runtime->predict_scores(
            obs,
            static_cast<std::size_t>(obs_len),
            action_mask,
            static_cast<std::size_t>(action_mask_len)
        );
        if (score_vector.size() != static_cast<std::size_t>(out_scores_len)) {
            throw std::runtime_error("Runtime output size does not match out_scores_len.");
        }

        std::copy(score_vector.begin(), score_vector.end(), out_scores);
        g_last_error.clear();
        return 0;
    } catch (const std::exception& error) {
        set_last_error(error.what());
        return 1;
    }
}

void ppo_ranker_onnx_destroy(void* handle) {
    auto* runtime = static_cast<PpoRankerONNXRuntime*>(handle);
    delete runtime;
}

const char* ppo_ranker_onnx_last_error() {
    return g_last_error.c_str();
}

}  // extern "C"
