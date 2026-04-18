#pragma once

#include <cstdint>

#if defined(_WIN32)
#if defined(PPO_RANKER_ONNX_RUNTIME_EXPORTS)
#define PPO_RANKER_ONNX_API __declspec(dllexport)
#else
#define PPO_RANKER_ONNX_API __declspec(dllimport)
#endif
#else
#define PPO_RANKER_ONNX_API
#endif

extern "C" {

PPO_RANKER_ONNX_API int ppo_ranker_onnx_create(const char* model_path, void** out_handle);

PPO_RANKER_ONNX_API int ppo_ranker_onnx_predict(
    void* handle,
    const float* obs,
    int obs_len,
    const int32_t* action_mask,
    int action_mask_len,
    float* out_scores,
    int out_scores_len
);

PPO_RANKER_ONNX_API void ppo_ranker_onnx_destroy(void* handle);

PPO_RANKER_ONNX_API const char* ppo_ranker_onnx_last_error();

}
