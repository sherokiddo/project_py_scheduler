#include <algorithm>
#include <chrono>
#include <torch/script.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> read_float_vector_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open float vector file: " + path);
    }

    std::size_t size = 0;
    file >> size;
    if (!file.good()) {
        throw std::runtime_error("Failed to read vector size from: " + path);
    }

    std::vector<float> values(size, 0.0f);
    for (std::size_t idx = 0; idx < size; ++idx) {
        file >> values[idx];
        if (!file.good()) {
            throw std::runtime_error("Failed to read float value at index " + std::to_string(idx) + " from: " + path);
        }
    }
    return values;
}

std::vector<int64_t> read_int_vector_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open int vector file: " + path);
    }

    std::size_t size = 0;
    file >> size;
    if (!file.good()) {
        throw std::runtime_error("Failed to read vector size from: " + path);
    }

    std::vector<int64_t> values(size, 0);
    for (std::size_t idx = 0; idx < size; ++idx) {
        file >> values[idx];
        if (!file.good()) {
            throw std::runtime_error("Failed to read int value at index " + std::to_string(idx) + " from: " + path);
        }
    }
    return values;
}

void print_first_scores(const std::vector<float>& values, const std::string& label) {
    constexpr std::size_t kMaxPrint = 8;
    std::cout << label << " [";
    const std::size_t count = std::min(values.size(), kMaxPrint);
    for (std::size_t idx = 0; idx < count; ++idx) {
        if (idx > 0) {
            std::cout << ", ";
        }
        std::cout << std::fixed << std::setprecision(6) << values[idx];
    }
    if (values.size() > kMaxPrint) {
        std::cout << ", ...";
    }
    std::cout << "]\n";
}

std::size_t parse_size_t_arg(const std::string& value, const std::string& flag_name) {
    try {
        return static_cast<std::size_t>(std::stoull(value));
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid numeric value for " + flag_name + ": " + value);
    }
}

float percentile_from_sorted(const std::vector<double>& sorted_values, const double percentile) {
    if (sorted_values.empty()) {
        return 0.0f;
    }
    const double clamped = std::max(0.0, std::min(100.0, percentile));
    const double position = (clamped / 100.0) * static_cast<double>(sorted_values.size() - 1);
    const auto index = static_cast<std::size_t>(std::llround(position));
    return static_cast<float>(sorted_values[index]);
}

torch::Tensor run_forward(
    torch::jit::script::Module& module,
    const std::vector<torch::jit::IValue>& inputs
) {
    return module.forward(inputs).toTensor().to(torch::kCPU).contiguous().squeeze(0);
}

std::vector<float> tensor_to_float_vector(const torch::Tensor& tensor) {
    std::vector<float> values(tensor.numel(), 0.0f);
    std::memcpy(values.data(), tensor.data_ptr<float>(), values.size() * sizeof(float));
    return values;
}

}  // namespace

int main(int argc, const char* argv[]) {
    if (argc < 5) {
        std::cerr
            << "Usage:\n"
            << "  ppo_ranker_libtorch_smoke <module.ts> <obs.txt> <action_mask.txt> <expected_scores.txt> "
               "[--benchmark N] [--warmup N]\n";
        return 2;
    }

    const std::string module_path = argv[1];
    const std::string obs_path = argv[2];
    const std::string action_mask_path = argv[3];
    const std::string expected_scores_path = argv[4];
    std::size_t benchmark_iterations = 0;
    std::size_t warmup_iterations = 200;

    try {
        for (int arg_idx = 5; arg_idx < argc; ++arg_idx) {
            const std::string flag = argv[arg_idx];
            if (flag == "--benchmark") {
                if (arg_idx + 1 >= argc) {
                    throw std::runtime_error("Missing value after --benchmark.");
                }
                benchmark_iterations = parse_size_t_arg(argv[++arg_idx], "--benchmark");
            } else if (flag == "--warmup") {
                if (arg_idx + 1 >= argc) {
                    throw std::runtime_error("Missing value after --warmup.");
                }
                warmup_iterations = parse_size_t_arg(argv[++arg_idx], "--warmup");
            } else {
                throw std::runtime_error("Unknown argument: " + flag);
            }
        }

        std::vector<float> obs_values = read_float_vector_file(obs_path);
        std::vector<int64_t> mask_values = read_int_vector_file(action_mask_path);
        std::vector<float> expected_scores = read_float_vector_file(expected_scores_path);

        if (mask_values.size() != expected_scores.size()) {
            throw std::runtime_error("action_mask size must match expected_scores size.");
        }

        torch::jit::script::Module module = torch::jit::load(module_path, torch::kCPU);
        module.eval();

        torch::Tensor obs_tensor = torch::from_blob(
            obs_values.data(),
            {1, static_cast<long long>(obs_values.size())},
            torch::TensorOptions().dtype(torch::kFloat32)
        ).clone();

        torch::Tensor mask_tensor = torch::zeros(
            {1, static_cast<long long>(mask_values.size())},
            torch::TensorOptions().dtype(torch::kBool)
        );
        for (std::size_t idx = 0; idx < mask_values.size(); ++idx) {
            mask_tensor[0][static_cast<long long>(idx)] = (mask_values[idx] != 0);
        }

        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(obs_tensor);
        inputs.emplace_back(mask_tensor);

        torch::Tensor output = run_forward(module, inputs);
        if (output.numel() != static_cast<long long>(expected_scores.size())) {
            throw std::runtime_error("Output size does not match expected score vector size.");
        }

        std::vector<float> actual_scores = tensor_to_float_vector(output);

        double max_abs_error = 0.0;
        double mean_abs_error = 0.0;
        for (std::size_t idx = 0; idx < actual_scores.size(); ++idx) {
            const double abs_error = std::abs(static_cast<double>(actual_scores[idx]) - static_cast<double>(expected_scores[idx]));
            max_abs_error = std::max(max_abs_error, abs_error);
            mean_abs_error += abs_error;
        }
        mean_abs_error /= std::max<std::size_t>(actual_scores.size(), 1);

        std::cout << "TorchScript module: " << module_path << "\n";
        std::cout << "obs_dim: " << obs_values.size() << "\n";
        std::cout << "max_n_ue: " << mask_values.size() << "\n";
        print_first_scores(expected_scores, "expected_scores");
        print_first_scores(actual_scores, "actual_scores");
        std::cout << "max_abs_error: " << std::fixed << std::setprecision(8) << max_abs_error << "\n";
        std::cout << "mean_abs_error: " << std::fixed << std::setprecision(8) << mean_abs_error << "\n";

        constexpr double kTolerance = 1e-5;
        if (max_abs_error > kTolerance) {
            std::cerr << "Smoke-test FAILED: max_abs_error exceeds tolerance " << kTolerance << "\n";
            return 1;
        }

        std::cout << "Smoke-test PASSED\n";

        if (benchmark_iterations > 0) {
            volatile float benchmark_sink = 0.0f;

            for (std::size_t warmup_idx = 0; warmup_idx < warmup_iterations; ++warmup_idx) {
                torch::Tensor warmup_output = run_forward(module, inputs);
                benchmark_sink += warmup_output[0].item<float>();
            }

            std::vector<double> latencies_us;
            latencies_us.reserve(benchmark_iterations);

            for (std::size_t iter_idx = 0; iter_idx < benchmark_iterations; ++iter_idx) {
                const auto start = std::chrono::steady_clock::now();
                torch::Tensor iter_output = run_forward(module, inputs);
                const auto finish = std::chrono::steady_clock::now();

                benchmark_sink += iter_output[0].item<float>();

                const auto elapsed = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(finish - start);
                latencies_us.push_back(elapsed.count());
            }

            const double total_us = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
            const auto minmax = std::minmax_element(latencies_us.begin(), latencies_us.end());
            std::vector<double> sorted_latencies = latencies_us;
            std::sort(sorted_latencies.begin(), sorted_latencies.end());

            std::cout << "Benchmark mode: ENABLED\n";
            std::cout << "warmup_iterations: " << warmup_iterations << "\n";
            std::cout << "benchmark_iterations: " << benchmark_iterations << "\n";
            std::cout << "mean_us: " << std::fixed << std::setprecision(4)
                      << (total_us / static_cast<double>(benchmark_iterations)) << "\n";
            std::cout << "min_us: " << std::fixed << std::setprecision(4) << *minmax.first << "\n";
            std::cout << "p50_us: " << std::fixed << std::setprecision(4)
                      << percentile_from_sorted(sorted_latencies, 50.0) << "\n";
            std::cout << "p95_us: " << std::fixed << std::setprecision(4)
                      << percentile_from_sorted(sorted_latencies, 95.0) << "\n";
            std::cout << "max_us: " << std::fixed << std::setprecision(4) << *minmax.second << "\n";
            std::cout << "benchmark_sink: " << std::fixed << std::setprecision(6) << benchmark_sink << "\n";
        }

        return 0;
    } catch (const c10::Error& error) {
        std::cerr << "LibTorch error: " << error.what() << "\n";
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
