// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/format.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn_fixed/matmuls.hpp"

int main() {
    // Tensor shapes:
    // a: Shape([16384, 1536]) -> expanded to 4D: [1, 1, 16384, 1536]
    // b: Shape([1, 1, 1536, 384])
    // Result: [1, 1, 16384, 384]

    constexpr uint32_t M = 16384;
    constexpr uint32_t K = 1536;
    constexpr uint32_t N = 384;

    auto* device = &ttml::autograd::ctx().get_device();

    // Create tensor a with shape [1, 1, 16384, 1536]
    std::vector<float> a_data(M * K, 1.0F);  // Fill with 1.0 for simplicity
    auto a = ttml::core::from_vector(a_data, ttnn::Shape({1, 1, M, K}), device);

    // Create tensor b with shape [1, 1, 1536, 384]
    std::vector<float> b_data(K * N, 1.0F);  // Fill with 1.0 for simplicity
    auto b = ttml::core::from_vector(b_data, ttnn::Shape({1, 1, K, N}), device);

    fmt::print("Tensor a shape: {}\n", a.logical_shape());
    fmt::print("Tensor b shape: {}\n", b.logical_shape());

    // Perform matrix multiplication using ttnn_fixed::matmul
    auto result = ttml::ttnn_fixed::matmul(a, b);

    fmt::print("Result shape: {}\n", result.logical_shape());

    // Verify the result by reading a few values
    auto result_vec = ttml::core::to_vector(result);
    fmt::print("Result size: {}\n", result_vec.size());
    fmt::print("Expected size: {}\n", M * N);

    // With all 1s, each element should be K (1536)
    fmt::print("First element (expected {}): {}\n", K, result_vec[0]);
    fmt::print("Last element (expected {}): {}\n", K, result_vec[M * N - 1]);

    fmt::print("Matrix multiplication completed successfully!\n");

    return 0;
}
