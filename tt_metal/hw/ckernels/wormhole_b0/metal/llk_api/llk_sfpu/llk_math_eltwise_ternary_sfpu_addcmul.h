// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_ternary_sfpu_params.h"
#include "ckernel_sfpu_addcmul.h"

namespace ckernel {

template <bool APPROXIMATE, DataFormat data_format, int ITERATIONS = 8>
inline void llk_math_eltwise_ternary_sfpu_addcmul(
    uint dst_index0, uint dst_index1, uint dst_index2, uint odst, uint value, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_ternary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_addcmul<APPROXIMATE, data_format, ITERATIONS>,
        dst_index0,
        dst_index1,
        dst_index2,
        odst,
        vector_mode,
        value);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_ternary_sfpu_addcmul_init() {
    _llk_math_eltwise_ternary_sfpu_init_<SfpuType::addcmul>();
    ckernel::sfpu::init_addcmul<APPROXIMATE>();
}
}  // namespace ckernel
