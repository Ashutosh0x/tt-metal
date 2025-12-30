// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_ternary_sfpu_addcmul.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise addcmul operation with the three inputs: y = addcmul(x0,x1,x2,value)
 * Output overwrites odst in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 *
 * | Argument              | Description                                                              | Type     | Valid Range                                           | Required |
 * |-----------------------|--------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0                 | The index of the tile in DST register buffer to use as condition operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1                 | The index of the tile in DST register buffer to use as first operand     | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst2                 | The index of the tile in DST register buffer to use as second operand    | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | value                 | The value to add to the product of the three inputs                       | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst                  | The index of the tile in DST register buffer to use as output            | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void addcmul_tile(uint32_t idst0, uint32_t idst1, uint32_t idst2, uint32_t odst, uint32_t value) {
    MATH((llk_math_eltwise_ternary_sfpu_addcmul<APPROX, DataFormat::Float16_b>(idst0, idst1, idst2, odst, value)));
}

ALWI void addcmul_fp32_tile(uint32_t idst0, uint32_t idst1, uint32_t idst2, uint32_t odst, uint32_t value) {
    MATH((llk_math_eltwise_ternary_sfpu_addcmul<APPROX, DataFormat::Float32>(idst0, idst1, idst2, odst, value)));
}

ALWI void addcmul_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t idst2, uint32_t odst, uint32_t value) {
    MATH((llk_math_eltwise_ternary_sfpu_addcmul<APPROX, DataFormat::Int32>(idst0, idst1, idst2, odst, value)));
}

ALWI void addcmul_uint32_tile(uint32_t idst0, uint32_t idst1, uint32_t idst2, uint32_t odst, uint32_t value) {
    MATH((llk_math_eltwise_ternary_sfpu_addcmul<APPROX, DataFormat::UInt32>(idst0, idst1, idst2, odst, value)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void addcmul_tile_init() { MATH((llk_math_eltwise_ternary_sfpu_addcmul_init<APPROX>())); }

}  // namespace ckernel
