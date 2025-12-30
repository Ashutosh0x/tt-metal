// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, DataFormat data_format, int ITERATIONS>
inline void calculate_addcmul(
    const uint dst_index_in0,
    const uint dst_index_in1,
    const uint dst_index_in2,
    const uint dst_index_out,
    const uint value) {
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b ||
            data_format == DataFormat::Int32 || data_format == DataFormat::UInt32,
        "Unsupported data format for calculate_addcmul(). Only Float32, Int32, UInt32, and Float16_b are allowed.");
    constexpr InstrModLoadStore mod0 = (data_format == DataFormat::Int32 || data_format == DataFormat::UInt32)
                                           ? InstrModLoadStore::INT32
                                       : (data_format == DataFormat::UInt16)    ? InstrModLoadStore::LO16
                                       : (data_format == DataFormat::Float32)   ? InstrModLoadStore::FP32
                                       : (data_format == DataFormat::Float16_b) ? InstrModLoadStore::FP16B
                                                                                : InstrModLoadStore::INT32;
    // size of each tile in Dest is 64 rows
    constexpr uint dst_tile_size = 64;
    // Hardcode 1.0 as uint32_t (IEEE 754 float32: 0x3F800000)
    // uint32_t hardcoded_value = 0x3F800000;
    // TT_SFPLOADI(p_sfpu::LREG3, 10, hardcoded_value & 0xFFFF);
    // TT_SFPLOADI(p_sfpu::LREG3, 8, hardcoded_value >> 16);
    TT_SFPLOADI(p_sfpu::LREG3, 10, value & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG3, 8, value >> 16);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, mod0, ADDR_MOD_7, dst_index_in0 * dst_tile_size);
        TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_7, dst_index_in1 * dst_tile_size);
        TT_SFPLOAD(p_sfpu::LREG2, mod0, ADDR_MOD_7, dst_index_in2 * dst_tile_size);
        TT_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);
        TTI_SFPNOP;
        TT_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG4, p_sfpu::LREG0, p_sfpu::LREG5, 0);
        TTI_SFPNOP;
        if constexpr (DATA_FORMAT == DataFormat::Float16_b) {
            TT_SFP_STOCH_RND(1, 0, 0, p_sfpu::LREG5, p_sfpu::LREG5, 1);
        }
        TT_SFPSTORE(p_sfpu::LREG5, mod0, ADDR_MOD_7, dst_index_out * dst_tile_size);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void init_addcmul() {
    // No initialization required
}
}  // namespace ckernel::sfpu
