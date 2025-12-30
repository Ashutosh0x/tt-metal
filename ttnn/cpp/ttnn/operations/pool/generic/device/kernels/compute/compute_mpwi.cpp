// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/add_int_sfpu.h"
#include "compute_kernel_api/copy_dest_values.h"

#define DEBUG_PRINT 1

#if DEBUG_PRINT == 1
#include "api/debug/dprint.h"
#include "api/debug/dprint_pages.h"
#include "api/debug/dprint_tensix.h"
#include "tools/profiler/kernel_profiler.hpp"
#endif

#define ALWI inline __attribute__((always_inline))

#define FACE_HEIGHT 16
#define FACE_WIDTH 16
#define TILE_HEIGHT 32
#define TILE_WIDTH 32

namespace NAMESPACE {

void MAIN {
    // NOTE: here it is assumed that in_ntiles_hw == 1. General cases not handled yet. When ntiles_hw > 1 the large
    // kernel is called
    constexpr uint32_t in_ntiles_c = get_compile_time_arg_val(0);
    constexpr uint32_t window_size_hw = get_compile_time_arg_val(1);

    constexpr uint32_t split_reader = get_compile_time_arg_val(2);

    constexpr uint32_t max_out_sticks_per_core = get_compile_time_arg_val(3);
    constexpr uint32_t in_c = get_compile_time_arg_val(4);
    constexpr uint32_t in_nblocks_c = get_compile_time_arg_val(5);
    constexpr uint32_t max_sticks_for_reduction = get_compile_time_arg_val(6);

    constexpr uint32_t in_cb_id_0 = get_compile_time_arg_val(7);
    constexpr uint32_t in_cb_id_1 = get_compile_time_arg_val(8);  // for split reader
    constexpr uint32_t in_scalar_cb_id_0 = get_compile_time_arg_val(9);
    constexpr uint32_t in_scalar_cb_id_1 = get_compile_time_arg_val(10);
    constexpr uint32_t in_idx_cb_id = get_compile_time_arg_val(11);
    constexpr uint32_t pack_tmp_cb_id = get_compile_time_arg_val(12);
    constexpr uint32_t pack_idx_tmp_cb_id = get_compile_time_arg_val(13);
    constexpr uint32_t right_inc_cb_id = get_compile_time_arg_val(14);
    constexpr uint32_t down_left_wrap_inc_cb_id = get_compile_time_arg_val(15);
    constexpr uint32_t up_left_wrap_inc_cb_id = get_compile_time_arg_val(16);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(17);
    constexpr uint32_t out_idx_cb_id = get_compile_time_arg_val(18);
    constexpr bool one_scalar_per_core = get_compile_time_arg_val(19);
    constexpr uint32_t pre_tilize_cb_id = get_compile_time_arg_val(20);
    constexpr bool is_output_tiled = get_compile_time_arg_val(21);  // 1 = TILED, 0 = ROW_MAJOR
    constexpr bool is_output_block_format = (bool)get_compile_time_arg_val(22);
    constexpr bool return_indices = (bool)get_compile_time_arg_val(23);
    constexpr uint32_t stride_h = get_compile_time_arg_val(24);
    constexpr uint32_t stride_w = get_compile_time_arg_val(25);
    constexpr uint32_t in_h_padded = get_compile_time_arg_val(26);
    constexpr uint32_t in_w_padded = get_compile_time_arg_val(27);
    constexpr uint32_t eff_kernel_h = get_compile_time_arg_val(28);
    constexpr uint32_t eff_kernel_w = get_compile_time_arg_val(29);
    constexpr uint32_t pad_l = get_compile_time_arg_val(30);
    constexpr uint32_t intra_kernel_right_inc_cb_id = get_compile_time_arg_val(31);
    constexpr uint32_t intra_kernel_down_left_wrap_inc_cb_id = get_compile_time_arg_val(32);
    constexpr uint32_t compute_tmp_idx_cb_id = get_compile_time_arg_val(33);
    constexpr uint32_t zero_inc_cb_id = get_compile_time_arg_val(34);
    constexpr uint32_t kernel_h = get_compile_time_arg_val(35);
    constexpr uint32_t kernel_w = get_compile_time_arg_val(36);

    constexpr uint32_t mpwi_cb_tile_idx = 0;
    constexpr uint32_t data_dst_idx = 0;
    // dst 1 used for accumulation for large kernel
    constexpr uint32_t index_dst_idx = 2;
    // dst 3 used for accumulation for large kernel
    constexpr uint32_t inc_dst_idx = 4;
    constexpr uint32_t index_scratch_out_dst_idx = 6;
    constexpr uint32_t index_temp_dst_idx = 5;  // only used for large kernels
    constexpr uint32_t zero_dst_idx = 7;

    constexpr uint32_t face_r_dim = FACE_HEIGHT;
    constexpr bool last_tile_is_partial = in_c % TILE_WIDTH != 0;
    constexpr uint32_t num_faces_in_input_tile = 4;
    constexpr uint32_t num_faces_in_output_tile = 2;
    constexpr uint32_t num_faces_in_last_output_tile = last_tile_is_partial && in_c % TILE_WIDTH <= FACE_WIDTH ? 1 : 2;
    constexpr uint32_t num_out_sticks = 1;

    // average pool with large kernels requires fp32 accumulation so we can only reduce 4 tiles at a time,
    // otherwise we can reduce 8 tiles at a time.
    constexpr bool is_large_kernel = window_size_hw > max_sticks_for_reduction;
    constexpr uint32_t MAX_TILES_PER_REDUCTION = 1;
    constexpr uint32_t max_tiles_per_iter =
        in_ntiles_c < MAX_TILES_PER_REDUCTION ? in_ntiles_c : MAX_TILES_PER_REDUCTION;
    constexpr uint32_t partial_iter_output_tiles =
        in_ntiles_c % MAX_TILES_PER_REDUCTION == 0 ? max_tiles_per_iter : in_ntiles_c % MAX_TILES_PER_REDUCTION;

    static_assert(REDUCE_OP == PoolType::MAX, "Only supports REDUCE_OP = MAX");
    constexpr bool neginf_srca_maxpool = true;
    constexpr bool zero_srca_avgpool = false;

    constexpr uint32_t w_chunks = kernel_w % max_sticks_for_reduction == 0 ? kernel_w / max_sticks_for_reduction
                                                                           : kernel_w / max_sticks_for_reduction + 1;
    constexpr uint32_t interm_reduction_chunks = is_large_kernel ? w_chunks * kernel_h : 1;

    cb_wait_front(in_scalar_cb_id_0, 1);

    uint32_t current_idx_col;
    uint32_t current_idx_row;
    const uint16_t start_row = (uint16_t)get_arg_val<uint32_t>(1);
    const uint16_t start_col = (uint16_t)get_arg_val<uint32_t>(2);
    current_idx_col = start_col;
    current_idx_row = start_row;

    constexpr uint32_t sticks_per_chunk = kernel_w <= max_sticks_for_reduction ? kernel_w : max_sticks_for_reduction;
    cb_wait_front(right_inc_cb_id, 1);
    cb_wait_front(down_left_wrap_inc_cb_id, 1);
    cb_wait_front(up_left_wrap_inc_cb_id, 1);
    cb_wait_front(zero_inc_cb_id, 1);
    if (is_large_kernel) {
        cb_wait_front(intra_kernel_right_inc_cb_id, 1);
        cb_wait_front(intra_kernel_down_left_wrap_inc_cb_id, 1);
    }

    unary_op_init_common(in_cb_id_0, in_cb_id_0);
    max_reduce_with_indices_init<ckernel::DataLayout::ROW_MAJOR>();

    // if max out sticks is non-zero then this will be used as the number of out sticks for every core
    // otherwise the runtime args are referenced for core-specific number of out sticks, for Pool2D
    // runtime args are used while for grid sample the max out sticks is set
    uint32_t num_out_sticks_this_core = max_out_sticks_per_core ? max_out_sticks_per_core : get_arg_val<uint32_t>(0);

    DPRINT << "num_out_sticks_this_core: " << num_out_sticks_this_core << ENDL();
    DPRINT << "in_nblocks_c: " << in_nblocks_c << ENDL();
    DPRINT << "interm_reduction_chunks: " << interm_reduction_chunks << ENDL();

    uint32_t tilize_stick_counter = 0;
    uint32_t tilize_stick_total = 0;
    bool first_iteration = true;
    for (uint32_t n = 0; n < num_out_sticks_this_core; ++n) {
        const uint32_t curr_scalar_cb_id = in_scalar_cb_id_0;
        const uint32_t curr_in_cb_id = in_cb_id_0;
        for (uint32_t c_i = 0; c_i < in_nblocks_c; c_i++) {
            const bool last_c_block = c_i == in_nblocks_c - 1;
            const bool first_c_block = c_i == 0;

            tile_regs_acquire();
            uint32_t intra_kernel_h = 0;
            uint32_t intra_kernel_w = 0;
            if (first_iteration) {
                cb_wait_front(in_idx_cb_id, 1);
                copy_tile_to_dst_init_short(in_idx_cb_id);
                reconfig_data_format_srca(in_idx_cb_id);
                // UNPACK(tt::compute::common::print_full_tile(in_idx_cb_id));
                copy_tile(
                    in_idx_cb_id, mpwi_cb_tile_idx, index_dst_idx);  // move the initial indexes from the reader to DST
                cb_pop_front(in_idx_cb_id, 1);
                first_iteration = false;
            } else {
                cb_wait_front(compute_tmp_idx_cb_id, 1);
                copy_tile_to_dst_init_short(compute_tmp_idx_cb_id);
                reconfig_data_format_srca(compute_tmp_idx_cb_id);
                // UNPACK(tt::compute::common::print_full_tile(compute_tmp_idx_cb_id));
                copy_tile(
                    compute_tmp_idx_cb_id,
                    mpwi_cb_tile_idx,
                    index_dst_idx);  // move incremented indexes from compute back to DST
                cb_pop_front(compute_tmp_idx_cb_id, 1);
            }
            if constexpr (is_large_kernel) {
                copy_dest_values_init();
                copy_dest_values(index_dst_idx, index_temp_dst_idx);  // save base indices for this C block
            }

            for (uint32_t chunk = 0; chunk < interm_reduction_chunks; chunk++) {
                bool first_chunk = chunk == 0;
                bool last_chunk = chunk == interm_reduction_chunks - 1;

                cb_wait_front(curr_in_cb_id, 1);
                copy_tile_to_dst_init_short(curr_in_cb_id);
                reconfig_data_format_srca(curr_in_cb_id);
                copy_tile(curr_in_cb_id, mpwi_cb_tile_idx, data_dst_idx);

                // increments happen between every chunk within a C block, and between C blocks
                bool increment_needed = false;
                if (last_c_block && last_chunk) {  // increment for the next kernel position
                    increment_needed = true;
                    // update the current index column
                    if (current_idx_col + stride_w + eff_kernel_w > in_w_padded) {
                        // we reached the right edge, wrap down and to the left
                        current_idx_col = 0;
                        if (current_idx_row + stride_h + eff_kernel_h > in_h_padded) {
                            // we reached the bottom right corner, wrap to the top and to the left
                            current_idx_row = 0;
                            copy_tile_to_dst_init_short(up_left_wrap_inc_cb_id);
                            reconfig_data_format_srca(up_left_wrap_inc_cb_id);
                            copy_tile(up_left_wrap_inc_cb_id, mpwi_cb_tile_idx, inc_dst_idx);
                        } else {
                            current_idx_row += stride_h;
                            copy_tile_to_dst_init_short(down_left_wrap_inc_cb_id);
                            reconfig_data_format_srca(down_left_wrap_inc_cb_id);
                            copy_tile(down_left_wrap_inc_cb_id, mpwi_cb_tile_idx, inc_dst_idx);
                        }
                    } else {
                        // we are still in the same row, move to the right
                        current_idx_col += stride_w;
                        copy_tile_to_dst_init_short(right_inc_cb_id);
                        reconfig_data_format_srca(right_inc_cb_id);
                        copy_tile(right_inc_cb_id, mpwi_cb_tile_idx, inc_dst_idx);
                    }
                } else if (is_large_kernel) {  // only need to increment within C block if multiple chunks
                    if (!last_chunk) {         // increment for the next chunk within the same C block
                        increment_needed = true;
                        if (intra_kernel_w + sticks_per_chunk < kernel_w) {  // move right in this row
                            intra_kernel_w += sticks_per_chunk;
                            copy_tile_to_dst_init_short(intra_kernel_right_inc_cb_id);
                            reconfig_data_format_srca(intra_kernel_right_inc_cb_id);
                            copy_tile(intra_kernel_right_inc_cb_id, mpwi_cb_tile_idx, inc_dst_idx);
                        } else {  // move down to the next row
                            intra_kernel_w = 0;
                            intra_kernel_h += 1;
                            copy_tile_to_dst_init_short(intra_kernel_down_left_wrap_inc_cb_id);
                            reconfig_data_format_srca(intra_kernel_down_left_wrap_inc_cb_id);
                            copy_tile(intra_kernel_down_left_wrap_inc_cb_id, mpwi_cb_tile_idx, inc_dst_idx);
                        }
                    }
                }

                if (!increment_needed) {
                    // no increment needed, just copy back the original indexes - copy_dest_values does not work
                    copy_tile_to_dst_init_short(zero_inc_cb_id);
                    reconfig_data_format_srca(zero_inc_cb_id);
                    copy_tile(zero_inc_cb_id, mpwi_cb_tile_idx, inc_dst_idx);
                }

                // we allow overflow here for negative values as this only occurs in padding regions
                add_int_tile_init();
                add_uint16_tile(index_dst_idx, inc_dst_idx, index_scratch_out_dst_idx);

                // the max_reduce_with_indices LLK function only supports kernel_size=9, pending
                // https://github.com/tenstorrent/tt-metal/issues/28141 but, since for return_indices the in_cb is
                // oversized (equal to 1 tile), and since this CB is filled with padding values in the beginning of
                // the data movement kernel, it is possible to still use max_reduce_with_indices with kernel sizes
                // smaller than 9 as the excess sticks are just filled with padding values
                constexpr uint32_t max_mpwi_kernel_size = window_size_hw <= 9 ? 9 : 32;
                // TODO update SFPU to do DST accumulation
                // TODO use 9 stick version for large kernel if chunk size allows
                max_reduce_with_indices_init<ckernel::DataLayout::ROW_MAJOR>();
                max_reduce_with_indices<max_mpwi_kernel_size, ckernel::DataLayout::ROW_MAJOR, is_large_kernel>(
                    data_dst_idx, index_dst_idx, chunk);

                MATH(DPRINT << "REDUCE" << ENDL());

                if constexpr (is_large_kernel) {
                    if (!last_chunk) {
                        copy_dest_values_init();
                        copy_dest_values(index_scratch_out_dst_idx, index_dst_idx);
                    }
                }

                cb_pop_front(curr_in_cb_id, 1);
            }

            // After all chunks: if not last C block, restore base indices for next C block
            if constexpr (is_large_kernel) {
                if (!last_c_block) {
                    copy_dest_values_init();
                    copy_dest_values(
                        index_temp_dst_idx, index_scratch_out_dst_idx);  // restore base indices for next C block
                }
            }

            tile_regs_commit();
            tile_regs_wait();

            cb_reserve_back(pack_tmp_cb_id, 1);
            pack_reconfig_data_format(pack_tmp_cb_id);
            pack_tile<true>(data_dst_idx, pack_tmp_cb_id, mpwi_cb_tile_idx);  // for reader (output data)
            cb_push_back(pack_tmp_cb_id, 1);

            cb_reserve_back(pack_idx_tmp_cb_id, 1);
            pack_reconfig_data_format(pack_idx_tmp_cb_id);
            pack_tile<true>(index_dst_idx, pack_idx_tmp_cb_id, mpwi_cb_tile_idx);  // for reader (output indexes)
            cb_push_back(pack_idx_tmp_cb_id, 1);

            // Only push to compute_tmp_idx_cb_id if there's a next iteration that will consume it
            // This prevents leaving stale data in the CB between program runs when using caching
            bool is_last_iteration = (n == num_out_sticks_this_core - 1) && last_c_block;
            if (!is_last_iteration) {
                cb_reserve_back(compute_tmp_idx_cb_id, 1);
                pack_reconfig_data_format(compute_tmp_idx_cb_id);
                // dprint_tensix_dest_reg(index_scratch_out_dst_idx);
                pack_tile<true>(
                    index_scratch_out_dst_idx,
                    compute_tmp_idx_cb_id,
                    mpwi_cb_tile_idx);  // for compute (incremented indexes)
                cb_push_back(compute_tmp_idx_cb_id, 1);
            }

            tile_regs_release();
        }
    }
}

}  // namespace NAMESPACE
