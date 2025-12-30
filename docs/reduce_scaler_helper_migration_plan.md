# Reduce Scaler Helper Library Migration Plan

## Overview

This document outlines the plan to unify and consolidate the various reduce scaler generation functions scattered across the tt-metal codebase into a single, well-documented kernel helper library.

## Analysis Summary

### What is a Reduce Scaler?

A reduce scaler is a scaling factor applied during reduction operations (particularly for AVG pooling) to normalize the result. It's typically:
- A `float` value converted to `bfloat16` and packed into a `uint32_t`
- For SUM/MAX reductions: typically `1.0f`
- For AVG reduction: `1/N` where N is the number of elements being averaged

The scaler is placed in a circular buffer tile with a specific fill pattern that the reduction hardware expects.

---

## Existing Helpers (Scattered Locations)

| Helper | Location | Purpose | Migration |
|--------|----------|---------|-----------|
| `generate_reduce_scaler<half_tile>` | `ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp` | Fill first row (8 elements) of each face with scaler | → `generate_scaler_tile<ReduceRowFull>` |
| `generate_mm_scaler` | `ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_mm_scaler.hpp` | Fill first column of top 2 faces | → `generate_scaler_tile<ReduceCol>` |
| `generate_partial_reduce_scaler` | `ttnn/cpp/ttnn/operations/normalization/kernel_util/dataflow/custom_tiles.h` | Fill only N columns with scaler | **Keep as-is** (runtime num_cols) |

### Out of Scope (Not Reduce Scalers)

The following functions are **general-purpose broadcast fill utilities**, not reduce scalers, and are excluded from this migration:

| Helper | Location | Actual Purpose | Why Excluded |
|--------|----------|----------------|--------------|
| `fill_cb_with_value` | `moreh_common.hpp` | Fill tile with optimizer hyperparameters (lr, beta1, eps, etc.) | Used for broadcast constants in Adam/SGD/etc., not reduction scaling |
| `fill_with_val_bfloat16` | `fill_tile_utils.hpp` | Fill tile for binary_ng broadcast operations | Part of eltwise broadcast utilities (row/col/scalar broadcast) |

### Fill Patterns Explained

#### 1. REDUCE_ROW Pattern (`generate_reduce_scaler`)
Used for row-wise reductions. Fills first row (8 uint32 elements = 16 bf16 values) of each of the 4 faces:
```
Face layout (each face is 16x16 elements):
[S S S S S S S S 0 0 0 0 0 0 0 0]  <- Row 0: first 8 elements filled
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
...
```

#### 2. REDUCE_COL Pattern (`generate_mm_scaler`)
Used for matrix multiply scaling. Fills first column of top 2 faces only:
```
Face 0 & 1: [S 0 0 0 ...] for each row
Face 2 & 3: [0 0 0 0 ...] (zeros)
```

#### 3. PARTIAL_ROW Pattern (`generate_partial_reduce_scaler`) - Out of Scope
Fills only the first N columns of each row with the scaler value. Used when reducing partial tiles.
**Not migrated** - requires runtime `num_cols`, will continue using existing `custom_tiles.h` implementation.

---

## Inline Implementations Found (Need Migration)

### Direct Duplicates

1. **`ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/reader_moreh_mean_h.cpp:27-52`**
   - Exact duplicate of `generate_reduce_scaler` logic
   - Lines 30-52 can be replaced with single function call

### Kernels Using Deprecated Helpers

Total of **36 files** reference `reduce_scaler` patterns:

**Reduction Operations:**
- `reader_tilize_untilize_interleaved.cpp`
- `reader_unary_reduce_universal_start_id.cpp`
- `reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp`
- `reader_unary_transpose_wh_universal_input_cols_partitioned.cpp`

**Transformer/SDPA:**
- `writer_decode_all.cpp`
- `writer_windowed.cpp`
- `joint_writer.cpp`
- `ring_joint_writer.cpp`
- `writer_interleaved.cpp` (multiple)

**Normalization:**
- `reader_unary_interleaved_sm.cpp`
- `reader_unary_interleaved_sm_large_tensor.cpp`
- `reader_unary_sharded_sm.cpp`
- `reader_unary_sharded_sm_causal_mask_hw_dims.cpp`
- `reader_unary_sharded_sm_rm_mask.cpp`
- `reader_unary_interleaved_ln.cpp`
- `reader_unary_interleaved_ln_rm_gb.cpp`
- `reader_unary_interleaved_ln_large_tensor.cpp`
- `writer_unary_sharded_ln.cpp`
- `writer_unary_sharded_ln_rm_gb.cpp`
- `writer_unary_sharded_ln_pre_all_gather.cpp`
- `welford_writer_unary_gn_rm_gb.cpp`
- `writer_unary_gn_rm_gb.cpp`

**LayerNorm Distributed:**
- `reader_layernorm_preallgather_2d.cpp`
- `reader_unary_interleaved_ln_rm_gb_post_allgather.cpp`
- `reader_unary_interleaved_ln_rm_gb_pre_allgather.cpp`

**Moreh Operations:**
- `reader_moreh_sum_h.cpp`
- `reader_moreh_sum_nc.cpp`
- `reader_moreh_bias_backward_h.cpp`

**Experimental:**
- `rms_post_allgather_reader.cpp`
- `rms_pre_allgather_reader.cpp`
- `reader_reduce_nc.cpp`
- `reader_ssm_1d_sum_reduce.cpp`
- `rms_writer.cpp`

---

## Proposed Unified Helper Library

### Location
`ttnn/cpp/ttnn/operations/kernel_lib/dataflow/scaler_tiles.hpp`

### Proposed API

```cpp
#pragma once
#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/constants.hpp>

namespace ttnn::kernel_lib::dataflow {

/**
 * @brief Generate a scalar tile for reduction operations
 *
 * Creates a tile in the specified circular buffer with the scaler value
 * placed according to the fill pattern. The tile is first zeroed, then
 * the appropriate positions are filled with the scaler.
 *
 * @tparam Region A FillRegion type specifying the fill pattern (default: ReduceRowFull)
 * @tparam zero_first If true, zero the tile before filling (default: true)
 * @param cb_id Circular buffer ID to write the tile to
 * @param scaler Packed 16-bit value (bf16 << 16 | bf16)
 *
 * @note The function handles cb_reserve_back and cb_push_back internally
 * @note For ReduceRowFull/ReduceRowHalf: fills positions [0-7] of each face's first row
 * @note For ReduceCol: fills first column of faces 0 and 1 only
 */
template <typename Region = ReduceRowFull, bool zero_first = true>
FORCE_INLINE void generate_scaler_tile(uint32_t cb_id, uint32_t scaler);

} // namespace ttnn::kernel_lib::dataflow
```

### Unified Function Implementation

Below is a generalized implementation that eliminates magic numbers by:
1. Defining tile/face layout constants
2. Using a single `fill_tile_region` function that all patterns call with different parameters

#### Tile Memory Layout

```
A 32x32 tile consists of 4 faces (16x16 each), stored in memory as:

┌─────────────────┬─────────────────┐
│     Face 0      │     Face 1      │   Top row of faces
│   (256 elem)    │   (256 elem)    │
├─────────────────┼─────────────────┤
│     Face 2      │     Face 3      │   Bottom row of faces
│   (256 elem)    │   (256 elem)    │
└─────────────────┴─────────────────┘

Memory layout (contiguous):
[Face0: 256 u16][Face1: 256 u16][Face2: 256 u16][Face3: 256 u16]

Within each face (16x16), rows are stored contiguously:
[Row0: 16 u16][Row1: 16 u16]...[Row15: 16 u16]

When using uint32_t* (2 bf16 packed per u32):
- Face size = 128 u32
- Row size = 8 u32
- Face offsets: 0, 128, 256, 384
```

#### Generalized Implementation

```cpp
#pragma once

#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/constants.hpp>

namespace ttnn::kernel_lib::dataflow {

//==============================================================================
// Tile Layout Constants
//==============================================================================

namespace tile_layout {

// Tile dimensions
constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_WIDTH = 32;
constexpr uint32_t TILE_HW = TILE_HEIGHT * TILE_WIDTH;  // 1024 elements

// Face dimensions (tile is 2x2 faces)
constexpr uint32_t FACE_HEIGHT = 16;
constexpr uint32_t FACE_WIDTH = 16;
constexpr uint32_t FACE_HW = FACE_HEIGHT * FACE_WIDTH;  // 256 elements per face
constexpr uint32_t NUM_FACES = 4;
constexpr uint32_t FACES_PER_ROW = 2;  // Faces 0,1 on top; Faces 2,3 on bottom

// Sizes in different pointer types
constexpr uint32_t TILE_SIZE_BYTES = TILE_HW * sizeof(uint16_t);      // 2048 bytes
constexpr uint32_t HALF_TILE_SIZE_BYTES = TILE_SIZE_BYTES / 2;        // 1024 bytes
constexpr uint32_t FACE_SIZE_U16 = FACE_HW;                           // 256 uint16 per face
constexpr uint32_t FACE_SIZE_U32 = FACE_HW / 2;                       // 128 uint32 per face
constexpr uint32_t ROW_SIZE_U16 = FACE_WIDTH;                         // 16 uint16 per row
constexpr uint32_t ROW_SIZE_U32 = FACE_WIDTH / 2;                     // 8 uint32 per row

// Face start offsets (in uint32)
constexpr uint32_t FACE_OFFSETS_U32[NUM_FACES] = {
    0 * FACE_SIZE_U32,   // Face 0: offset 0
    1 * FACE_SIZE_U32,   // Face 1: offset 128
    2 * FACE_SIZE_U32,   // Face 2: offset 256
    3 * FACE_SIZE_U32    // Face 3: offset 384
};

}  // namespace tile_layout

//==============================================================================
// Fill Region Specification
//==============================================================================

/**
 * Compile-time specification of which region of the tile to fill.
 * All patterns can be expressed as: "fill columns [start_col, start_col+num_cols)
 * of rows [start_row, start_row+num_rows) in faces [start_face, start_face+num_faces)"
 */
template <
    uint32_t StartFace,    // First face to fill (0-3)
    uint32_t NumFaces,     // Number of consecutive faces to fill (1-4)
    uint32_t StartRow,     // First row within each face (0-15)
    uint32_t NumRows,      // Number of rows to fill per face (1-16)
    uint32_t StartCol,     // First column within each row (0-7 for u32)
    uint32_t NumCols       // Number of columns to fill per row (1-8 for u32)
>
struct FillRegion {
    static_assert(StartFace < tile_layout::NUM_FACES, "StartFace must be < 4");
    static_assert(NumFaces > 0 && StartFace + NumFaces <= tile_layout::NUM_FACES, "Invalid face range");
    static_assert(StartRow < tile_layout::FACE_HEIGHT, "StartRow must be < 16");
    static_assert(NumRows > 0 && StartRow + NumRows <= tile_layout::FACE_HEIGHT, "Invalid row range");
    static_assert(StartCol < tile_layout::ROW_SIZE_U32, "StartCol must be < 8");
    static_assert(NumCols > 0 && StartCol + NumCols <= tile_layout::ROW_SIZE_U32, "Invalid col range");

    static constexpr uint32_t start_face = StartFace;
    static constexpr uint32_t num_faces = NumFaces;
    static constexpr uint32_t start_row = StartRow;
    static constexpr uint32_t num_rows = NumRows;
    static constexpr uint32_t start_col = StartCol;
    static constexpr uint32_t num_cols = NumCols;

    // Computed: total elements to fill
    static constexpr uint32_t total_elements = NumFaces * NumRows * NumCols;
    // Is this a full tile fill?
    static constexpr bool is_full_tile = (total_elements == tile_layout::TILE_HW / 2);
};

//==============================================================================
// Predefined Fill Patterns for Reduce Scalers
//==============================================================================

// REDUCE_ROW: First row of each face, all columns
// Used by: generate_reduce_scaler (standard reduction)
using ReduceRowFull = FillRegion<0, 4, 0, 1, 0, 8>;   // All 4 faces
using ReduceRowHalf = FillRegion<0, 2, 0, 1, 0, 8>;   // First 2 faces (half tile)

// REDUCE_COL: First column of top 2 faces, all rows
// Used by: generate_mm_scaler (matrix multiply)
using ReduceCol = FillRegion<0, 2, 0, 16, 0, 1>;      // Faces 0-1, all rows, col 0

//==============================================================================
// Core Implementation
//==============================================================================

namespace detail {

/**
 * Zero-fill faces using NOC reads from MEM_ZEROS_BASE
 * Calculates bytes to zero based on the region's face coverage.
 */
template <typename Region>
FORCE_INLINE void zero_fill_region(uint32_t write_addr) {
    // Calculate bytes to zero: number of faces * bytes per face
    constexpr uint32_t bytes_to_zero = Region::num_faces * tile_layout::FACE_HW * sizeof(uint16_t);
    constexpr uint32_t num_zeros_reads = bytes_to_zero / MEM_ZEROS_SIZE;
    static_assert(num_zeros_reads > 0, "num_zeros_reads must be greater than 0");

    // Start at the correct face offset
    constexpr uint32_t start_offset = Region::start_face * tile_layout::FACE_HW * sizeof(uint16_t);
    write_addr += start_offset;

    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    noc_async_read_one_packet_set_state(zeros_noc_addr, MEM_ZEROS_SIZE);

    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        noc_async_read_one_packet_with_state(zeros_noc_addr, write_addr);
        write_addr += MEM_ZEROS_SIZE;
    }
    noc_async_read_barrier();
}

/**
 * Generic fill function - fills the region specified by FillRegion template.
 * Works for all patterns including full tile (compiler optimizes the loops).
 */
template <typename Region>
FORCE_INLINE void fill_region(volatile tt_l1_ptr uint32_t* ptr, uint32_t scaler) {
    if (scaler == 0) return;

    #pragma unroll
    for (uint32_t face = Region::start_face; face < Region::start_face + Region::num_faces; ++face) {
        const uint32_t face_offset = tile_layout::FACE_OFFSETS_U32[face];

        #pragma unroll
        for (uint32_t row = Region::start_row; row < Region::start_row + Region::num_rows; ++row) {
            const uint32_t row_offset = face_offset + row * tile_layout::ROW_SIZE_U32;

            #pragma unroll
            for (uint32_t col = Region::start_col; col < Region::start_col + Region::num_cols; ++col) {
                ptr[row_offset + col] = scaler;
            }
        }
    }
}

}  // namespace detail

//==============================================================================
// Public API
//==============================================================================

/**
 * @brief Generate a scaler tile with the specified fill region
 *
 * @tparam Region A FillRegion type specifying which part of the tile to fill
 * @tparam zero_first If true, zero the tile before filling (default: true)
 * @param cb_id Circular buffer ID
 * @param scaler Packed 16-bit scaler value (bf16 << 16 | bf16)
 */
template <typename Region, bool zero_first = true>
FORCE_INLINE void generate_scaler_tile(uint32_t cb_id, uint32_t scaler) {
    cb_reserve_back(cb_id, 1);
    uint32_t write_addr = get_write_ptr(cb_id);

    // Zero-fill first (skip for full tile fill since we overwrite everything)
    if constexpr (zero_first && !Region::is_full_tile) {
        detail::zero_fill_region<Region>(write_addr);
    }

    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);
    detail::fill_region<Region>(ptr, scaler);

    cb_push_back(cb_id, 1);
}

}  // namespace ttnn::kernel_lib::dataflow
```

#### Key Improvements

1. **Named constants** in `tile_layout` namespace - no magic numbers
2. **`FillRegion` template** - compile-time specification of what to fill
3. **Predefined patterns** as type aliases (`ReduceRowFull`, `ReduceRowHalf`, `ReduceCol`)
4. **Single `fill_region` function** - handles all patterns with the same code
5. **Static assertions** - catch invalid fill specs at compile time
6. **Extensible** - add new patterns by defining new `FillRegion` types

**Example - custom pattern:**
```cpp
// Fill middle 4 columns of rows 8-15 in faces 2-3
using CustomRegion = FillRegion<2, 2, 8, 8, 2, 4>;
generate_scaler_tile<CustomRegion>(cb_id, scaler);
```

**Out of scope:** `generate_partial_reduce_scaler` (runtime `num_cols`) remains in `custom_tiles.h`.

### Usage Examples

**Before (using deprecated helper):**
```cpp
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    // ...
    generate_reduce_scaler(cb_id_in2, scaler);
}
```

**After (using unified helper):**
```cpp
#include "ttnn/cpp/ttnn/operations/kernel_lib/dataflow/scaler_tiles.hpp"

using namespace ttnn::kernel_lib::dataflow;

void kernel_main() {
    // ...
    generate_scaler_tile<ReduceRowFull>(cb_id_in2, scaler);
}
```

**All available patterns:**
```cpp
using namespace ttnn::kernel_lib::dataflow;

// Standard reduce scaler - first row of all 4 faces
generate_scaler_tile<ReduceRowFull>(cb_id, scaler);

// Half tile reduce scaler - first row of first 2 faces only
generate_scaler_tile<ReduceRowHalf>(cb_id, scaler);

// Matrix multiply scaler - first column of top 2 faces
generate_scaler_tile<ReduceCol>(cb_id, scaler);

// Custom pattern - define your own FillRegion if needed
using MyCustomRegion = FillRegion<0, 2, 0, 8, 2, 4>;  // faces 0-1, rows 0-7, cols 2-5
generate_scaler_tile<MyCustomRegion>(cb_id, scaler);
```

**Note:** `generate_partial_reduce_scaler` (runtime `num_cols`) is NOT included in this library.
Kernels using it will continue to use the existing implementation in `custom_tiles.h`.

### Design Trade-offs

| Approach | Pros | Cons |
|----------|------|------|
| **Unified template function** | Single entry point, compile-time pattern selection, no runtime overhead, extensible | Requires understanding template syntax |
| **Separate functions** | Simpler to understand, matches current design | Code duplication, harder to add new patterns |

The unified approach was chosen because:
1. **Zero runtime overhead** - pattern selection is compile-time via templates
2. **Extensibility** - adding a new pattern only requires defining a new `FillRegion` type alias
3. **Discoverability** - all predefined patterns are listed as type aliases in one place
4. **No code duplication** - single `fill_region` function handles all patterns

---

## Migration Plan

### Tier 1: Easy (Direct replacement, no logic changes)

These kernels simply include the deprecated header and call `generate_reduce_scaler`. Migration is mechanical.

| Kernel | File Path | Notes |
|--------|-----------|-------|
| `reader_unary_transpose_wh_universal_input_cols_partitioned.cpp` | `reduction/generic/device/kernels/dataflow/` | Simple include swap |
| `reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp` | `reduction/generic/device/kernels/dataflow/` | Simple include swap |
| `reader_unary_reduce_universal_start_id.cpp` | `reduction/generic/device/kernels/dataflow/` | Also uses `generate_mm_scaler` conditionally |
| `reader_tilize_untilize_interleaved.cpp` | `reduction/tilize_untilize/device/kernels/dataflow/` | Conditional on `REDUCE_OP` define |
| `reader_reduce_nc.cpp` | `experimental/reduction/fast_reduce_nc/device/kernels/` | Simple include swap |
| `joint_writer.cpp` | `transformer/sdpa/device/kernels/dataflow/` | Simple include swap |
| `ring_joint_writer.cpp` | `transformer/sdpa/device/kernels/dataflow/` | Simple include swap |
| `writer_interleaved.cpp` (SDPA) | `transformer/sdpa/device/kernels/dataflow/` | Simple include swap |
| `reader_moreh_sum_h.cpp` | `moreh/moreh_sum/device/moreh_sum_h_impl_kernels/` | Simple include swap |
| `reader_ssm_1d_sum_reduce.cpp` | `experimental/ssm/hc_sum_reduce/device/kernels/` | Simple include swap |

**Estimated effort:** 10 files, ~30 minutes

### Tier 2: Moderate (Inline code to migrate or multiple helpers)

| Kernel | File Path | Current State | Migration Notes |
|--------|-----------|---------------|-----------------|
| `reader_moreh_mean_h.cpp` | `moreh/moreh_mean/device/kernels/` | **Inline implementation** | Replace lines 27-52 with function call |
| `reader_unary_sharded_sm.cpp` | `normalization/softmax/device/kernels/attention/dataflow/` | Multiple conditional paths | Test all code paths |
| `reader_unary_interleaved_ln_rm_gb.cpp` | `normalization/layernorm/device/kernels/dataflow/` | Uses `generate_reduce_scaler` | Migrate reduce_scaler only |
| All softmax readers | `normalization/softmax/device/kernels/attention/dataflow/` | Various conditional compilation | Careful testing required |

**Estimated effort:** 14 files, ~2-3 hours

### Tier 3: Complex (Architecture-specific or specialized)

| Kernel | File Path | Notes |
|--------|-----------|-------|
| `custom_tiles.h` | `normalization/kernel_util/dataflow/` | Keep `generate_partial_reduce_scaler` - specialized partial column fill with runtime `num_cols` |

**Estimated effort:** No migration needed for this tier

---

## Recommended Migration Order

### Phase 1: Create Library & Migrate Easiest Cases (Day 1)

1. **Create the helper library header**
   - `ttnn/cpp/ttnn/operations/kernel_lib/dataflow/scaler_tiles.hpp`
   - Copy implementation from `generate_reduce_scaler.hpp`
   - Add proper documentation

2. **Migrate `reader_moreh_mean_h.cpp`** (First win - inline code)
   - Replace lines 27-52 with single function call
   - Verify test passes: `pytest tests/ttnn/unit_tests/operations/moreh/test_moreh_mean.py -v`

3. **Migrate 5 simple reduction kernels**
   - Update includes, verify no behavior change

### Phase 2: Migrate SDPA & More Reductions (Day 2)

4. **Migrate SDPA kernels** (6 files)
   - `joint_writer.cpp`, `ring_joint_writer.cpp`, `writer_interleaved.cpp`, etc.
   - Run SDPA tests

5. **Migrate remaining reduction kernels**
   - `reader_reduce_nc.cpp`, `reader_ssm_1d_sum_reduce.cpp`

### Phase 3: Migrate Normalization Kernels (Day 3-4)

6. **Migrate softmax kernels** (5 files)
   - Multiple conditional compilation paths - test thoroughly

7. **Migrate LayerNorm kernels** (6 files)
   - Includes both sharded and interleaved variants

8. **Migrate GroupNorm kernels** (2 files)
   - Only kernels using `generate_reduce_scaler`

### Phase 4: Deprecate Old Headers (Day 5)

9. **Add deprecation warnings to old helpers**
    ```cpp
    [[deprecated("Use ttnn::kernel_lib::dataflow::generate_scaler_tile instead")]]
    ```

10. **Update documentation**
    - Add to CLAUDE.md if appropriate
    - Create usage examples

---

## Testing Strategy

### Unit Tests to Run After Each Migration

```bash
# Reduction operations
pytest tests/ttnn/unit_tests/operations/reduction/ -v

# SDPA
pytest tests/ttnn/unit_tests/operations/transformer/ -v -k sdpa

# Normalization
pytest tests/ttnn/unit_tests/operations/normalization/ -v

# Moreh operations
pytest tests/ttnn/unit_tests/operations/moreh/ -v

# Full sweep (after all migrations)
pytest tests/ttnn/unit_tests/operations/ -v
```

### Specific Tests for Key Operations

```bash
# Moreh mean (first migration target)
pytest tests/ttnn/unit_tests/operations/moreh/test_moreh_mean.py -v

# LayerNorm
pytest tests/ttnn/unit_tests/operations/normalization/test_layernorm.py -v

# Softmax
pytest tests/ttnn/unit_tests/operations/normalization/test_softmax.py -v
```

---

## Start Point Recommendation

**Start with `reader_moreh_mean_h.cpp`** because:

1. Has inline code that's an exact duplicate of the helper (lines 27-52)
2. Self-contained - no conditional compilation affecting the scaler generation
3. Low risk - isolated moreh operation
4. Proves the library works before tackling more complex cases
5. Clear test to validate: `test_moreh_mean.py`

---

## File Changes Summary

### New Files
- `ttnn/cpp/ttnn/operations/kernel_lib/dataflow/scaler_tiles.hpp`

### Files to Modify (36 total)
- See "Kernels Using Deprecated Helpers" section above

### Files to Deprecate (Eventually)
- `ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp`
- `ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_mm_scaler.hpp`

### Files NOT Modified (Out of Scope)
- `ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp` - `fill_cb_with_value` is for broadcast constants, not reduce scalers
- `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp` - broadcast utilities for binary ops
