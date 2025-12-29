// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rms_allgather_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/rms_allgather/rms_allgather.hpp"

namespace ttnn::operations::experimental::ccl {

void bind_fused_rms_norm(nb::module_& mod) {
    ttnn::bind_registered_operation(
        mod,
        ttnn::fused_rms_norm,
        R"doc(
Fused RMS Normalization with All-Gather for Multi-Device Execution
===================================================================

This operation performs Root Mean Square (RMS) Normalization fused with an optional residual addition
and multi-device all-gather synchronization. It is optimized for distributed inference across multiple
devices using fabric communication.

Mathematical Formulation
------------------------

RMS Normalization is computed as:

.. math::
    \text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}

.. math::
    \text{output} = \frac{x}{\text{RMS}(x)} \cdot \gamma = x \cdot \text{rsqrt}\left(\text{mean}(x^2) + \epsilon\right) \cdot \gamma

When optional residual addition is enabled:

.. math::
    x' = x + \text{residual}

.. math::
    \text{output} = x' \cdot \text{rsqrt}\left(\text{mean}(x'^2) + \epsilon\right) \cdot \gamma

The operation also updates the residual tensor in-place with ``x + residual`` when residual is provided.

Parallelization Strategy
------------------------

The operation is parallelized across:

1. **Multiple cores on a single device**: Using width sharding where each core processes a portion of the width dimension.
2. **Multiple devices**: Using fabric all-gather for statistics synchronization across devices.

The algorithm proceeds in phases:

1. **Pre-phase (local computation)**: Optional residual add, compute x², compute partial E(x²) via row reduction.
2. **All-to-all reduction phase**: Cores gather partial statistics from other cores on same device.
3. **All-gather phase (across devices)**: Statistics gathered across all devices via fabric.
4. **Post-phase**: Reduce gathered stats, compute 1/sqrt(Var + ε), apply normalization and gamma.

Parameters
----------

input_tensor : ttnn.Tensor
    Input activation tensor to be normalized.

    **Constraints**:
        - Must be rank 4 with shape ``(1, 1, 32, N)`` where N is divisible by tile width (32)
        - Layout must be ``TILE_LAYOUT``
        - Data type must be one of: ``FLOAT32``, ``BFLOAT16``, ``BFLOAT8_B``
        - Storage type must be ``DEVICE`` with allocated buffer
        - Memory layout must be width-sharded (``WIDTH_SHARDED``) or block-sharded
        - Height sharding (``HEIGHT_SHARDED``) is NOT supported
        - Shard orientation must be ``ROW_MAJOR``

program_config : LayerNormShardedMultiCoreProgramConfig
    Configuration for the sharded multi-core program execution.

    **Fields**:
        - ``compute_with_storage_grid_size``: CoreCoord specifying the compute grid dimensions
        - ``subblock_w``: Width of compute subblocks (must divide ``block_w``)
        - ``block_h``: Height of blocks (must be 1 for this operation)
        - ``block_w``: Width of blocks in tiles (must equal K / num_cores where K is the width in tiles)
        - ``inplace``: Whether to perform computation in-place

cluster_axis : int
    The axis of the mesh device cluster along which to perform the all-gather operation.
    Use 0 for column-wise gathering or 1 for row-wise gathering across devices.

mesh_device : MeshDevice
    The mesh device object representing the multi-device configuration.
    Must be a 2D mesh for the cluster_axis API.

global_semaphore : GlobalSemaphore
    Semaphore handle for synchronizing across devices during the all-gather phase.
    Create using ``ttnn.create_global_semaphore(mesh_device, core_grid, initial_value)``.

persistent_output_tensor : Optional[ttnn.Tensor], keyword-only
    Optional pre-allocated output tensor for persistent memory usage.
    Default: None (output tensor created internally).

num_links : Optional[int], keyword-only
    Number of fabric links to use for all-gather communication.
    Default: 1.

topology : ttnn.ccl.Topology, keyword-only
    The CCL topology for multi-device communication.
    Options: ``Topology.Linear`` or ``Topology.Ring``.
    Default: ``Topology.Linear``.

subdevice_id : Optional[SubDeviceId], keyword-only
    Sub-device identifier for sub-device execution.
    Default: None.

dtype : Optional[DataType], keyword-only
    Output data type. If None, uses the input tensor's data type.
    Default: None.

compute_kernel_config : Optional[DeviceComputeKernelConfig], keyword-only
    Configuration for compute kernels including math fidelity and precision settings.
    Default: None (uses HiFi4 math fidelity with FP32 dest accumulation enabled).

memory_config : Optional[MemoryConfig], keyword-only
    Output memory configuration.

    **Constraints**:
        - Must be sharded (not ``HEIGHT_SHARDED``)
        - Buffer type must match input tensor's buffer type
        - Memory layout must match input tensor's memory layout
        - Shard orientation must be ``ROW_MAJOR``

    Default: Uses input tensor's memory config.

residual_input_tensor : Optional[ttnn.Tensor], keyword-only
    Optional residual tensor to add to the input before normalization.

    **Constraints** (when provided):
        - Layout must be ``TILE_LAYOUT``
        - Shape must match input tensor shape exactly
        - Must be allocated on device
        - Must be on the same device as input tensor
        - Must be sharded if input is sharded
        - Must have same shard spec and memory config as input tensor
        - Height must be 32 (tile height)

    **Note**: This tensor is updated in-place with ``input + residual``.

    Default: None.

epsilon : float, keyword-only
    Small constant added to variance for numerical stability in the sqrt computation.
    Default: 1e-12.

weight : Optional[ttnn.Tensor], keyword-only
    Gamma weight tensor for scaling the normalized output.

    **Constraints**:
        - Required (not optional despite the type hint)
        - Layout must be ``ROW_MAJOR`` (preferred) or ``TILE_LAYOUT``
        - For ``ROW_MAJOR``: width must equal input tile width (32), physical_volume/width must equal input_width/tile_width
        - For ``TILE_LAYOUT``: last dimension must match input's last dimension, height must be 32
        - Data type must be ``FLOAT32`` or ``BFLOAT16`` (for row major)
        - Must be allocated on device on the same device as input tensor

stats : Optional[ttnn.Tensor], keyword-only
    Pre-allocated tensor to store intermediate statistics during the all-gather phase.
    This tensor is used for communication across devices.

    **Constraints**:
        - Required (not optional despite the type hint)
        - Should be allocated with appropriate shape for storing partial statistics
        - Typically shape ``(1, 1, 32, num_devices * num_stats_tiles)``
        - Must be sharded with ``TILE_LAYOUT``

use_noc1_only : bool, keyword-only
    Whether to use only NOC1 for data movement. When False, uses preferred NOC for
    DRAM read/write operations. Set to True for specific NOC routing requirements.
    Default: False.

Shapes
------

**Supported Input Shape**: ``(1, 1, 32, N)``
    - Batch dimensions must be 1
    - Height must be exactly 32 (tile height)
    - Width N must be divisible by 32 and by the number of cores × number of devices

**Output Shape**: Same as input shape ``(1, 1, 32, N)``
    - Output may be resharded to different core grid based on ``memory_config``

**Stats Tensor Shape**: ``(1, 1, 32, num_devices * stats_tiles_per_device)``
    - Used for storing and communicating partial statistics

**Weight Tensor Shape**:
    - Row major: ``(1, 1, N/32, 32)`` where N is the input width
    - Tile layout: ``(1, 1, 32, N)``

Example Usage
-------------

.. code-block:: python

    import torch
    import ttnn

    # Setup mesh device (e.g., 8x4 grid with 4 devices)
    num_devices = 4
    elements_per_batch = 8192

    # Calculate dimensions
    num_cores = 16  # 8x2 grid = 16 cores
    total_cores = num_cores * num_devices
    padded_dim_per_core = int(math.ceil(elements_per_batch / total_cores / 32) * 32)
    padded_dim = padded_dim_per_core * total_cores
    size_per_device = padded_dim // num_devices

    # Input shape: (1, 1, 32, padded_dim)
    input_shape = (1, 1, 32, padded_dim)

    # Create shard grids
    input_shard_grid = ttnn.CoreRangeSet({
        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))
    })

    # Create sharded memory config
    input_memory_config = ttnn.create_sharded_memory_config(
        shape=(32, padded_dim_per_core),
        core_grid=input_shard_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Create program config
    layer_norm_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(8, 2),
        subblock_w=1,
        block_h=1,
        block_w=(size_per_device // num_cores) // 32,
        inplace=False,
    )

    # Create global semaphore for synchronization
    ccl_semaphore = ttnn.create_global_semaphore(mesh_device, input_shard_grid, 0)

    # Create stats tensor for all-gather communication
    stats_memory_config = ttnn.create_sharded_memory_config(
        shape=(32, 128),
        core_grid=input_shard_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    stats_tensor = ttnn.from_torch(
        torch.zeros((1, 1, 32, 128), dtype=torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=stats_memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=(1, num_devices)),
    )

    # Prepare input tensors
    input_torch = torch.randn(input_shape)
    residual_torch = torch.randn(input_shape)
    gamma_torch = torch.randn((1, 1, 1, input_shape[3]))

    input_tensor = ttnn.as_tensor(
        input_torch,
        dtype=ttnn.bfloat8_b,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=(1, num_devices)),
        layout=ttnn.TILE_LAYOUT,
        memory_config=input_memory_config,
    )

    residual_tensor = ttnn.as_tensor(
        residual_torch,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=(1, num_devices)),
        layout=ttnn.TILE_LAYOUT,
        memory_config=input_memory_config,
    )

    gamma_tensor = ttnn.as_tensor(
        gamma_torch.reshape([1, 1, padded_dim // 32, 32]),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 2), mesh_shape=(1, num_devices)),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Execute fused RMS norm with all-gather
    output = ttnn.fused_rms_norm(
        input_tensor,
        layer_norm_config,
        cluster_axis=1,  # Gather along row axis
        mesh_device=mesh_device,
        global_semaphore=ccl_semaphore,
        topology=ttnn.Topology.Linear,
        memory_config=input_memory_config,
        epsilon=1e-5,
        weight=gamma_tensor,
        residual_input_tensor=residual_tensor,
        stats=stats_tensor,
    )

Notes
-----

- This operation is specifically optimized for LLM inference workloads with distributed normalization.
- The name ``fused_rms_norm`` refers to the original target shape but the operation supports
  various widths that meet the sharding constraints.
- Blackhole architecture with DRAM buffers is NOT supported.
- For best performance, ensure proper sub-device configuration when using with fabric.
        )doc",
        // Stats is internally computed
        ttnn::nanobind_arguments_t{
            // Used by all
            nb::arg("input_tensor"),
            nb::arg("program_config"),
            nb::arg("cluster_axis"),
            nb::arg("mesh_device"),
            nb::arg("global_semaphore"),  // TODO: Build this internally
            nb::kw_only(),
            // all gather
            nb::arg("persistent_output_tensor") = nb::none(),
            nb::arg("num_links") = nb::none(),
            nb::arg("topology") = ttnn::ccl::Topology::Linear,
            nb::arg("subdevice_id") = nb::none(),
            // common
            nb::arg("dtype") = nb::none(),  // Should default to BFLOAT 16 on pre, nullopt on post
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            // on pre only
            nb::arg("residual_input_tensor") = nb::none(),
            // on post only
            nb::arg("epsilon") = 1e-12,  // constant 1e-12 on pre, value only affects post
            nb::arg("weight") = nb::none(),
            nb::arg("stats") = nb::none(),
            nb::arg("use_noc1_only") = false});
}
}  // namespace ttnn::operations::experimental::ccl
