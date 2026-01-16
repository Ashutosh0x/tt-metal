# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from dataclasses import dataclass
from typing import Optional


@dataclass
class PhiConfig:
    num_hidden_layers: int = 24
    hidden_size: int = 2048
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    intermediate_size: int = 8192
    hidden_act: str = "gelu_new"
    layer_norm_eps: float = 1e-05
    vocab_size: int = 51200
    max_position_embeddings: int = 2048
    partial_rotary_factor: float = 0.5
    rope_theta: float = 10000.0
    
    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def rotary_dim(self):
        return int(self.head_dim * self.partial_rotary_factor)


class TtPhiArgs:
    def __init__(self, device, dummy_weights=False):
        self.device = device
        self.dummy_weights = dummy_weights
        
        # Model config
        self.config = PhiConfig()
        
        # Memory config
        self.model_config = {
            "DEFAULT_DTYPE": ttnn.bfloat16,
            "WEIGHTS_DTYPE": ttnn.bfloat8_b,
            "ACTIVATIONS_DTYPE": ttnn.bfloat16,
            "KV_CACHE_DTYPE": ttnn.bfloat8_b,
            "DEFAULT_MEMCFG": ttnn.DRAM_MEMORY_CONFIG,
            "L1_MEMCFG": ttnn.L1_MEMORY_CONFIG,
        }

        # Sharding Configs (Assuming Wormhole 8x8 grid)
        self.compute_grid_size = device.compute_with_storage_grid_size()
        self.num_cores = self.compute_grid_size.x * self.compute_grid_size.y
        
        # Height sharding for hidden_size=2048
        # 2048 / 32 (TILE) = 64 tiles. We can shard across 64 cores or 32 cores.
        self.sharded_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}), # 32 cores example
                [32, 2048], # Shard shape: [seq_per_core, hidden_size]
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
