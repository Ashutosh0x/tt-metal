# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


class TtPhiKVCache:
    def __init__(self, device, args, bsz, max_seq_len, dtype):
        self.device = device
        self.args = args
        self.config = args.config
        
        self.max_seq_len = max_seq_len
        self.bsz = bsz
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.config.head_dim
        
        cache_shape = (bsz, self.num_heads, max_seq_len, self.head_dim)
        
        self.k_cache = ttnn.zeros(
            cache_shape,
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.v_cache = ttnn.zeros(
            cache_shape,
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def update(self, k, v, layer_past_len):
        """
        k, v: [bsz, num_heads, new_seq_len, head_dim]
        layer_past_len: int
        """
        # In ttnn, we can use slice_set for cache update
        # However, for simplicity in Stage 2, we can implement it as a full update
        # For a hard bounty, we would use paged_attention or efficient slice_set
        
        # Simplified update (copying into the cache at the correct offset)
        new_seq_len = k.shape[2]
        
        self.k_cache = ttnn.slice_set(self.k_cache, k, [0, 0, layer_past_len, 0])
        self.v_cache = ttnn.slice_set(self.v_cache, v, [0, 0, layer_past_len, 0])
        
        return self.k_cache, self.v_cache

    def get_past_kv(self, current_len):
        # Return valid parts of the cache
        k = ttnn.slice(self.k_cache, [0, 0, 0, 0], [self.bsz, self.num_heads, current_len, self.head_dim])
        v = ttnn.slice(self.v_cache, [0, 0, 0, 0], [self.bsz, self.num_heads, current_len, self.head_dim])
        return k, v
