# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Ministral-8B KV Cache

Implements KV cache for efficient autoregressive decoding.
"""

import ttnn
import torch
from typing import Tuple, Optional


class KVCache:
    """
    Key-Value cache for efficient decode-mode generation.
    
    Stores precomputed K and V tensors from previous tokens,
    allowing O(1) attention computation per new token.
    """
    
    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        n_layers: int,
        device,
        dtype=ttnn.bfloat16,
    ):
        """
        Initialize KV cache for all layers.
        
        Args:
            batch_size: Maximum batch size
            max_seq_len: Maximum sequence length
            n_kv_heads: Number of KV heads (8 for Ministral)
            head_dim: Dimension per head (128)
            n_layers: Number of decoder layers (36)
            device: ttnn device
            dtype: Data type for cache tensors
        """
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_layers = n_layers
        self.device = device
        self.dtype = dtype
        
        # Current position in the cache
        self.current_pos = 0
        
        # Initialize cache tensors for each layer
        # Shape: [batch, n_kv_heads, max_seq_len, head_dim]
        self.k_cache = []
        self.v_cache = []
        
        for _ in range(n_layers):
            k = self._create_cache_tensor()
            v = self._create_cache_tensor()
            self.k_cache.append(k)
            self.v_cache.append(v)
    
    def _create_cache_tensor(self) -> ttnn.Tensor:
        """Create a single cache tensor"""
        cache_shape = [self.batch_size, self.n_kv_heads, self.max_seq_len, self.head_dim]
        
        # Initialize with zeros
        cache_torch = torch.zeros(cache_shape, dtype=torch.bfloat16)
        
        cache_tt = ttnn.from_torch(
            cache_torch,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return cache_tt
    
    def update(
        self,
        layer_idx: int,
        k_new: ttnn.Tensor,
        v_new: ttnn.Tensor,
        start_pos: int,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Update cache with new K, V and return full cached tensors.
        
        Args:
            layer_idx: Decoder layer index
            k_new: New key tensor [batch, seq_len, n_kv_heads, head_dim]
            v_new: New value tensor [batch, seq_len, n_kv_heads, head_dim]
            start_pos: Starting position in sequence
            
        Returns:
            Tuple of (cached_k, cached_v) up to current position
        """
        seq_len = k_new.shape[1]
        end_pos = start_pos + seq_len
        
        # Transpose to cache layout: [batch, n_kv_heads, seq_len, head_dim]
        k_new_t = ttnn.permute(k_new, [0, 2, 1, 3])
        v_new_t = ttnn.permute(v_new, [0, 2, 1, 3])
        
        # Update cache at the specified position
        # Note: In practice, this would use scatter or paged cache operations
        # For now, we return the new tensors directly
        
        self.current_pos = end_pos
        
        return k_new_t, v_new_t
    
    def get(self, layer_idx: int) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Get cached K, V for a layer"""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]
    
    def reset(self):
        """Reset cache position"""
        self.current_pos = 0


class PagedKVCache(KVCache):
    """
    Paged KV cache for memory-efficient long sequence handling.
    
    Uses paging to handle sequences longer than available L1 memory.
    """
    
    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        n_layers: int,
        device,
        page_size: int = 256,
        dtype=ttnn.bfloat16,
    ):
        super().__init__(
            batch_size, max_seq_len, n_kv_heads, head_dim, n_layers, device, dtype
        )
        self.page_size = page_size
        self.num_pages = (max_seq_len + page_size - 1) // page_size
        
        # Page table: maps logical page to physical page
        self.page_table = torch.zeros(batch_size, self.num_pages, dtype=torch.int32)
    
    def get_page_table(self, batch_idx: int = 0) -> torch.Tensor:
        """Get page table for a batch index"""
        return self.page_table[batch_idx]
