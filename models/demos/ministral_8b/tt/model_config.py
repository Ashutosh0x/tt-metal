# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Ministral-8B-Instruct-2410 Model Configuration

Model Architecture:
- 36 layers
- 4096 hidden dim
- 32 query heads, 8 KV heads (GQA 4:1)
- 128 head dim
- 128k context with sliding-window attention
"""

import ttnn
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class MinistralConfig:
    """Configuration for Ministral-8B-Instruct-2410"""
    
    # Model architecture
    dim: int = 4096
    n_layers: int = 36
    n_heads: int = 32  # Query heads
    n_kv_heads: int = 8  # KV heads (GQA)
    vocab_size: int = 32768
    hidden_dim: int = 14336  # FFN intermediate dim (calculated: 4096 * 3.5)
    
    # Attention config
    head_dim: int = 128  # dim // n_heads
    max_seq_len: int = 128 * 1024  # 128k context
    sliding_window: int = 4096  # Sliding window attention size
    
    # RoPE config
    rope_theta: float = 10000.0
    
    # Precision
    norm_eps: float = 1e-5
    
    # Runtime config
    max_batch_size: int = 1
    
    def __post_init__(self):
        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"


class TtMinistralArgs:
    """
    Tenstorrent-specific model arguments for Ministral-8B.
    Provides memory configs, program configs, and weight loading utilities.
    """
    
    def __init__(
        self,
        mesh_device,
        max_batch_size: int = 1,
        max_seq_len: int = 2048,
        dummy_weights: bool = False,
    ):
        self.mesh_device = mesh_device
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.dummy_weights = dummy_weights
        
        # Model architecture params
        self.config = MinistralConfig(max_batch_size=max_batch_size)
        self.dim = self.config.dim
        self.n_layers = self.config.n_layers
        self.n_heads = self.config.n_heads
        self.n_kv_heads = self.config.n_kv_heads
        self.head_dim = self.config.head_dim
        self.hidden_dim = self.config.hidden_dim
        self.vocab_size = self.config.vocab_size
        self.norm_eps = self.config.norm_eps
        self.rope_theta = self.config.rope_theta
        self.sliding_window = self.config.sliding_window
        
        # Derived params
        self.kv_heads_per_group = self.n_heads // self.n_kv_heads
        
        # Memory configurations
        self._setup_memory_configs()
    
    def _setup_memory_configs(self):
        """Setup ttnn memory configurations for optimal performance"""
        
        # Default DRAM config for weights
        self.weight_mem_config = ttnn.DRAM_MEMORY_CONFIG
        
        # L1 sharded config for activations
        self.l1_mem_config = ttnn.L1_MEMORY_CONFIG
        
        # Interleaved config for intermediate tensors
        self.interleaved_mem_config = ttnn.DRAM_MEMORY_CONFIG
    
    def get_model_config(self):
        """Return model configuration dictionary"""
        return {
            "dim": self.dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "hidden_dim": self.hidden_dim,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "max_batch_size": self.max_batch_size,
            "sliding_window": self.sliding_window,
        }
    
    def get_state_dict_prefix(self, base: str, layer_num: Optional[int] = None) -> str:
        """Get the state dict prefix for a given layer"""
        if layer_num is not None:
            return f"model.layers.{layer_num}.{base}"
        return f"model.{base}"
