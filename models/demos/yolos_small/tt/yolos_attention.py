# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
YOLOS-small Attention Module

Implements multi-head self-attention for the ViT encoder.
"""

import ttnn
import torch
import math
from models.common.lightweightmodule import LightweightModule


class TtYolosAttention(LightweightModule):
    """
    Multi-head self-attention for YOLOS.
    """
    
    def __init__(
        self,
        args,
        mesh_device,
        dtype,
        state_dict,
        layer_num: int,
        weight_cache_path=None,
    ):
        super().__init__()
        
        self.args = args
        self.mesh_device = mesh_device
        self.layer_num = layer_num
        
        self.hidden_size = args.hidden_size
        self.num_attention_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Load weights
        self._load_weights(state_dict, layer_num, dtype)
    
    def _load_weights(self, state_dict, layer_num, dtype):
        """Load attention weights"""
        prefix = f"vit.encoder.layer.{layer_num}.attention.attention"
        
        if state_dict is not None and not self.args.dummy_weights:
            self.wq = self._create_weight(state_dict.get(f"{prefix}.query.weight"), dtype)
            self.bq = self._create_weight(state_dict.get(f"{prefix}.query.bias"), dtype)
            self.wk = self._create_weight(state_dict.get(f"{prefix}.key.weight"), dtype)
            self.bk = self._create_weight(state_dict.get(f"{prefix}.key.bias"), dtype)
            self.wv = self._create_weight(state_dict.get(f"{prefix}.value.weight"), dtype)
            self.bv = self._create_weight(state_dict.get(f"{prefix}.value.bias"), dtype)
            
            out_prefix = f"vit.encoder.layer.{layer_num}.attention.output"
            self.wo = self._create_weight(state_dict.get(f"{out_prefix}.dense.weight"), dtype)
            self.bo = self._create_weight(state_dict.get(f"{out_prefix}.dense.bias"), dtype)
        else:
            self.wq = self.bq = self.wk = self.bk = self.wv = self.bv = None
            self.wo = self.bo = None
    
    def _create_weight(self, weight, dtype):
        """Convert weight to ttnn tensor"""
        if weight is None:
            return None
        
        if len(weight.shape) == 2:
            weight = weight.T.contiguous()
        
        return ttnn.from_torch(
            weight,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    
    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [batch, seq_len, hidden_size]
            
        Returns:
            Output [batch, seq_len, hidden_size]
        """
        if self.wq is None:
            return x
        
        batch_size, seq_len, _ = x.shape
        
        # QKV projections
        q = ttnn.linear(x, self.wq, bias=self.bq)
        k = ttnn.linear(x, self.wk, bias=self.bk)
        v = ttnn.linear(x, self.wv, bias=self.bv)
        
        # Reshape to multi-head
        q = ttnn.reshape(q, [batch_size, seq_len, self.num_attention_heads, self.head_dim])
        k = ttnn.reshape(k, [batch_size, seq_len, self.num_attention_heads, self.head_dim])
        v = ttnn.reshape(v, [batch_size, seq_len, self.num_attention_heads, self.head_dim])
        
        # Transpose for attention: [B, heads, seq, head_dim]
        q = ttnn.permute(q, [0, 2, 1, 3])
        k = ttnn.permute(k, [0, 2, 1, 3])
        v = ttnn.permute(v, [0, 2, 1, 3])
        
        # Attention
        k_t = ttnn.permute(k, [0, 1, 3, 2])
        scores = ttnn.matmul(q, k_t)
        scores = ttnn.mul(scores, self.scale)
        attn_weights = ttnn.softmax(scores, dim=-1)
        attn_out = ttnn.matmul(attn_weights, v)
        
        # Transpose back and reshape
        attn_out = ttnn.permute(attn_out, [0, 2, 1, 3])
        attn_out = ttnn.reshape(attn_out, [batch_size, seq_len, self.hidden_size])
        
        # Output projection
        output = ttnn.linear(attn_out, self.wo, bias=self.bo)
        
        return output
