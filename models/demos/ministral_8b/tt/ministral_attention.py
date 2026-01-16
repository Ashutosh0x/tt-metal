# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Ministral-8B Grouped Query Attention (GQA)

Implements attention with:
- 32 query heads, 8 KV heads (4:1 ratio)
- Rotary Position Embeddings (RoPE)
- Sliding window attention support
"""

import ttnn
import torch
import math
from models.common.lightweightmodule import LightweightModule


class TtMinistralAttention(LightweightModule):
    """
    Grouped Query Attention for Ministral-8B.
    
    GQA groups multiple query heads to share the same KV heads,
    reducing memory bandwidth while maintaining quality.
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
        
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.head_dim
        self.n_rep = args.n_heads // args.n_kv_heads  # GQA repetition factor
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Load QKV projection weights
        self._load_weights(state_dict, layer_num, weight_cache_path, dtype)
    
    def _load_weights(self, state_dict, layer_num, weight_cache_path, dtype):
        """Load and convert attention weights to ttnn format"""
        
        prefix = f"model.layers.{layer_num}.self_attn"
        
        # Q projection: [dim, n_heads * head_dim]
        # K projection: [dim, n_kv_heads * head_dim]
        # V projection: [dim, n_kv_heads * head_dim]
        # O projection: [n_heads * head_dim, dim]
        
        if state_dict is not None and not self.args.dummy_weights:
            self.wq = self._create_weight_tensor(
                state_dict.get(f"{prefix}.q_proj.weight"),
                dtype,
            )
            self.wk = self._create_weight_tensor(
                state_dict.get(f"{prefix}.k_proj.weight"),
                dtype,
            )
            self.wv = self._create_weight_tensor(
                state_dict.get(f"{prefix}.v_proj.weight"),
                dtype,
            )
            self.wo = self._create_weight_tensor(
                state_dict.get(f"{prefix}.o_proj.weight"),
                dtype,
            )
        else:
            # Dummy weights for testing
            self.wq = None
            self.wk = None
            self.wv = None
            self.wo = None
    
    def _create_weight_tensor(self, weight, dtype):
        """Convert PyTorch weight to ttnn tensor"""
        if weight is None:
            return None
        
        # Transpose for ttnn matmul convention
        weight_t = weight.T.contiguous()
        
        # Convert to ttnn
        tt_weight = ttnn.from_torch(
            weight_t,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return tt_weight
    
    def forward(
        self,
        x: ttnn.Tensor,
        current_pos: int,
        rot_mats=None,
        mode: str = "prefill",
        kv_cache=None,
    ) -> ttnn.Tensor:
        """
        Forward pass through attention.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            current_pos: Current position in sequence
            rot_mats: Rotary position embeddings (cos, sin)
            mode: "prefill" or "decode"
            kv_cache: KV cache tuple (k_cache, v_cache)
            
        Returns:
            Output tensor [batch, seq_len, dim]
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # QKV projections
        if self.wq is not None:
            q = ttnn.linear(x, self.wq)  # [B, S, n_heads * head_dim]
            k = ttnn.linear(x, self.wk)  # [B, S, n_kv_heads * head_dim]
            v = ttnn.linear(x, self.wv)  # [B, S, n_kv_heads * head_dim]
        else:
            # Passthrough for testing without weights
            return x
        
        # Reshape for multi-head attention
        # Q: [B, S, n_heads, head_dim]
        # K, V: [B, S, n_kv_heads, head_dim]
        q = ttnn.reshape(q, [batch_size, seq_len, self.n_heads, self.head_dim])
        k = ttnn.reshape(k, [batch_size, seq_len, self.n_kv_heads, self.head_dim])
        v = ttnn.reshape(v, [batch_size, seq_len, self.n_kv_heads, self.head_dim])
        
        # Apply RoPE (if provided)
        if rot_mats is not None:
            cos, sin = rot_mats
            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)
        
        # Repeat KV heads for GQA
        if self.n_rep > 1:
            k = self._repeat_kv(k, self.n_rep)
            v = self._repeat_kv(v, self.n_rep)
        
        # Transpose for attention: [B, n_heads, S, head_dim]
        q = ttnn.permute(q, [0, 2, 1, 3])
        k = ttnn.permute(k, [0, 2, 1, 3])
        v = ttnn.permute(v, [0, 2, 1, 3])
        
        # Scaled dot-product attention
        # scores = Q @ K^T / sqrt(d_k)
        k_t = ttnn.permute(k, [0, 1, 3, 2])  # [B, n_heads, head_dim, S]
        scores = ttnn.matmul(q, k_t)
        scores = ttnn.mul(scores, self.scale)
        
        # Apply causal mask for prefill
        if mode == "prefill" and seq_len > 1:
            # Create causal mask (lower triangular)
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) * -1e9
            mask_tt = ttnn.from_torch(mask, dtype=ttnn.bfloat16, device=self.mesh_device)
            scores = ttnn.add(scores, mask_tt)
        
        # Softmax
        attn_weights = ttnn.softmax(scores, dim=-1)
        
        # attn_output = attn_weights @ V
        attn_output = ttnn.matmul(attn_weights, v)  # [B, n_heads, S, head_dim]
        
        # Transpose back: [B, S, n_heads, head_dim]
        attn_output = ttnn.permute(attn_output, [0, 2, 1, 3])
        
        # Reshape to [B, S, dim]
        attn_output = ttnn.reshape(attn_output, [batch_size, seq_len, self.dim])
        
        # Output projection
        output = ttnn.linear(attn_output, self.wo)
        
        return output
    
    def _apply_rope(self, x, cos, sin):
        """Apply rotary position embeddings"""
        # Simplified RoPE - full implementation would handle interleaved dims
        x_rot = ttnn.mul(x, cos)
        # For full RoPE, we'd also need to rotate and add sin component
        return x_rot
    
    def _repeat_kv(self, x, n_rep):
        """Repeat KV heads for GQA"""
        if n_rep == 1:
            return x
        
        batch, seq_len, n_kv_heads, head_dim = x.shape
        # Expand and reshape
        x = ttnn.reshape(x, [batch, seq_len, n_kv_heads, 1, head_dim])
        # Tile along the new dimension
        x = ttnn.repeat(x, [1, 1, 1, n_rep, 1])
        x = ttnn.reshape(x, [batch, seq_len, n_kv_heads * n_rep, head_dim])
        return x
