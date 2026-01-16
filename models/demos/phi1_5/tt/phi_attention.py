# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import torch
import ttnn
from models.common.lightweightmodule import LightweightModule


class TtPhiAttention(LightweightModule):
    def __init__(self, device, args, state_dict, layer_num, dtype):
        super().__init__()
        self.device = device
        self.args = args
        self.config = args.config
        
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.config.head_dim
        self.rotary_dim = self.config.rotary_dim
        
        prefix = f"model.layers.{layer_num}.self_attn"
        
        self.w_qkv = self._load_weight(state_dict, f"{prefix}.q_proj.weight", f"{prefix}.k_proj.weight", f"{prefix}.v_proj.weight", dtype)
        self.b_qkv = self._load_bias(state_dict, f"{prefix}.q_proj.bias", f"{prefix}.k_proj.bias", f"{prefix}.v_proj.bias", dtype)
        
        self.w_out = self._load_weight_single(state_dict, f"{prefix}.dense.weight", dtype)
        self.b_out = self._load_bias_single(state_dict, f"{prefix}.dense.bias", dtype)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _load_weight_single(self, state_dict, name, dtype):
        if state_dict is None or name not in state_dict:
            return None
        weight = state_dict[name]
        if len(weight.shape) == 2:
            weight = weight.T.contiguous()
        return ttnn.from_torch(
            weight,
            device=self.device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _load_bias_single(self, state_dict, name, dtype):
        if state_dict is None or name not in state_dict:
            return None
        bias = state_dict[name].reshape(1, 1, 1, -1)
        return ttnn.from_torch(
            bias,
            device=self.device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _load_weight(self, state_dict, q_name, k_name, v_name, dtype):
        if state_dict is None:
            return None
        q = state_dict[q_name]
        k = state_dict[k_name]
        v = state_dict[v_name]
        qkv = torch.cat([q, k, v], dim=0).T.contiguous()
        return ttnn.from_torch(
            qkv,
            device=self.device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _load_bias(self, state_dict, q_name, k_name, v_name, dtype):
        if state_dict is None:
            return None
        q = state_dict[q_name]
        k = state_dict[k_name]
        v = state_dict[v_name]
        qkv_bias = torch.cat([q, k, v], dim=0).reshape(1, 1, 1, -1)
        return ttnn.from_torch(
            qkv_bias,
            device=self.device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, x, cos, sin, mask=None, layer_past=None, layer_past_len=0):
        bsz, seq_len, _ = x.shape
        
        # Projected QKV
        qkv = ttnn.linear(x, self.w_qkv, bias=self.b_qkv)
        
        # Split Q, K, V
        q, k, v = ttnn.split(qkv, self.hidden_size, dim=-1)
        
        # Reshape for heads
        q = ttnn.reshape(q, (bsz, seq_len, self.num_heads, self.head_dim))
        k = ttnn.reshape(k, (bsz, seq_len, self.num_heads, self.head_dim))
        v = ttnn.reshape(v, (bsz, seq_len, self.num_heads, self.head_dim))
        
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.permute(v, (0, 2, 1, 3))

        # Partial RoPE (applied to Q and K)
        # Note: cos/sin must match the sequence length of q/k
        if self.rotary_dim < self.head_dim:
            q_rot = ttnn.slice(q, [0, 0, 0, 0], [bsz, self.num_heads, seq_len, self.rotary_dim])
            q_pass = ttnn.slice(q, [0, 0, 0, self.rotary_dim], [bsz, self.num_heads, seq_len, self.head_dim])
            
            k_rot = ttnn.slice(k, [0, 0, 0, 0], [bsz, self.num_heads, seq_len, self.rotary_dim])
            k_pass = ttnn.slice(k, [0, 0, 0, self.rotary_dim], [bsz, self.num_heads, seq_len, self.head_dim])
            
            q_rot = ttnn.experimental.rotary_embedding(q_rot, cos, sin)
            k_rot = ttnn.experimental.rotary_embedding(k_rot, cos, sin)
            
            q = ttnn.concat([q_rot, q_pass], dim=-1)
            k = ttnn.concat([k_rot, k_pass], dim=-1)
        else:
            q = ttnn.experimental.rotary_embedding(q, cos, sin)
            k = ttnn.experimental.rotary_embedding(k, cos, sin)

        # KV Cache Update
        if layer_past is not None:
            k, v = layer_past.update(k, v, layer_past_len)
            # After update, k and v contain full history up to layer_past_len + seq_len
            # For SDPA, we use the cached versions

        # Scale dot product attention
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=(layer_past_len == 0), scale=self.scale
        )
        
        # Reshape and project out
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, (bsz, seq_len, self.hidden_size))
        
        output = ttnn.linear(attn_output, self.w_out, bias=self.b_out)
        
        return output
