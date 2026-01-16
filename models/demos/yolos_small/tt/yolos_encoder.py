# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
YOLOS-small Encoder Block

ViT encoder block with LayerNorm, Attention, and MLP.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.demos.yolos_small.tt.yolos_attention import TtYolosAttention
from models.demos.yolos_small.tt.yolos_mlp import TtYolosMLP


class TtYolosEncoderBlock(LightweightModule):
    """
    Single ViT encoder block for YOLOS.
    
    Structure: LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual
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
        self.layer_num = layer_num
        
        # LayerNorm before attention and MLP
        prefix = f"vit.encoder.layer.{layer_num}"
        
        if state_dict is not None and not args.dummy_weights:
            ln1_weight = state_dict.get(f"{prefix}.layernorm_before.weight")
            ln1_bias = state_dict.get(f"{prefix}.layernorm_before.bias")
            ln2_weight = state_dict.get(f"{prefix}.layernorm_after.weight")
            ln2_bias = state_dict.get(f"{prefix}.layernorm_after.bias")
        else:
            ln1_weight = ln1_bias = ln2_weight = ln2_bias = None
        
        # LayerNorm 1 (before attention)
        self.ln1 = self._create_layernorm(ln1_weight, ln1_bias, mesh_device, dtype)
        
        # Attention
        self.attention = TtYolosAttention(
            args, mesh_device, dtype, state_dict, layer_num, weight_cache_path
        )
        
        # LayerNorm 2 (before MLP)
        self.ln2 = self._create_layernorm(ln2_weight, ln2_bias, mesh_device, dtype)
        
        # MLP
        self.mlp = TtYolosMLP(
            args, mesh_device, dtype, state_dict, layer_num, weight_cache_path
        )
    
    def _create_layernorm(self, weight, bias, device, dtype):
        """Create LayerNorm parameters"""
        if weight is None:
            return None, None
        
        weight_tt = ttnn.from_torch(
            weight,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        bias_tt = ttnn.from_torch(
            bias,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ) if bias is not None else None
        
        return weight_tt, bias_tt
    
    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [batch, seq_len, hidden_size]
            
        Returns:
            Output [batch, seq_len, hidden_size]
        """
        # LayerNorm + Attention + Residual
        if self.ln1[0] is not None:
            normed = ttnn.layer_norm(x, weight=self.ln1[0], bias=self.ln1[1])
        else:
            normed = x
        attn_out = self.attention.forward(normed)
        x = ttnn.add(x, attn_out)
        
        # LayerNorm + MLP + Residual
        if self.ln2[0] is not None:
            normed = ttnn.layer_norm(x, weight=self.ln2[0], bias=self.ln2[1])
        else:
            normed = x
        mlp_out = self.mlp.forward(normed)
        x = ttnn.add(x, mlp_out)
        
        return x
