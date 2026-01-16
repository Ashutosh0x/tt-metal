# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.phi1_5.tt.phi_attention import TtPhiAttention
from models.demos.phi1_5.tt.phi_mlp import TtPhiMLP


class TtPhiDecoderLayer(LightweightModule):
    def __init__(self, device, args, state_dict, layer_num, dtype):
        super().__init__()
        self.device = device
        self.args = args
        
        prefix = f"model.layers.{layer_num}"
        
        # LayerNorm
        ln_weight = state_dict[f"{prefix}.input_layernorm.weight"]
        ln_bias = state_dict[f"{prefix}.input_layernorm.bias"]
        
        self.input_layernorm_weight = ttnn.from_torch(
            ln_weight.reshape(1, 1, 1, -1),
            device=self.device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.input_layernorm_bias = ttnn.from_torch(
            ln_bias.reshape(1, 1, 1, -1),
            device=self.device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        
        self.self_attn = TtPhiAttention(device, args, state_dict, layer_num, dtype)
        self.mlp = TtPhiMLP(device, args, state_dict, layer_num, dtype)

    def forward(self, x, cos, sin, mask=None, layer_past=None, layer_past_len=0):
        residual = x
        
        # Norm
        norm_x = ttnn.layer_norm(
            x, 
            weight=self.input_layernorm_weight, 
            bias=self.input_layernorm_bias,
            memory_config=self.args.model_config["L1_MEMCFG"]
        )
        
        # Parallel Attention and MLP
        attn_out = self.self_attn(norm_x, cos, sin, mask, layer_past, layer_past_len)
        mlp_out = self.mlp(norm_x)
        
        # x = residual + attn_out + mlp_out
        x = ttnn.add(residual, attn_out, memory_config=self.args.model_config["L1_MEMCFG"])
        x = ttnn.add(x, mlp_out, memory_config=self.args.model_config["L1_MEMCFG"])
        
        return x
