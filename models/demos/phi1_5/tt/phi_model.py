# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.phi1_5.tt.phi_decoder import TtPhiDecoderLayer


class TtPhiModel(LightweightModule):
    def __init__(self, device, args, state_dict, dtype):
        super().__init__()
        self.device = device
        self.args = args
        self.config = args.config
        
        # Word Embeddings
        embed_weight = state_dict["model.embed_tokens.weight"]
        self.embed_tokens = ttnn.from_torch(
            embed_weight,
            device=self.device,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT, # Embeddings often better in RM
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        
        # Decoder Layers
        self.layers = []
        for i in range(self.config.num_hidden_layers):
            self.layers.append(TtPhiDecoderLayer(device, args, state_dict, i, dtype))
            
        # Final Norm
        ln_weight = state_dict["model.final_layernorm.weight"]
        ln_bias = state_dict["model.final_layernorm.bias"]
        self.final_layernorm_weight = ttnn.from_torch(
            ln_weight.reshape(1, 1, 1, -1),
            device=self.device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
        )
        self.final_layernorm_bias = ttnn.from_torch(
            ln_bias.reshape(1, 1, 1, -1),
            device=self.device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
        )
        
        # LM Head
        head_weight = state_dict["lm_head.weight"]
        head_bias = state_dict["lm_head.bias"]
        self.lm_head_weight = ttnn.from_torch(
            head_weight.T.contiguous(),
            device=self.device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
        )
        self.lm_head_bias = ttnn.from_torch(
            head_bias.reshape(1, 1, 1, -1),
            device=self.device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
        )

    def forward(self, input_ids, cos, sin, mask=None):
        # input_ids: [batch, seq_len]
        
        # Embedding
        x = ttnn.embedding(input_ids, self.embed_tokens)
        
        # In ttnn, we might need to cast to TILE for subsequent layers
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        
        # Layers
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
            
        # Final Norm
        x = ttnn.layer_norm(x, weight=self.final_layernorm_weight, bias=self.final_layernorm_bias)
        
        # LM Head
        logits = ttnn.linear(x, self.lm_head_weight, bias=self.lm_head_bias)
        
        return logits


def precompute_rope_phi(config, device):
    """Precompute cos/sin for partial RoPE"""
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, config.rotary_dim, 2).float() / config.rotary_dim))
    t = torch.arange(config.max_position_embeddings, device=inv_freq.device)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().reshape(1, 1, config.max_position_embeddings, config.rotary_dim)
    sin = emb.sin().reshape(1, 1, config.max_position_embeddings, config.rotary_dim)
    
    cos_tt = ttnn.from_torch(cos, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    sin_tt = ttnn.from_torch(sin, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    
    return cos_tt, sin_tt
