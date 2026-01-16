# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
YOLOS-small MLP Module

Implements the feed-forward MLP in each ViT encoder block.
"""

import ttnn
import torch
from models.common.lightweightmodule import LightweightModule


class TtYolosMLP(LightweightModule):
    """
    Feed-forward MLP for YOLOS encoder blocks.
    Uses GELU activation.
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
        self.intermediate_size = args.intermediate_size
        
        # Load weights
        self._load_weights(state_dict, layer_num, dtype)
    
    def _load_weights(self, state_dict, layer_num, dtype):
        """Load MLP weights"""
        prefix = f"vit.encoder.layer.{layer_num}.intermediate"
        out_prefix = f"vit.encoder.layer.{layer_num}.output"
        
        if state_dict is not None and not self.args.dummy_weights:
            self.w_fc1 = self._create_weight(state_dict.get(f"{prefix}.dense.weight"), dtype)
            self.b_fc1 = self._create_weight(state_dict.get(f"{prefix}.dense.bias"), dtype)
            self.w_fc2 = self._create_weight(state_dict.get(f"{out_prefix}.dense.weight"), dtype)
            self.b_fc2 = self._create_weight(state_dict.get(f"{out_prefix}.dense.bias"), dtype)
        else:
            self.w_fc1 = self.b_fc1 = self.w_fc2 = self.b_fc2 = None
    
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
        if self.w_fc1 is None:
            return x
        
        # FC1 + GELU
        hidden = ttnn.linear(x, self.w_fc1, bias=self.b_fc1)
        hidden = ttnn.gelu(hidden)
        
        # FC2
        output = ttnn.linear(hidden, self.w_fc2, bias=self.b_fc2)
        
        return output
