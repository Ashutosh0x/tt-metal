# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Ministral-8B MLP (SwiGLU)

Implements the feed-forward network with:
- Gate projection (up_proj)
- Up projection (gate_proj)
- Down projection
- SiLU activation with gating
"""

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtMinistralMLP(LightweightModule):
    """
    SwiGLU MLP for Ministral-8B.
    
    Architecture:
    x -> gate_proj -> SiLU
                        * -> down_proj -> output
    x -> up_proj   ->
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
        self.hidden_dim = args.hidden_dim
        
        # Load MLP weights
        self._load_weights(state_dict, layer_num, weight_cache_path, dtype)
    
    def _load_weights(self, state_dict, layer_num, weight_cache_path, dtype):
        """Load and convert MLP weights to ttnn format"""
        
        prefix = f"model.layers.{layer_num}.mlp"
        
        if state_dict is not None and not self.args.dummy_weights:
            # Gate projection (for SiLU gating)
            self.w_gate = self._create_weight_tensor(
                state_dict.get(f"{prefix}.gate_proj.weight"),
                dtype,
            )
            # Up projection
            self.w_up = self._create_weight_tensor(
                state_dict.get(f"{prefix}.up_proj.weight"),
                dtype,
            )
            # Down projection
            self.w_down = self._create_weight_tensor(
                state_dict.get(f"{prefix}.down_proj.weight"),
                dtype,
            )
        else:
            # Dummy weights for testing
            self.w_gate = None
            self.w_up = None
            self.w_down = None
    
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
        mode: str = "prefill",
    ) -> ttnn.Tensor:
        """
        Forward pass through MLP.
        
        SwiGLU: output = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            mode: "prefill" or "decode"
            
        Returns:
            Output tensor [batch, seq_len, dim]
        """
        if self.w_gate is None:
            # Passthrough for testing without weights
            return x
        
        # Gate projection with SiLU activation
        gate = ttnn.linear(x, self.w_gate)  # [B, S, hidden_dim]
        gate = ttnn.silu(gate)
        
        # Up projection
        up = ttnn.linear(x, self.w_up)  # [B, S, hidden_dim]
        
        # Element-wise multiplication (gating)
        hidden = ttnn.mul(gate, up)
        
        # Down projection
        output = ttnn.linear(hidden, self.w_down)  # [B, S, dim]
        
        # Deallocate intermediate tensors
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        ttnn.deallocate(hidden)
        
        return output
