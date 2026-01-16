# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Ministral-8B Decoder Block

Implements a single transformer decoder layer with:
- RMSNorm pre-normalization
- Grouped Query Attention (GQA)
- SwiGLU MLP
- Residual connections
"""

import ttnn
from models.common.rmsnorm import RMSNorm
from models.common.lightweightmodule import LightweightModule


class TtMinistralDecoderBlock(LightweightModule):
    """
    Single transformer decoder block for Ministral-8B.
    
    Architecture:
    x -> RMSNorm -> Attention -> + -> RMSNorm -> MLP -> + -> output
         |__________________________|    |______________|
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
        
        # Attention norm
        self.attention_norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            state_dict=state_dict,
            state_dict_prefix=args.get_state_dict_prefix("input_layernorm", layer_num),
            weight_cache_path=weight_cache_path if not args.dummy_weights else None,
            weight_dtype=ttnn.bfloat16,
            weight_key="weight",
        )
        
        # FFN norm
        self.ffn_norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            state_dict=state_dict,
            state_dict_prefix=args.get_state_dict_prefix("post_attention_layernorm", layer_num),
            weight_cache_path=weight_cache_path if not args.dummy_weights else None,
            weight_dtype=ttnn.bfloat16,
            weight_key="weight",
        )
        
        # Attention and MLP will be added in subsequent implementation
        # For now, we define the forward pass structure
        self.attention = None  # TtMinistralAttention (to be implemented)
        self.mlp = None  # TtMinistralMLP (to be implemented)
    
    def forward(
        self,
        x: ttnn.Tensor,
        current_pos: int,
        rot_mats=None,
        mode: str = "prefill",
        kv_cache=None,
    ) -> ttnn.Tensor:
        """
        Forward pass through the decoder block.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            current_pos: Current position in sequence
            rot_mats: Rotary position embeddings
            mode: "prefill" or "decode"
            kv_cache: KV cache for decode mode
            
        Returns:
            Output tensor [batch, seq_len, dim]
        """
        # Pre-attention normalization
        normed_x = self.attention_norm(x)
        
        # Self-attention with residual
        if self.attention is not None:
            attn_out = self.attention(
                normed_x,
                current_pos=current_pos,
                rot_mats=rot_mats,
                mode=mode,
                kv_cache=kv_cache,
            )
        else:
            # Placeholder: pass through for initial testing
            attn_out = normed_x
        
        # Residual connection
        h = ttnn.add(x, attn_out)
        
        # Pre-FFN normalization
        normed_h = self.ffn_norm(h)
        
        # MLP with residual
        if self.mlp is not None:
            mlp_out = self.mlp(normed_h, mode=mode)
        else:
            # Placeholder: pass through for initial testing
            mlp_out = normed_h
        
        # Residual connection
        out = ttnn.add(h, mlp_out)
        
        return out
