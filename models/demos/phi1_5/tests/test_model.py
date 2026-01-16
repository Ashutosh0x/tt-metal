# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from models.demos.phi1_5.tt.phi_model import TtPhiModel, precompute_rope_phi
from models.demos.phi1_5.tt.model_config import TtPhiArgs

def test_phi_1_5_skeleton(device):
    # Initialize args
    args = TtPhiArgs(device, dummy_weights=True)
    config = args.config
    
    # Create mock state dict
    state_dict = {}
    dtype = ttnn.bfloat16
    
    # Embeddings
    state_dict["model.embed_tokens.weight"] = torch.randn(config.vocab_size, config.hidden_size)
    
    # Layers
    for i in range(config.num_hidden_layers):
        prefix = f"model.layers.{i}"
        state_dict[f"{prefix}.input_layernorm.weight"] = torch.randn(config.hidden_size)
        state_dict[f"{prefix}.input_layernorm.bias"] = torch.randn(config.hidden_size)
        
        # Attn
        attn_prefix = f"{prefix}.self_attn"
        state_dict[f"{attn_prefix}.q_proj.weight"] = torch.randn(config.hidden_size, config.hidden_size)
        state_dict[f"{attn_prefix}.k_proj.weight"] = torch.randn(config.hidden_size, config.hidden_size)
        state_dict[f"{attn_prefix}.v_proj.weight"] = torch.randn(config.hidden_size, config.hidden_size)
        state_dict[f"{attn_prefix}.q_proj.bias"] = torch.randn(config.hidden_size)
        state_dict[f"{attn_prefix}.k_proj.bias"] = torch.randn(config.hidden_size)
        state_dict[f"{attn_prefix}.v_proj.bias"] = torch.randn(config.hidden_size)
        state_dict[f"{attn_prefix}.dense.weight"] = torch.randn(config.hidden_size, config.hidden_size)
        state_dict[f"{attn_prefix}.dense.bias"] = torch.randn(config.hidden_size)
        
        # MLP
        mlp_prefix = f"{prefix}.mlp"
        state_dict[f"{mlp_prefix}.fc1.weight"] = torch.randn(config.intermediate_size, config.hidden_size)
        state_dict[f"{mlp_prefix}.fc1.bias"] = torch.randn(config.intermediate_size)
        state_dict[f"{mlp_prefix}.fc2.weight"] = torch.randn(config.hidden_size, config.intermediate_size)
        state_dict[f"{mlp_prefix}.fc2.bias"] = torch.randn(config.hidden_size)
        
    # Final
    state_dict["model.final_layernorm.weight"] = torch.randn(config.hidden_size)
    state_dict["model.final_layernorm.bias"] = torch.randn(config.hidden_size)
    state_dict["lm_head.weight"] = torch.randn(config.vocab_size, config.hidden_size)
    state_dict["lm_head.bias"] = torch.randn(config.vocab_size)
    
    # Initialize model
    tt_model = TtPhiModel(device, args, state_dict, dtype)
    
    # Precompute RoPE
    cos, sin = precompute_rope_phi(config, device)
    
    # Prepare input
    batch = 1
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
    input_tt = ttnn.from_torch(input_ids, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    
    # Forward pass
    logits = tt_model(input_tt, cos, sin)
    
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (batch, 1, seq_len, config.vocab_size)

if __name__ == "__main__":
    # For local debugging
    pass
