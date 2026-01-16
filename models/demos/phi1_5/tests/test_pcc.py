# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from transformers import PhiForCausalLM, AutoConfig
from models.demos.phi1_5.tt.phi_model import TtPhiModel, precompute_rope_phi
from models.demos.phi1_5.tt.model_config import TtPhiArgs
from models.utility_functions import comp_pcc

def test_phi_1_5_pcc(device):
    # Load reference model
    model_name = "microsoft/phi-1.5"
    hf_model = PhiForCausalLM.from_pretrained(model_name)
    hf_model.eval()
    config = hf_model.config
    
    # Initialize args
    args = TtPhiArgs(device, dummy_weights=False)
    state_dict = hf_model.state_dict()
    
    # Initialize tt model
    dtype = ttnn.bfloat16
    tt_model = TtPhiModel(device, args, state_dict, dtype)
    
    # Precompute RoPE
    cos, sin = precompute_rope_phi(config, device)
    
    # Take a sample input
    batch = 1
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
    
    # Reference forward pass
    with torch.no_grad():
        hf_outputs = hf_model(input_ids)
        hf_logits = hf_outputs.logits
        
    # Tenstorrent forward pass
    input_tt = ttnn.from_torch(input_ids, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_logits = tt_model(input_tt, cos, sin)
    
    # Convert tt output back to torch for comparison
    tt_logits_torch = ttnn.to_torch(tt_logits)
    # tt_logits is [batch, 1, seq_len, vocab_size] -> need to reshape to [batch, seq_len, vocab_size]
    tt_logits_torch = tt_logits_torch.squeeze(1)
    
    # PCC Verification
    pcc_res = comp_pcc(hf_logits, tt_logits_torch, 0.99)
    print(f"PCC Result: {pcc_res}")
    
    assert pcc_res[0], f"PCC is too low: {pcc_res[1]}"

if __name__ == "__main__":
    # For local debugging
    pass
