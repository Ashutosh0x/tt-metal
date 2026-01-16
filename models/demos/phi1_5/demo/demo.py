# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from transformers import AutoTokenizer, PhiForCausalLM
from models.demos.phi1_5.tt.phi_model import TtPhiModel, precompute_rope_phi
from models.demos.phi1_5.tt.model_config import TtPhiArgs

def run_phi_demo(device, prompt):
    # Load model and tokenizer
    model_name = "microsoft/phi-1.5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = PhiForCausalLM.from_pretrained(model_name)
    hf_model.eval()
    config = hf_model.config
    
    # Initialize tt model
    args = TtPhiArgs(device, dummy_weights=False)
    state_dict = hf_model.state_dict()
    tt_model = TtPhiModel(device, args, state_dict, ttnn.bfloat16)
    
    # Precompute RoPE
    cos, sin = precompute_rope_phi(config, device)
    
    # Initialize KV cache
    batch = 1
    max_seq_len = 512
    kv_cache = ttnn.model.init_kv_cache(device, args, batch, max_seq_len, ttnn.bfloat16) # Use helper from __init__ or ttnn.model
    # Wait, ttnn.model doesn't have it, I should use my helper
    from models.demos.phi1_5.tt.phi_model import init_kv_cache
    kv_cache = init_kv_cache(device, args, batch, max_seq_len, ttnn.bfloat16)

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    curr_input_ids = input_ids
    
    print(f"Prompt: {prompt}")
    
    generated = input_ids
    max_new_tokens = 50
    curr_len = 0
    
    for i in range(max_new_tokens):
        # Tenstorrent forward pass
        # In decode mode, we only pass the last token if we have a KV cache
        input_tt = ttnn.from_torch(curr_input_ids, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        
        with torch.no_grad():
            tt_logits = tt_model(input_tt, cos, sin, layer_past=kv_cache, layer_past_len=curr_len)
            
        # Get next token
        logits_torch = ttnn.to_torch(tt_logits).squeeze(1) # [batch, new_token_len, vocab]
        next_token_logits = logits_torch[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        generated = torch.cat([generated, next_token], dim=-1)
        
        # Prepare for next token
        curr_len += curr_input_ids.shape[1]
        curr_input_ids = next_token
        
        # Stop if EOS
        if next_token.item() == tokenizer.eos_token_id:
            break
            
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Generated text: {output_text}")

if __name__ == "__main__":
    # Example usage:
    # run_phi_demo(device, "Instruct: How to make a cake?\nOutput:")
    pass
