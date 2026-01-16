# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Ministral-8B Demo

Text generation using the Ministral-8B-Instruct-2410 model on Tenstorrent hardware.
"""

import argparse
import time
import torch
import ttnn
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.demos.ministral_8b.tt.model_config import TtMinistralArgs
from models.demos.ministral_8b.tt.ministral_model import TtMinistralModel


def main():
    parser = argparse.ArgumentParser(description="Ministral-8B Demo")
    parser.add_argument("--prompt", type=str, default="Hello, I am", help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--device_id", type=int, default=0, help="Device ID")
    parser.add_argument("--dummy", action="store_true", help="Use dummy weights for testing")
    args = parser.parse_args()
    
    print(f"[Demo] Initializing Ministral-8B on device {args.device_id}")
    
    # Open device
    device = ttnn.open_device(device_id=args.device_id)
    
    try:
        model_name = "mistralai/Ministral-8B-Instruct-2410"
        
        # Load tokenizer
        print(f"[Demo] Loading tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load reference model for weights (or use dummy)
        if args.dummy:
            print("[Demo] Using dummy weights for testing")
            state_dict = None
        else:
            print(f"[Demo] Loading weights from {model_name}...")
            ref_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
            state_dict = ref_model.state_dict()
            del ref_model  # Free memory
        
        # Initialize TT model
        print("[Demo] Initializing TT model...")
        tt_args = TtMinistralArgs(
            mesh_device=device,
            max_batch_size=1,
            max_seq_len=2048,
            dummy_weights=args.dummy,
        )
        
        tt_model = TtMinistralModel(
            args=tt_args,
            mesh_device=device,
            dtype=ttnn.bfloat16,
            state_dict=state_dict,
        )
        
        # Tokenize input
        print(f"[Demo] Input: {args.prompt}")
        inputs = tokenizer(args.prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"]
        
        # Generate tokens
        print(f"[Demo] Generating {args.max_tokens} tokens...")
        start_time = time.time()
        
        generated_ids = input_ids.clone()
        
        for i in range(args.max_tokens):
            # Convert to ttnn
            tokens_tt = ttnn.from_torch(generated_ids, device=device)
            
            # Forward pass
            logits = tt_model.forward(tokens_tt, mode="prefill")
            
            # Get next token
            logits_torch = ttnn.to_torch(logits)
            next_token_logits = logits_torch[0, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)
            
            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
        
        end_time = time.time()
        
        # Decode output
        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        print(f"\n[Demo] Output: {output_text}")
        print(f"\n[Demo] Generated {generated_ids.shape[1] - input_ids.shape[1]} tokens in {end_time - start_time:.2f}s")
        print(f"[Demo] Tokens/sec: {(generated_ids.shape[1] - input_ids.shape[1]) / (end_time - start_time):.2f}")
        
    finally:
        ttnn.close_device(device)
        print("[Demo] Device closed")


if __name__ == "__main__":
    main()
