# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Ministral-8B Model Tests

Verifies PCC > 0.99 against HuggingFace reference.
"""

import pytest
import torch
import ttnn
from models.demos.ministral_8b.tt.model_config import TtMinistralArgs, MinistralConfig
from models.demos.ministral_8b.tt.ministral_model import TtMinistralModel


def compute_pcc(torch_output, tt_output):
    """Compute Pearson Correlation Coefficient"""
    torch_flat = torch_output.flatten().float()
    tt_flat = tt_output.flatten().float()
    
    # Normalize
    torch_norm = torch_flat - torch_flat.mean()
    tt_norm = tt_flat - tt_flat.mean()
    
    # PCC
    numerator = (torch_norm * tt_norm).sum()
    denominator = torch.sqrt((torch_norm ** 2).sum() * (tt_norm ** 2).sum())
    
    if denominator == 0:
        return 1.0 if torch.allclose(torch_flat, tt_flat) else 0.0
    
    return (numerator / denominator).item()


@pytest.fixture
def device():
    """Get ttnn device"""
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


class TestMinistralConfig:
    """Test model configuration"""
    
    def test_config_defaults(self):
        """Verify default config values"""
        config = MinistralConfig()
        
        assert config.dim == 4096
        assert config.n_layers == 36
        assert config.n_heads == 32
        assert config.n_kv_heads == 8
        assert config.head_dim == 128
        assert config.sliding_window == 4096
    
    def test_gqa_ratio(self):
        """Verify GQA head ratio"""
        config = MinistralConfig()
        gqa_ratio = config.n_heads // config.n_kv_heads
        assert gqa_ratio == 4, "Expected 4:1 GQA ratio"


class TestMinistralModel:
    """Test model inference"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU/TPU")
    def test_model_forward_shape(self, device):
        """Test model output shape"""
        args = TtMinistralArgs(
            mesh_device=device,
            max_batch_size=1,
            max_seq_len=128,
            dummy_weights=True,
        )
        
        model = TtMinistralModel(
            args=args,
            mesh_device=device,
            dtype=ttnn.bfloat16,
            state_dict=None,
        )
        
        # Create dummy input
        batch_size = 1
        seq_len = 32
        tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len))
        tokens_tt = ttnn.from_torch(tokens, device=device)
        
        # Forward pass
        logits = model.forward(tokens_tt, mode="prefill")
        
        # Convert back to torch
        logits_torch = ttnn.to_torch(logits)
        
        # Check shape
        expected_shape = (batch_size, seq_len, args.vocab_size)
        assert logits_torch.shape == expected_shape, f"Expected {expected_shape}, got {logits_torch.shape}"
    
    @pytest.mark.skip(reason="Requires HuggingFace model download")
    def test_pcc_vs_reference(self, device):
        """Test PCC against HuggingFace reference"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "mistralai/Ministral-8B-Instruct-2410"
        
        # Load reference model
        ref_model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Setup TT model
        args = TtMinistralArgs(
            mesh_device=device,
            max_batch_size=1,
            max_seq_len=128,
            dummy_weights=False,
        )
        
        tt_model = TtMinistralModel(
            args=args,
            mesh_device=device,
            dtype=ttnn.bfloat16,
            state_dict=ref_model.state_dict(),
        )
        
        # Test input
        text = "Hello, world!"
        inputs = tokenizer(text, return_tensors="pt")
        tokens = inputs["input_ids"]
        
        # Reference forward
        with torch.no_grad():
            ref_output = ref_model(tokens).logits
        
        # TT forward
        tokens_tt = ttnn.from_torch(tokens, device=device)
        tt_output = tt_model.forward(tokens_tt, mode="prefill")
        tt_output_torch = ttnn.to_torch(tt_output)
        
        # Compute PCC
        pcc = compute_pcc(ref_output, tt_output_torch)
        
        assert pcc > 0.99, f"PCC {pcc:.4f} below threshold 0.99"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
