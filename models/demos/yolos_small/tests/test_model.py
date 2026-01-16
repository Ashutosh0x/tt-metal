# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from transformers import YolosForObjectDetection, AutoImageProcessor
from models.demos.yolos_small.tt.yolos_model import TtYolosModel
from models.demos.yolos_small.tt.model_config import TtYolosArgs
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)

def test_yolos_small_inference(device):
    # Load reference model
    model_name = "hustvl/yolos-small"
    hf_model = YolosForObjectDetection.from_pretrained(model_name)
    hf_model.eval()
    
    # Initialize tt model
    args = TtYolosArgs(device, dummy_weights=False)
    state_dict = hf_model.state_dict()
    
    tt_model = TtYolosModel(
        args=args,
        mesh_device=device,
        dtype=ttnn.bfloat16,
        state_dict=state_dict,
    )
    
    # Prepare dummy input
    batch_size = 1
    image_size = args.image_size
    # YOLOS expects pixel_values: [batch, 3, 512, 512]
    # In our implementation, we'll need to handle the patch embedding
    # For Stage 1, we can mock the input to the encoder
    seq_len = args.num_patches + args.num_detection_tokens + 1 # +1 for CLS
    hidden_size = args.hidden_size
    
    input_torch = torch.randn(batch_size, seq_len, hidden_size)
    input_tt = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    
    # Run inference
    tt_output = tt_model.forward(input_tt)
    
    # Verify outputs
    # Stage 1: Verify shapes and logic flow
    print(f"Logits shape: {tt_output['logits'].shape}")
    print(f"Boxes shape: {tt_output['pred_boxes'].shape}")
    
    assert tt_output["logits"].shape == (batch_size, seq_len, args.num_labels + 1)
    assert tt_output["pred_boxes"].shape == (batch_size, seq_len, 4)

if __name__ == "__main__":
    # For local debugging
    pass
