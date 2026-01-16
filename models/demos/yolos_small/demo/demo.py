# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from PIL import Image
from transformers import YolosForObjectDetection, AutoImageProcessor
from models.demos.yolos_small.tt.yolos_model import TtYolosModel
from models.demos.yolos_small.tt.model_config import TtYolosArgs

def run_yolos_small_demo(device):
    # Load model and processor
    model_name = "hustvl/yolos-small"
    processor = AutoImageProcessor.from_pretrained(model_name)
    hf_model = YolosForObjectDetection.from_pretrained(model_name)
    hf_model.eval()
    
    # Initialize tt model
    args = TtYolosArgs(device)
    state_dict = hf_model.state_dict()
    
    tt_model = TtYolosModel(
        args=args,
        mesh_device=device,
        dtype=ttnn.bfloat16,
        state_dict=state_dict,
    )
    
    # Load and preprocess image
    # For now, use a dummy image or a sample from COCO
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(torch.hub.download_url_to_file(url, "cats.jpg") if "cats.jpg" not in locals() else "cats.jpg")
    inputs = processor(images=image, return_tensors="pt")
    
    # Run reference inference
    with torch.no_grad():
        hf_outputs = hf_model(**inputs)
    
    # Prepare input for ttnn
    # Stage 1: Manual patch embedding or mock input
    pixel_values = inputs["pixel_values"]  # [1, 3, 512, 512]
    print(f"Input image shape: {pixel_values.shape}")
    
    # Patchify: [1, 3, 512, 512] -> [1, 1024, 768]
    # 512/16 = 32 patches in each dimension
    B, C, H, W = pixel_values.shape
    P = args.patch_size
    assert H % P == 0 and W % P == 0
    
    patches = pixel_values.unfold(2, P, P).unfold(3, P, P) # [B, C, 32, 32, 16, 16]
    patches = patches.permute(0, 2, 3, 4, 5, 1) # [B, 32, 32, 16, 16, 3]
    patches = patches.reshape(B, (H//P)*(W//P), P*P*C) # [B, 1024, 768]
    
    tt_input = ttnn.from_torch(
        patches,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    
    tt_output = tt_model.forward(tt_input)
    
    # Post-process results
    logits = tt_output["logits"]
    bboxes = tt_output["pred_boxes"]
    
    print(f"Logits shape: {logits.shape}")
    print(f"BBoxes shape: {bboxes.shape}")
    
    # Print statistics
    print("Demo completed successfully!")

if __name__ == "__main__":
    # For local execution
    pass
