# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
YOLOS-small Main Model

Implements the full YOLOS model with:
- Patch embeddings
- Detection tokens
- ViT encoder
- Detection heads (class + bbox)
"""

import ttnn
import torch
from typing import Optional, Dict, Any
from models.common.lightweightmodule import LightweightModule
from models.demos.yolos_small.tt.model_config import TtYolosArgs
from models.demos.yolos_small.tt.yolos_encoder import TtYolosEncoderBlock


class TtYolosModel(LightweightModule):
    """
    YOLOS (You Only Look at One Sequence) model for object detection.
    """
    
    def __init__(
        self,
        args: TtYolosArgs,
        mesh_device,
        dtype,
        state_dict=None,
        weight_cache_path=None,
    ):
        super().__init__()
        
        self.args = args
        self.mesh_device = mesh_device
        self.dtype = dtype
        
        # Embeddings
        self._setup_embeddings(state_dict, dtype)
        
        # Encoder blocks
        self.encoder_blocks = []
        for layer_num in range(args.num_hidden_layers):
            block = TtYolosEncoderBlock(
                args, mesh_device, dtype, state_dict, layer_num, weight_cache_path
            )
            self.encoder_blocks.append(block)
        
        # Final LayerNorm
        self._setup_final_norm(state_dict, dtype)
        
        # Detection heads
        self._setup_detection_heads(state_dict, dtype)
    
    def _setup_embeddings(self, state_dict, dtype):
        """Setup patch embeddings and detection tokens"""
        if state_dict is not None and not self.args.dummy_weights:
            # Patch embedding projection
            # In YOLOS, this is a Conv2d(3, 384, kernel_size=16, stride=16)
            # We map it to a Linear layer for Stage 1 bring-up
            patch_weight = state_dict.get("vit.embeddings.patch_embeddings.projection.weight")
            patch_bias = state_dict.get("vit.embeddings.patch_embeddings.projection.bias")
            
            # Reshape patch_weight from [384, 3, 16, 16] to [768, 384] (Transposed for linear)
            if patch_weight is not None:
                patch_weight = patch_weight.permute(2, 3, 1, 0).reshape(-1, self.args.hidden_size)
            
            # CLS token [1, 1, 384] and detection tokens [1, 100, 384]
            cls_token = state_dict.get("vit.embeddings.cls_token")
            det_tokens = state_dict.get("vit.embeddings.detection_tokens")
            
            # Position embeddings [1, 1125, 384]
            # 1 (CLS) + 100 (DET) + 1024 (PATCHES) = 1125
            pos_embed = state_dict.get("vit.embeddings.position_embeddings")
            
            self.patch_weight = self._to_ttnn(patch_weight, dtype)
            self.patch_bias = self._to_ttnn(patch_bias, dtype) if patch_bias is not None else None
            self.cls_token = self._to_ttnn(cls_token, dtype) if cls_token is not None else None
            self.det_tokens = self._to_ttnn(det_tokens, dtype) if det_tokens is not None else None
            self.pos_embed = self._to_ttnn(pos_embed, dtype) if pos_embed is not None else None
        else:
            self.patch_weight = self.patch_bias = None
            self.cls_token = self.det_tokens = self.pos_embed = None
    
    def _setup_final_norm(self, state_dict, dtype):
        """Setup final LayerNorm"""
        if state_dict is not None and not self.args.dummy_weights:
            ln_weight = state_dict.get("vit.layernorm.weight")
            ln_bias = state_dict.get("vit.layernorm.bias")
            self.final_ln_weight = self._to_ttnn(ln_weight, dtype)
            self.final_ln_bias = self._to_ttnn(ln_bias, dtype) if ln_bias is not None else None
        else:
            self.final_ln_weight = self.final_ln_bias = None
    
    def _setup_detection_heads(self, state_dict, dtype):
        """Setup class and bbox prediction heads"""
        if state_dict is not None and not self.args.dummy_weights:
            # Class head (MLP: Linear(384, 384) -> ReLU -> Linear(384, 92))
            # COCO has 91 classes + 1 for 'no object'
            cls_w1 = state_dict.get("class_labels_classifier.0.weight")
            cls_b1 = state_dict.get("class_labels_classifier.0.bias")
            cls_w2 = state_dict.get("class_labels_classifier.3.weight")
            cls_b2 = state_dict.get("class_labels_classifier.3.bias")
            
            # BBox head (MLP: Linear(384, 384) -> ReLU -> Linear(384, 384) -> ReLU -> Linear(384, 4) -> Sigmoid)
            box_w1 = state_dict.get("bbox_predictor.0.weight")
            box_b1 = state_dict.get("bbox_predictor.0.bias")
            box_w2 = state_dict.get("bbox_predictor.3.weight")
            box_b2 = state_dict.get("bbox_predictor.3.bias")
            box_w3 = state_dict.get("bbox_predictor.6.weight")
            box_b3 = state_dict.get("bbox_predictor.6.bias")
            
            self.cls_w1 = self._to_ttnn(cls_w1, dtype)
            self.cls_b1 = self._to_ttnn(cls_b1, dtype)
            self.cls_w2 = self._to_ttnn(cls_w2, dtype)
            self.cls_b2 = self._to_ttnn(cls_b2, dtype)
            
            self.box_w1 = self._to_ttnn(box_w1, dtype)
            self.box_b1 = self._to_ttnn(box_b1, dtype)
            self.box_w2 = self._to_ttnn(box_w2, dtype)
            self.box_b2 = self._to_ttnn(box_b2, dtype)
            self.box_w3 = self._to_ttnn(box_w3, dtype)
            self.box_b3 = self._to_ttnn(box_b3, dtype)
        else:
            self.cls_w1 = self.cls_b1 = self.cls_w2 = self.cls_b2 = None
            self.box_w1 = self.box_b1 = self.box_w2 = self.box_b2 = self.box_w3 = self.box_b3 = None
    
    def _to_ttnn(self, tensor, dtype):
        """Convert tensor to ttnn"""
        if tensor is None:
            return None
        if len(tensor.shape) == 2:
            tensor = tensor.T.contiguous()
        return ttnn.from_torch(
            tensor,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    
    def forward(
        self,
        pixel_values: ttnn.Tensor,
    ) -> Dict[str, ttnn.Tensor]:
        """
        Forward pass.
        
        Args:
            pixel_values: Input patches or image. 
                         Stage 1: [batch, num_patches, 768]
            
        Returns:
            Dict with 'logits' and 'pred_boxes'
        """
        batch_size = pixel_values.shape[0]
        
        # 1. Patch projection
        # [B, 1024, 768] -> [B, 1024, 384]
        x = ttnn.linear(pixel_values, self.patch_weight, bias=self.patch_bias)
        
        # 2. Prepend CLS and DET tokens
        # [1, 1, 384] and [1, 100, 384]
        # In ttnn, we need to repeat tokens for batch_size if > 1
        # For Stage 1, we assume batch_size=1
        x = ttnn.concat([self.cls_token, self.det_tokens, x], dim=1)
        
        # 3. Add position embeddings [1, 1125, 384]
        if self.pos_embed is not None:
            x = ttnn.add(x, self.pos_embed)
        
        # 4. Encoder
        for block in self.encoder_blocks:
            x = block.forward(x)
        
        # 5. Final LayerNorm
        if self.final_ln_weight is not None:
            x = ttnn.layer_norm(x, weight=self.final_ln_weight, bias=self.final_ln_bias)
        
        # 6. Detection heads (Class + BBox)
        # Class predictions
        if self.cls_w1 is not None:
            cls_hidden = ttnn.linear(x, self.cls_w1, bias=self.cls_b1)
            cls_hidden = ttnn.relu(cls_hidden)
            logits = ttnn.linear(cls_hidden, self.cls_w2, bias=self.cls_b2)
        else:
            logits = x
        
        # Box predictions
        if self.box_w1 is not None:
            box_hidden = ttnn.linear(x, self.box_w1, bias=self.box_b1)
            box_hidden = ttnn.relu(box_hidden)
            box_hidden = ttnn.linear(box_hidden, self.box_w2, bias=self.box_b2)
            box_hidden = ttnn.relu(box_hidden)
            pred_boxes = ttnn.linear(box_hidden, self.box_w3, bias=self.box_b3)
            pred_boxes = ttnn.sigmoid(pred_boxes)
        else:
            pred_boxes = x
        
        # Returns all outputs (CLS + DET + PATCHES)
        # Usually DET tokens are at indices [1:101]
        return {
            "logits": logits,
            "pred_boxes": pred_boxes,
        }


def custom_preprocessor(model, name):
    """Preprocessor for weight loading"""
    return {}
