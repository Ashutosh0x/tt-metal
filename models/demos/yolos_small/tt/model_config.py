# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
YOLOS-small Model Configuration

YOLOS (You Only Look at One Sequence) - ViT-based object detection.

Model Architecture (yolos-small):
- Backbone: ViT-Small encoder
- 6 layers (encoder blocks)
- 384 hidden dimension
- 6 attention heads
- 64 head dimension
- 1536 intermediate (MLP) dimension
- 100 detection tokens (learnable embeddings)
- No pixel_mask required (unlike DETR)
"""

import ttnn
from dataclasses import dataclass
from typing import Optional


@dataclass
class YolosConfig:
    """Configuration for YOLOS-small"""
    
    # Model architecture (from hustvl/yolos-small)
    hidden_size: int = 384
    num_hidden_layers: int = 6
    num_attention_heads: int = 6
    intermediate_size: int = 1536
    hidden_act: str = "gelu"
    
    # Vision config
    image_size: int = 512
    patch_size: int = 16
    num_channels: int = 3
    
    # Detection config
    num_detection_tokens: int = 100  # Learnable detection tokens
    num_labels: int = 91  # COCO classes
    
    # Derived
    head_dim: int = 64  # hidden_size // num_attention_heads
    num_patches: int = (512 // 16) ** 2  # 1024 patches for 512x512
    
    # Normalization
    layer_norm_eps: float = 1e-6
    
    # Runtime
    max_batch_size: int = 1


class TtYolosArgs:
    """
    Tenstorrent-specific model arguments for YOLOS-small.
    """
    
    def __init__(
        self,
        mesh_device,
        max_batch_size: int = 1,
        dummy_weights: bool = False,
    ):
        self.mesh_device = mesh_device
        self.max_batch_size = max_batch_size
        self.dummy_weights = dummy_weights
        
        # Model config
        self.config = YolosConfig(max_batch_size=max_batch_size)
        self.hidden_size = self.config.hidden_size
        self.num_hidden_layers = self.config.num_hidden_layers
        self.num_attention_heads = self.config.num_attention_heads
        self.intermediate_size = self.config.intermediate_size
        self.head_dim = self.config.head_dim
        self.image_size = self.config.image_size
        self.patch_size = self.config.patch_size
        self.num_detection_tokens = self.config.num_detection_tokens
        self.num_labels = self.config.num_labels
        self.num_patches = self.config.num_patches
        
        # Memory configurations
        self._setup_memory_configs()
    
    def _setup_memory_configs(self):
        """Setup ttnn memory configurations"""
        self.weight_mem_config = ttnn.DRAM_MEMORY_CONFIG
        self.l1_mem_config = ttnn.L1_MEMORY_CONFIG
        self.interleaved_mem_config = ttnn.DRAM_MEMORY_CONFIG
    
    def get_model_config(self):
        """Return model configuration dictionary"""
        return {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "head_dim": self.head_dim,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "num_detection_tokens": self.num_detection_tokens,
            "num_labels": self.num_labels,
            "max_batch_size": self.max_batch_size,
        }
