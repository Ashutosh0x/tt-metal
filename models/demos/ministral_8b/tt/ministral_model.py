# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Ministral-8B-Instruct-2410 Model

Complete transformer model implementation using ttnn.
"""

import ttnn
import torch
from typing import Optional, List
from models.common.lightweightmodule import LightweightModule
from models.demos.ministral_8b.tt.model_config import TtMinistralArgs
from models.demos.ministral_8b.tt.ministral_decoder import TtMinistralDecoderBlock


class TtMinistralModel(LightweightModule):
    """
    Ministral-8B Transformer Model.
    
    Architecture:
    - Token Embedding
    - 36 Decoder Blocks (RMSNorm + GQA + SwiGLU MLP)
    - Final RMSNorm
    - LM Head
    """
    
    def __init__(
        self,
        args: TtMinistralArgs,
        mesh_device,
        dtype=ttnn.bfloat16,
        state_dict=None,
        weight_cache_path=None,
    ):
        super().__init__()
        
        self.args = args
        self.mesh_device = mesh_device
        self.dtype = dtype
        
        self.dim = args.dim
        self.n_layers = args.n_layers
        self.vocab_size = args.vocab_size
        
        # Token embedding
        self._load_embeddings(state_dict, dtype)
        
        # Decoder blocks
        self.layers: List[TtMinistralDecoderBlock] = []
        for layer_num in range(self.n_layers):
            layer = TtMinistralDecoderBlock(
                args=args,
                mesh_device=mesh_device,
                dtype=dtype,
                state_dict=state_dict,
                layer_num=layer_num,
                weight_cache_path=weight_cache_path,
            )
            self.layers.append(layer)
        
        # Final normalization
        self._load_final_norm(state_dict, dtype)
        
        # LM head
        self._load_lm_head(state_dict, dtype)
    
    def _load_embeddings(self, state_dict, dtype):
        """Load token embeddings"""
        if state_dict is not None and not self.args.dummy_weights:
            embed_weight = state_dict.get("model.embed_tokens.weight")
            if embed_weight is not None:
                self.tok_embeddings = ttnn.from_torch(
                    embed_weight,
                    dtype=dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            else:
                self.tok_embeddings = None
        else:
            self.tok_embeddings = None
    
    def _load_final_norm(self, state_dict, dtype):
        """Load final RMSNorm"""
        if state_dict is not None and not self.args.dummy_weights:
            norm_weight = state_dict.get("model.norm.weight")
            if norm_weight is not None:
                self.norm_weight = ttnn.from_torch(
                    norm_weight,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            else:
                self.norm_weight = None
        else:
            self.norm_weight = None
    
    def _load_lm_head(self, state_dict, dtype):
        """Load LM head projection"""
        if state_dict is not None and not self.args.dummy_weights:
            lm_head_weight = state_dict.get("lm_head.weight")
            if lm_head_weight is not None:
                self.lm_head = ttnn.from_torch(
                    lm_head_weight.T.contiguous(),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            else:
                self.lm_head = None
        else:
            self.lm_head = None
    
    def forward(
        self,
        tokens: ttnn.Tensor,
        current_pos: int = 0,
        rot_mats=None,
        mode: str = "prefill",
        kv_cache=None,
    ) -> ttnn.Tensor:
        """
        Forward pass through the model.
        
        Args:
            tokens: Input token IDs [batch, seq_len]
            current_pos: Current position for decode mode
            rot_mats: Rotary embeddings (cos, sin)
            mode: "prefill" or "decode"
            kv_cache: List of KV caches per layer
            
        Returns:
            Logits tensor [batch, seq_len, vocab_size]
        """
        # Token embedding
        if self.tok_embeddings is not None:
            h = ttnn.embedding(tokens, self.tok_embeddings)
        else:
            # Placeholder for testing
            batch_size = tokens.shape[0]
            seq_len = tokens.shape[1]
            h = ttnn.zeros([batch_size, seq_len, self.dim], device=self.mesh_device)
        
        # Pass through decoder layers
        for layer_idx, layer in enumerate(self.layers):
            layer_kv_cache = kv_cache[layer_idx] if kv_cache else None
            h = layer.forward(
                h,
                current_pos=current_pos,
                rot_mats=rot_mats,
                mode=mode,
                kv_cache=layer_kv_cache,
            )
        
        # Final normalization
        if self.norm_weight is not None:
            h = ttnn.rms_norm(h, epsilon=self.args.norm_eps, weight=self.norm_weight)
        
        # LM head projection
        if self.lm_head is not None:
            logits = ttnn.linear(h, self.lm_head)
        else:
            logits = h
        
        return logits


def custom_preprocessor(torch_model, name):
    """
    Preprocess PyTorch model weights for ttnn.
    
    Args:
        torch_model: HuggingFace model
        name: Model name/path
        
    Returns:
        Dictionary of preprocessed parameters
    """
    state_dict = torch_model.state_dict()
    
    # The state dict can be used directly - ttnn conversion happens in the model
    return state_dict
