# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Ministral-8B Rotary Position Embeddings (RoPE)

Implements rotary position embeddings for the attention mechanism.
"""

import torch
import math
from typing import Tuple


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute the frequency tensor for RoPE.
    
    Args:
        dim: Dimension of the embeddings (head_dim)
        end: Maximum sequence length
        theta: Base for the frequency computation
        
    Returns:
        Tuple of (cos, sin) tensors of shape [end, dim//2]
    """
    # Compute inverse frequencies
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    
    # Create position indices
    t = torch.arange(end, dtype=torch.float32)
    
    # Outer product: [end, dim//2]
    freqs = torch.outer(t, freqs)
    
    # Compute cos and sin
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    
    return cos, sin


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to queries and keys.
    
    Args:
        xq: Query tensor [batch, seq_len, n_heads, head_dim]
        xk: Key tensor [batch, seq_len, n_kv_heads, head_dim]
        cos: Cosine frequencies [seq_len, head_dim//2]
        sin: Sine frequencies [seq_len, head_dim//2]
        
    Returns:
        Tuple of rotated (query, key) tensors
    """
    # Reshape for rotation
    xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    # Split into real and imaginary
    xq_real, xq_imag = xq_r[..., 0], xq_r[..., 1]
    xk_real, xk_imag = xk_r[..., 0], xk_r[..., 1]
    
    # Reshape cos/sin for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]
    sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]
    
    # Apply rotation
    xq_out_real = xq_real * cos - xq_imag * sin
    xq_out_imag = xq_real * sin + xq_imag * cos
    xk_out_real = xk_real * cos - xk_imag * sin
    xk_out_imag = xk_real * sin + xk_imag * cos
    
    # Interleave back
    xq_out = torch.stack([xq_out_real, xq_out_imag], dim=-1).flatten(-2)
    xk_out = torch.stack([xk_out_real, xk_out_imag], dim=-1).flatten(-2)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RoPEEmbedding:
    """
    Rotary Position Embedding manager.
    
    Precomputes and caches cos/sin tensors for efficiency.
    """
    
    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 131072,  # 128k context
        theta: float = 10000.0,
    ):
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute frequencies
        self.cos, self.sin = precompute_freqs_cis(head_dim, max_seq_len, theta)
    
    def get_rotary_emb(
        self,
        seq_len: int,
        start_pos: int = 0,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get rotary embeddings for a given sequence length and position.
        
        Args:
            seq_len: Length of the sequence
            start_pos: Starting position (for decode mode)
            device: Device to place tensors on
            
        Returns:
            Tuple of (cos, sin) for the sequence range
        """
        end_pos = start_pos + seq_len
        assert end_pos <= self.max_seq_len, f"Position {end_pos} exceeds max {self.max_seq_len}"
        
        cos = self.cos[start_pos:end_pos].to(device)
        sin = self.sin[start_pos:end_pos].to(device)
        
        return cos, sin
    
    def apply(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to queries and keys.
        
        Args:
            xq: Query tensor [batch, seq_len, n_heads, head_dim]
            xk: Key tensor [batch, seq_len, n_kv_heads, head_dim]
            start_pos: Starting position for decode mode
            
        Returns:
            Tuple of rotated (query, key) tensors
        """
        seq_len = xq.shape[1]
        cos, sin = self.get_rotary_emb(seq_len, start_pos, xq.device)
        return apply_rotary_emb(xq, xk, cos, sin)
