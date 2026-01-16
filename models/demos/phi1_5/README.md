# Phi-1.5 on Tenstorrent Wormhole

This directory contains the implementation of the `microsoft/phi-1.5` model for Tenstorrent hardware using the `ttnn` API.

## Model Overview

Phi-1.5 is a 1.3 billion parameter transformer-based model with a "parallel" configuration (Attention and MLP in parallel). It uses partial Rotary Positional Embeddings (RoPE).

- **Layers**: 24
- **Hidden Size**: 2048
- **Heads**: 32
- **Vocab Size**: 51200
- **Partial RoPE Factor**: 0.5 (32/64 dims)

## Implementation Progress

- [x] Model configuration and architecture planning
- [x] Decoder layers (Parallel Attention/MLP)
- [x] Partial RoPE handling
- [x] Skeleton verification with dummy weights
- [ ] Weight mapping from HuggingFace
- [ ] PCC Verification (> 0.99)
- [ ] Performance optimizations (Sharding)

## Running Tests

```bash
pytest models/demos/phi1_5/tests/test_model.py
```

## Running the Demo

(Instructions coming soon...)
