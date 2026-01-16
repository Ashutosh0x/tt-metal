# Ministral-8B-Instruct-2410 Port

This directory contains the ttnn implementation of `mistralai/Ministral-8B-Instruct-2410` for Tenstorrent hardware.

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Layers | 36 |
| Hidden Dim | 4096 |
| Query Heads | 32 |
| KV Heads | 8 (GQA) |
| Head Dim | 128 |
| Vocab Size | ~32k |
| Context | 128k (sliding window) |

## Quick Start

```bash
# Run demo
python models/demos/ministral_8b/demo/demo.py

# Run tests
pytest models/demos/ministral_8b/tests/test_model.py
```

## Directory Structure

```
ministral_8b/
├── README.md
├── demo/
│   └── demo.py          # Text generation demo
├── tests/
│   └── test_model.py    # PCC verification tests
└── tt/
    ├── model_config.py       # Model configuration
    ├── ministral_model.py    # Main model class
    ├── ministral_decoder.py  # Decoder block
    ├── ministral_attention.py # GQA attention
    └── ministral_mlp.py      # SwiGLU MLP
```

## Bounty Status

- **Issue**: #19420
- **Reward**: $500-$2500 (staged)
- **Status**: In Progress

## References

- [HuggingFace Model](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410)
- [Mistral AI Blog](https://mistral.ai/)
