# PR Description: [YOLOS-small] Initial Implementation

## Description
This PR implements **YOLOS-small** (You Only Look at One Sequence) for object detection on Tenstorrent hardware using `ttnn`.

YOLOS is a Vision Transformer based object detection model that treats detection as a pure sequence-to-sequence problem by appending learnable detection tokens to the patch sequence.

### Key Features
- **Architecture**: 6-layer ViT-Small backbone (384 hidden dim, 6 heads).
- **Detection Head**: 100 detection tokens with 3-layer MLP for BBox prediction and Linear head for class labels (COCO 91 classes).
- **Weight Loading**: Full mapping from `hustvl/yolos-small` HuggingFace weights.
- **Inference Flow**: Verified with a demo script and PCC tests.
- **Optimizations**: Initial bring-up uses bfloat16/bfloat8_b with DRAM memory configs.

### Directory Structure
- `models/demos/yolos_small/tt/`: Core implementation.
- `models/demos/yolos_small/demo/`: Functional demo.
- `models/demos/yolos_small/tests/`: PCC and shape verification tests.

## Bounty Status
- Claimed: #30874
- Stage 1: Basic Bring-up (Complete)
- Stage 2: Performance (Pending Assignment)

## Verification
- PCC > 0.99 verified vs HF reference.
- Run `tests/test_model.py` to verify output shapes and logic.
- Run `demo/demo.py` for image inference demonstration.
