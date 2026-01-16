# YOLOS-small implementation on Tenstorrent

This directory contains the Tenstorrent implementation of the YOLOS-small (You Only Look at One Sequence) model for object detection.

## Model Overview

YOLOS is a minimalist approach to object detection based on a vanilla Vision Transformer (ViT) architecture. It treats object detection as a sequence-to-sequence task, appending learnable "detection tokens" to the patch tokens.

- **Encoder**: 6-layer Vision Transformer
- **Hidden Size**: 384
- **Attention Heads**: 6
- **Detection Tokens**: 100
- **Dataset**: COCO (91 classes)

## Implementation Details

The implementation is written using `ttnn` for optimal performance on Tenstorrent hardware.

- [x] Initial skeleton and directory structure
- [x] Model configuration and arguments
- [x] ViT Encoder blocks (Attention + MLP)
- [x] Detection heads (Class + BBox MLP)
- [ ] Weight loading from HuggingFace
- [ ] Performance optimizations (sharding)
- [ ] PCC verification

## Running the Demo

(Instructions to follow...)

## Running Tests

(Instructions to follow...)
