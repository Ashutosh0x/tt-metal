### 📝 Background

This bounty is for bringing up the [Depth-Anything-V2-Large model](https://huggingface.co/depth-anything/Depth-Anything-V2-Large) using TTNN APIs on Tenstorrent hardware (Wormhole or Blackhole). 
Depth Anything V2 is a state-of-the-art monocular depth estimation (MDE) model that provides accurate depth maps from single RGB images. It's trained on 595K synthetic labeled images and 62M+ real unlabeled images, offering:
More fine-grained depth details than V1 .Superior robustness compared to Stable Diffusion-based models (e.g., Marigold, Geowizard)
10x faster inference than SD-based models and Lightweight architecture.
The model is widely applicable in autonomous vehicles, robotics, AR/VR, 3D reconstruction, and computer vision applications requiring spatial understanding. 
The goal is to enable this  model to run on  TT hardware for high-throughput, low-latency inference for Depth Estimation.

### 🎯 What Success Looks Like

A successful submission will fulfill all requirements in the following stages. Payout is made after all three stages are completed.

### Stage 1 — Bring-Up

- Implement Depth-Anything-V2-Large using TTNN APIs (Python)
- Model runs on either N150 or N300 hardware with no errors
- Produces valid depth maps on sample images
- Output is verifiable (visualize depth maps, compare with PyTorch reference)
- Achieves baseline throughput target (at least 15 FPS at 518x518 resolution)
- Accuracy evaluation: PCC > 0.99 against PyTorch reference on test images
- Clear instructions for setup and running the model.

### Stage 2 — Basic Optimizations

- Use optimal sharded/interleaved memory configs for ViT encoder , conv layers
- Implement efficient sharding strategy for patch embedding and transformer blocks
- Fuse simple ops where possible (e.g., GELU+LayerNorm, attention softmax)
- Store intermediate activations in L1 where beneficial
- Use recommended TTNN/tt-metal ViT flows
- Leverage TT library of fused ops for attention and MLP blocks
- Optimize DPT (Dense Prediction Transformer) decoder head

### Stage 3 — Deeper Optimization

- Maximize core counts used per inference
- Implement deeper TT-specific optimizations (e.g., Efficient multi-head attention with optimal head sharding , Optimized patch embedding and position encoding ,Minimize tensor reshaping and transpositions in decoder, Efficient upsampling etc.)
- Minimize memory and TM (tensor manipulation) overheads
- Document any advanced tuning, known limitations, or trade-offs

### 🧭 Guidance & Starting Points

- Use the [TTNN model bring-up tech report](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/ttnn/TTNN-model-bringup.md) as your primary reference
- Reference ViT implementation in tt-metal : [vit documentation](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/ViT-TTNN/vit.md) [vit implementation](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/wormhole/vit) 
- Reference [Yolov4 implementation (PR 29157)](https://github.com/tenstorrent/tt-metal/pull/29157) for example end-to-end flows
- Use the official Depth-Anything-V2 repository for model architecture details
- Refer to [TT Fused ops](https://github.com/tenstorrent/tt-metal/pull/29236)  for optimization opportunities
- Target input resolution: start with 518x518x3 (standard for Depth Anything V2)
- Ask for help or file issues if ops are missing in TTNN.

### 🔎 Possible Approaches

- Start from an existing [implementation of the model](https://github.com/DepthAnything/Depth-Anything-V2) /pytorch reference and port layers one by one to TTNN.
- Validate each submodule’s output against CPU/PyTorch reference before full integration.
- Experiment with different sharding strategies and memory configs for convolutions.
- Use TTNN profiling tools to identify bottlenecks and areas for fusion.
- Open a draft PR early to get feedback on your approach.

### 📊 Result submission guidelines notes

Beyond the model implementation itself. Contributors must submit the following material as a proof of work. However, feel free to open a PR at any time if you want us checking you are on the right track. Just understand that payout is only made after all 3 stages are completed.

- Functional model implementation
- Validation logs (output correctness)
- Performance report + header for final review

Links:
- [Link to perf sheet](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/ttnn/TTNN-model-bringup.md#41-performance-sheet)
- [Perf header](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/profiling_ttnn_operations.html#perf-report-headers)

---

### 📚 Resources

- [TTNN model bring-up tech report](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/ttnn/TTNN-model-bringup.md)
- [CNN Bring-up & Optimization in TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/CNNs/cnn_optimizations.md)
- [Yolov4 TTNN implementation PR](https://github.com/tenstorrent/tt-metal/pull/29157)
- [Depth-Anything-V2-Large on Hugging Face](https://huggingface.co/depth-anything/Depth-Anything-V2-Large)
- [Depth-Anything-V2 Official Repository](https://github.com/DepthAnything/Depth-Anything-V2)
- [Depth Anything V2 Paper (arXiv:2406.09414)](https://arxiv.org/abs/2406.09414)
- [Depth Anything V2 Demo Space](https://huggingface.co/spaces/depth-anything/Depth-Anything-V2)
- [Project Page](https://depth-anything-v2.github.io)

- [TT fused ops PR](https://github.com/tenstorrent/tt-metal/pull/29236)
- [Perf report header](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/profiling_ttnn_operations.html#perf-report-headers)