# t2i-diffusion
Diffusion experiments

## Notes:
> Source: https://developer.nvidia.com/blog/nvidia-blackwell-delivers-world-record-deepseek-r1-inference-performance/

Use NVIDIA TensorRT (10.8) ecosystem for inference with FP4 on Blackwell, and FP8 on Hopper.
1. TensorRT Model Optimizer: quantization, distillation, pruning, sparsity, speculative decoding. **0.25 version** supports Blacwell FP4, for post-training quantization (PTQ), and quantization-aware training (QAT).
2. TensorRT-LLM: Inference framework. **0.17 version** supports FP4. In-flight batching, KV cache, speculative decoding.

* Speeds up Flux.1 diffusion models 3x (throughput), and VRAM usage reduced by 2.6x.
* FP4 has a normal, and a low-VRAM mode. Use low-VRAM mode for low mem GPUs like GeForce RTX 5070.

Example: https://github.com/NVIDIA/TensorRT/tree/release/10.8/demo/Diffusion

* Different usage options: There's Torch-TensorRT, and then Python API for TensorRT.

* Environment compatibility issues: Python version<=3.10.x is indicated on github readme. Further, nvidia driver and CUDA version will be something to be careful about.
