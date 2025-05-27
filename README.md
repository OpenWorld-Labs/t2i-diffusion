# t2i-diffusion
Diffusion experiments

## Notes:
* FP4 inference on Blackwell, and FP8 on Hopper.
* Blacwell FP4, for post-training quantization (PTQ), and quantization-aware training (QAT).

## Commands:
```
docker build -t t2i-diffusion .
docker run --gpus all -it \
  -v "$PWD":/app \
  -v /home/$USER/data:/app/data \
  -v /home/$USER/logs:/app/logs \
  --workdir /app \
  --name t2i-diffusion \
  t2i-diffusion:latest
```

## Dev Notes:
Install dev tools from requirements-devtools.txt.

Run pre-commit hook: `pre-commit run --all-files`.

Example torch tensor type checking, which can also be merge with typing module of Python:

```
from torch import Tensor
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

# Enable type checking
patch_typeguard()

# Define tensor shapes using torchtyping
# Format: TensorType[<batch_dims>, <feature_dims>]
# Common dimensions:
# B: batch size
# C: channels
# H: height
# W: width
# D: depth
# L: sequence length
# E: embedding dimension

# for runtime type checking
@typechecked
def transformer_forward(
    x: TensorType["B", "L", "E", float],  # Input sequence
    mask: TensorType["B", "L", bool]  # Attention mask
) -> TensorType["B", "L", "E", float]:
    # Your code here
    return x

# For variable dimensions, use ... (ellipsis)
@typechecked
def flexible_batch(
    x: TensorType[..., "C", "H", "W", float]  # Any number of batch dimensions
) -> TensorType[..., "C", "H", "W", float]:
    return x
```