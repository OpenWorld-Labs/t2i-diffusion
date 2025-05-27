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

Torch tensor type checking docs: https://docs.kidger.site/jaxtyping/api/array/

```
import torch
from jaxtyping import Float, Bool, jaxtyped
import typeguard

# for runtime type checking
@jaxtyped(typechecker=typeguard.typechecked)
def transformer_forward(
    x: Float[torch.Tensor, "batch seq 128"],
    mask: Bool[torch.Tensor, "seq 128"]
) -> Float[torch.Tensor, "batch seq 128"]:
    # Your code here
    return x
```