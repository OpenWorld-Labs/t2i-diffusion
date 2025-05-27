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