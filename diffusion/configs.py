from dataclasses import dataclass
from typing import List, Optional, Union

import yaml
from omegaconf import OmegaConf


@dataclass
class TransformerConfig:
    model_id: str = None

    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 384

    patch_size: int = 1
    channels: int = 3
    sample_size: int = 512

    cfg_prob: float = 0.1
    patch: bool = True
    uvit: bool = False


@dataclass
class TrainingConfig:
    trainer_id: str = None
    data_id: str = None

    target_batch_size: int = 128
    batch_size: int = 2

    epochs: int = 200

    opt: str = "AdamW"
    opt_kwargs: dict = None

    loss_weights: dict = None

    scheduler: str = None
    scheduler_kwargs: dict = None

    checkpoint_dir: str = "checkpoints/v0"  # Where checkpoints saved
    resume_ckpt: str = None

    # Distillation related
    teacher_ckpt: str = None
    teacher_cfg: str = None

    sample_interval: int = 1000
    save_interval: int = 1000

    # vae
    vae_cfg_path_or_paths: Optional[Union[List[str], str]] = None
    vae_ckpt_path_or_paths: Optional[Union[List[str], str]] = None
    vae_scale: float = 0.13


@dataclass
class WANDBConfig:
    name: str = None
    project: str = None
    run_name: str = None


@dataclass
class Config:
    model: TransformerConfig
    train: TrainingConfig
    wandb: WANDBConfig

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            raw_cfg = yaml.safe_load(f)

        cfg = OmegaConf.create(raw_cfg)
        return OmegaConf.structured(cls(**cfg))
