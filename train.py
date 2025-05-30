from diffusion.configs import Config
from diffusion.trainers import get_trainer_cls
from diffusion.utils.ddp import setup, cleanup

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to config YAML file")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config_path)
    
    trainer = get_trainer_cls(cfg.train.trainer_id)(
        cfg.train,
        cfg.wandb,
        cfg.model,
        *setup()
    )
    trainer.train()
    cleanup()