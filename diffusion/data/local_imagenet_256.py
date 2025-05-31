from datasets import load_dataset

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms.functional as TF

def get_loader(batch_size):
    """Get ImageNet dataloader

    Args:
        batch_size: Batch size per GPU

    Returns:
        DataLoader for ImageNet dataset
    """
    world_size = 1
    global_rank = 0
    if dist.is_initialized():
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()

    dataset = load_dataset("benjamin-paine/imagenet-1k-256x256")
    train_dataset = dataset["train"]

    def collate_fn(examples):
        imgs = [example['image'].convert('RGB') for example in examples]
        imgs = [img.resize((256, 256), Image.BILINEAR) for img in imgs]
        tensors = [TF.to_tensor(img) * 2 - 1 for img in imgs]
        batch = torch.stack(tensors)
        return batch

    if world_size > 1:
        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=True
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        collate_fn=collate_fn
    )

    return train_loader