from dotenv import load_dotenv

import os
import torch
from huggingface_hub import hf_hub_download

load_dotenv()

import tarfile
import shutil
import numpy as np
import time
from PIL import Image
import random

from torch.utils.data import IterableDataset, DataLoader
import torch.distributed as dist

import threading
import io

from huggingface_hub.utils import disable_progress_bars

disable_progress_bars()

class RandomizedQueue:
    def __init__(self):
        self.items = []

    def add(self, item):
        self.items.append(item)

    def pop(self):
        if not self.items:
            return None
        idx = random.randint(0, len(self.items) - 1)
        return self.items.pop(idx)

def filter_img(img):
    return True

class HFImageDataset(IterableDataset):
    def __init__(self, rank = 0, world_size = 1):
        super().__init__()

        self.rank = rank
        self.world_size = world_size

        # Max to have in each queue
        self.max_tars = 2
        self.max_imgs = 1000

        self.tar_queue = RandomizedQueue()
        self.img_queue = RandomizedQueue()

        self.staging_dir = "hftmp"
        os.makedirs(self.staging_dir, exist_ok = True)

        # Start background threads
        self.tar_thread = threading.Thread(target=self.background_download_tars, daemon=True)
        self.img_thread = threading.Thread(target=self.background_load_images, daemon=True)
        self.tar_thread.start()
        self.img_thread.start()

    def random_sample_prefix(self):
        tars_per_rank = 4000 // self.world_size
        start_tar = self.rank * tars_per_rank
        end_tar = start_tar + tars_per_rank - 1
        tar_num = random.randint(start_tar, end_tar)
        return f"imagenet22k-train-{tar_num:04d}.tar"

    def background_download_tars(self):
        """
        Thread that downloads tars and adds them to the queue
        Hangs when queue is filled.
        """
        while True:
            if len(self.tar_queue.items) < self.max_tars:
                filename = self.random_sample_prefix()
                local_path = os.path.join(self.staging_dir, filename)

                try:
                    downloaded_file_path = hf_hub_download(
                        repo_id="timm/imagenet-22k-wds",
                        repo_type='dataset',
                        filename=filename
                    )
                    # Copy to staging dir
                    shutil.copy(downloaded_file_path, local_path)
                    self.tar_queue.add(local_path)
                except Exception as e:
                    print(f"Error downloading tar {filename}: {e}")
                    if os.path.exists(local_path):
                        os.remove(local_path)
            else:
                time.sleep(1)

    def background_load_images(self):
        """
        Thread that pops tars then loads images into image queue.
        Once queue is filled, hangs.
        Once one tar is exhausted, moves on to the next.
        """
        while True:
            if len(self.img_queue.items) < self.max_imgs:
                tar_path = self.tar_queue.pop()
                if tar_path is None:
                    time.sleep(1)
                    continue

                try:
                    with tarfile.open(tar_path) as tar:
                        members = tar.getmembers()
                        for member in members:
                            if len(self.img_queue.items) >= self.max_imgs:
                                break
                            if member.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                                f = tar.extractfile(member)
                                if f is not None:
                                    img_data = f.read()
                                    img = Image.open(io.BytesIO(img_data))
                                    if filter_img(img):
                                        self.img_queue.add(img)

                    os.remove(tar_path)
                except Exception as e:
                    print(f"Error processing tar {tar_path}: {e}")
                    if os.path.exists(tar_path):
                        os.remove(tar_path)
            else:
                time.sleep(1)

    def __iter__(self):
        while True:
            img = self.img_queue.pop()
            if img is not None:
                yield img
            else:
                time.sleep(0.1)

def collate_fn(images):
    # images list of PIL images
    processed = []
    for img in images:
        # Resize to 512x512
        img = img.convert("RGB")
        img = img.resize((256, 256), Image.BILINEAR)
        # Convert to tensor and normalize to [-1,1]
        img = torch.from_numpy(np.array(img)).float()
        img = img.permute(2, 0, 1) # [h,w,c] -> [c,h,w]
        img = (img / 127.5) - 1.0
        processed.append(img)

    # Stack into batch
    batch = torch.stack(processed)
    #batch = F.interpolate(batch, (512, 512), mode = 'bicubic')
    return batch # [b,c,h,w] in range [-1,1]

def get_loader(batch_size, **dataloader_kwargs):
    if 'collate_fn' in dataloader_kwargs:
        del dataloader_kwargs['collate_fn']

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    ds = HFImageDataset(rank, world_size)
    return DataLoader(ds, batch_size = batch_size, collate_fn=collate_fn, **dataloader_kwargs)

if __name__ == "__main__":
    import time

    loader = get_loader(256)

    for batch in loader:
        print(batch.shape)
        print(batch.min(), batch.max(), batch.mean())