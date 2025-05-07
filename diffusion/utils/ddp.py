import torch 
import torch.distributed as dist

def setup():
    try:
        dist.init_process_group(backend="nccl")
        global_rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()
        
        return global_rank, local_rank, world_size
    except:
        return 0, 0, 1

def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()