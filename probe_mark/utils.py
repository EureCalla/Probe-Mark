import os

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group


def ddp_setup():
    """Initialize distributed training environment."""
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def ddp_cleanup():
    """Clean up distributed training environment."""
    destroy_process_group()


def seed_everything(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
