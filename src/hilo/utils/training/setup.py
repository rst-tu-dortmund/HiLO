import torch
import numpy as np
import os


def set_fixed_seed(seed, deterministic=True):
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
            ":4096:8"  # Set cublas workspace config for reproducibility
        )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    np.random.seed(seed)
