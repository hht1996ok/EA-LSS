import numpy as np
import torch

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    elif isinstance(x, (int, float)):
        # modified
        d = np.array([x],dtype=np.float32)
        return torch.from_numpy(d).float(), True
    return x, False

