import numpy as np
import torch
import random
import torch
import os

def setSeed():
    seed = 490
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def log(*args):
    debug = os.getenv("DEBUG_MODE", "false").lower() == "true"
    if debug:
        print(args)
