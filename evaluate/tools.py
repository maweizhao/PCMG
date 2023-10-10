import yaml


def format_metrics(metrics, formatter="{:.6}"):
    newmetrics = {}
    for key, val in metrics.items():
        newmetrics[key] = formatter.format(val)
    return newmetrics


def save_metrics(path, metrics):
    with open(path, "w") as yfile:
        yaml.dump(metrics, yfile)

        
def load_metrics(path):
    with open(path, "r") as yfile:
        string = yfile.read()
        return yaml.load(string, yaml.loader.BaseLoader)

import numpy as np
import torch
import random


def fixseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


SEED = 10
EVALSEED = 0
# Provoc warning: not fully functionnal yet
# torch.set_deterministic(True)
torch.backends.cudnn.benchmark = False

fixseed(SEED)