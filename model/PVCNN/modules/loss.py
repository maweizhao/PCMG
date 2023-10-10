import torch.nn as nn

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.dirname(BASE_DIR)
PROJECT_ROOT_DIR = os.path.dirname(PROJECT_ROOT_DIR)
PROJECT_ROOT_DIR = os.path.dirname(PROJECT_ROOT_DIR)
#print(PROJECT_ROOT_DIR)
sys.path.append(PROJECT_ROOT_DIR)

import model.PVCNN.modules.functional as F

__all__ = ['KLLoss']


class KLLoss(nn.Module):
    def forward(self, x, y):
        return F.kl_loss(x, y)
