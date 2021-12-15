import torch, torch.nn as nn, numpy as np, random

from .front_resnet import ResNet18, ResNet34, ResNet34SE
from .pooling import StatsPool

class ResNet34StatsPool(nn.Module):
    def __init__(self, in_planes, embedding_size, dropout=0.5, **kwargs):
        super(ResNet34StatsPool, self).__init__()
        self.front = ResNet34(in_planes, **kwargs)
        self.pool = StatsPool()
        self.bottleneck = nn.Linear(in_planes*8*2, embedding_size)
        self.drop = nn.Dropout(dropout) if dropout else None
        
    def forward(self, x):
        x = self.front(x.unsqueeze(dim=1))
        x = self.pool(x)
        x = self.bottleneck(x)
        if self.drop:
            x = self.drop(x)
        return x
    