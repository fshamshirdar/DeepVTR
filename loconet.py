import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from collections import OrderedDict
from torch.legacy.nn import SpatialCrossMapLRN as SpatialCrossMapLRNOld

# TODO

class LocoNet(nn.Module):
    def __init__(self):
        super(LocoNet, self).__init__()
        self.prob = nn.Softmax()

    def forward(self, x):
        return x
