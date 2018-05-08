import torch
import torch.nn as nn
import torch.nn.functional as F
from loconet import LocoNet

class Navigation:
    def __init__(self):
        self.loconet = LocoNet()

    def load_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.loconet.load_state_dict(checkpoint['state_dict'])

    def cuda(self):
        self.loconet.cuda()

    def forward(self, input):
        return self.loconet(input) # get representation
