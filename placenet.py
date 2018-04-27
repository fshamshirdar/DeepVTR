import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from collections import OrderedDict
from torch.legacy.nn import SpatialCrossMapLRN as SpatialCrossMapLRNOld

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta


    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x

class PlaceNet(nn.Module):
    def __init__(self):
        super(PlaceNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1))
        self.norm1 = LRN(local_size=5, alpha=0.000100, beta=0.750000)
        self.conv2 = nn.Conv2d (96, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1))
        self.norm2 = LRN(local_size=5, alpha=0.000100, beta=0.750000)
        self.conv3 = nn.Conv2d (256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d (384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d (384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2)
        self.relu6 = nn.ReLU(inplace=True)
        self.pool6 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1))

        # unused
        self.fc7_new = nn.Linear(in_features=9216, out_features=4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout(p=0.5, inplace=True)
        self.fc8_new = nn.Linear(in_features=4096, out_features=2543)
        self.prob = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.pool6(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        return x
