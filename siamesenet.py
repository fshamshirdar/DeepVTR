import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNet(nn.Module):
    def __init__(self, embeddingnet):
        super(SiameseNet, self).__init__()
        self.embeddingnet = embeddingnet

        self.fc1 = nn.Linear(in_features=9216, out_features=4096)
        self.relu1 = nn.Linear(inplace=True)
        self.drop1 = nn.Dropout(p=0.5, inplace=True)
        self.fc2 = nn.Linear(in_features=4096, out_features=1024)
        self.relu2 = nn.Linear(inplace=True)
        self.drop2 = nn.Dropout(p=0.5, inplace=True)
        self.fc3 = nn.Linear(in_features=1024, out_features=2)
        
    def forward(self, x1, x2):
        embedded_x1 = self.embeddingnet(x1)
        embedded_x2 = self.embeddingnet(x2)

        x = self.fc1(torch.cat([embedded_x1, embedded_x2], dim=1)) # dim should be 1, why doesn't work?
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        return x
