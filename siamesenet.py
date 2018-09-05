import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNet(nn.Module):
    def __init__(self, embeddingnet):
        super(SiameseNet, self).__init__()
        self.embeddingnet = embeddingnet

        self.bn0 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(in_features=1024, out_features=1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu1 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(in_features=1024, out_features=1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.relu3 = nn.ReLU(inplace=True)

        self.fc4 = nn.Linear(in_features=1024, out_features=2)
        
    def forward(self, x1, x2, embedding_required=True):
        if (embedding_required):
            embedded_x1 = self.embeddingnet(x1)
            embedded_x2 = self.embeddingnet(x2)
        else:
            embedded_x1 = x1
            embedded_x2 = x2

        x = torch.cat([embedded_x1, embedded_x2], dim=1) # dim should be 1, why doesn't work?
        x = self.bn0(x)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.fc4(x)
        return x
