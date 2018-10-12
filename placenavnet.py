import torch
import torch.nn as nn

class PlaceNavNet(nn.Module):
    def __init__(self, num_classes=6):
        super(PlaceNavNet, self).__init__()
#         self.conv7 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=0)
#         self.relu7 = nn.ReLU(inplace=True)
#         self.pool7 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1))

        self.conv7 = nn.Conv2d(512, 512, kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.relu7 = nn.ReLU()
        self.fc8 = nn.Linear(in_features=512*3*3, out_features=1024)
        self.relu8 = nn.ReLU()
        self.drop8 = nn.Dropout(p=0.5)
        self.fc9 = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        x = self.conv7(x)
        x = self.relu7(x)
        x = x.view(x.size(0), 512 * 3 * 3)
        x = self.fc8(x)
        x = self.relu8(x)
        x = self.drop8(x)
        x = self.fc9(x)
        return x


