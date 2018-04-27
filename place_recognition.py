import torch
import torch.nn as nn
import torch.nn.functional as F
from placenet import PlaceNet

class PlaceRecognition:
    def __init__(self):
        self.placenet = PlaceNet()

    def load_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.placenet.load_state_dict(checkpoint['state_dict'])

    def cuda(self):
        self.placenet.cuda()

    def forward(self, input):
        return self.placenet(input) # get representation

    def compute_similarity_score(self, rep1, rep2):
        similarity_score = F.cosine_similarity(rep1, rep2)
        return similarity_score
