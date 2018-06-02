import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.autograd import Variable

from sptm import SPTM
from place_recognition import PlaceRecognition
from navigation import Navigation
import constants

import os
from PIL import Image
import numpy as np
from collections import deque

class Agent:
    def __init__(self, placeRecognition=None, navigation=None):
        if placeRecognition == None:
            placeRecognition = PlaceRecognition()
        if navigation == None:
            navigation = PlaceRecognition()
        self.place_recognition = placeRecognition
        self.sptm = SPTM(self.place_recognition)
        self.navigation = navigation

    def load_place_weights(self, place_checkpoint_path):
        self.place_recognition.load_weights(place_checkpoint_path)

    def load_navigation_weights(self, navigation_checkpoint_path):
        self.navigation.load_weights(navigation_checkpoint_path)

    def cuda(self):
        self.place_recognition.cuda()
        self.navigation.cuda()
