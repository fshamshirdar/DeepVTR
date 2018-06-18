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

    def path_lookahead(self, previous_state, current_state, path):
        selected_action, selected_prob, selected_future_state, selected_index = None, None, None, 0
        i = 1
        for i in range(1, len(path)):
            future_state = self.sptm.memory[path[i]].state
            actions = self.navigation.forward(previous_state, current_state, future_state)
            prob, pred = torch.max(actions.data, 1)
            prob = prob.data.cpu().item()
            action = pred.data.cpu().item()
            print (action, prob)

            if selected_action == None:
                selected_action, selected_prob, selected_future_state, selected_index = action, prob, future_state, i
            if (prob < constants.ACTION_LOOKAHEAD_PROB_THRESHOLD):
                break

            if (action == 1 or action == 2):
                selected_action, selected_prob, selected_future_state, selected_index = action, prob, future_state, i

        if (selected_index >= 3):
            for i in range(path[0], path[selected_index-3]):
                self.sptm.add_shortcut(i, path[selected_index], selected_prob)
        return selected_action, selected_prob, selected_future_state
