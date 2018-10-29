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
    def __init__(self, placeRecognition, navigation):
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

    def navigate(self, current_state, path, previous_action):
        if (constants.ACTION_LOOKAHEAD_ENABLED):
            actions, future_state = self.path_lookahead(current_state, path)
            actions = torch.squeeze(actions)

            action = np.random.choice(np.arange(0, constants.LOCO_NUM_CLASSES), p=actions.data.cpu().numpy())
            prob = actions[action].data.cpu().item()

            # sorted_actions, indices = torch.sort(actions, descending=True)
            # action = indices[0]
            # if ((previous_action == constants.ACTION_MOVE_FORWARD and action == constants.ACTION_MOVE_BACKWARD) or
            #     (previous_action == constants.ACTION_MOVE_BACKWARD and action == constants.ACTION_MOVE_FORWARD) or
            #     (previous_action == constants.ACTION_TURN_RIGHT and action == constants.ACTION_TURN_LEFT) or
            #     (previous_action == constants.ACTION_TURN_LEFT and action == constants.ACTION_TURN_RIGHT) or
            #     (previous_action == constants.ACTION_MOVE_RIGHT and action == constants.ACTION_MOVE_LEFT) or
            #     (previous_action == constants.ACTION_MOVE_LEFT and action == constants.ACTION_MOVE_RIGHT)):
            #     action = indices[1]
        else:
            future_state = self.sptm.memory[path[1]].state
            actions = self.navigation.forward(current_state, None, future_state)
            actions = torch.squeeze(actions)
            sorted_actions, indices = torch.sort(actions, descending=True)
            action = indices[0]
            if ((previous_action == constants.ACTION_MOVE_FORWARD and action == constants.ACTION_MOVE_BACKWARD) or
                (previous_action == constants.ACTION_MOVE_BACKWARD and action == constants.ACTION_MOVE_FORWARD) or
                (previous_action == constants.ACTION_TURN_RIGHT and action == constants.ACTION_TURN_LEFT) or
                (previous_action == constants.ACTION_TURN_LEFT and action == constants.ACTION_TURN_RIGHT) or
                (previous_action == constants.ACTION_MOVE_RIGHT and action == constants.ACTION_MOVE_LEFT) or
                (previous_action == constants.ACTION_MOVE_LEFT and action == constants.ACTION_MOVE_RIGHT)):
                action = indices[1]

            # prob, pred = torch.max(actions.data, 0)
            # prob = prob.data.cpu().item()
            # action = pred.data.cpu().item()
            print ("actions: ", actions, action)

            # select based on probability distribution
            # action = np.random.choice(np.arange(0, constants.LOCO_NUM_CLASSES), p=actions.data.cpu().numpy())
            # prob = actions[action].data.cpu().item()
        return action, future_state

    def path_lookahead(self, current_state, path):
        selected_action, selected_prob, selected_future_state, selected_index = None, None, None, 0
        i = 1
        # for i in range(1, len(path)):
        lookahead_len = min(constants.ACTION_LOOKAHEAD_LEN, len(path))
        for i in range(1, lookahead_len):
            future_state = self.sptm.memory[path[i]].state
            # actions = self.navigation.forward(previous_state, current_state, future_state)
            actions = self.navigation.forward(current_state, self.sptm.memory[path[0]].state, future_state)
            prob, pred = torch.max(actions.data, 1)
            prob = prob.data.cpu().item()
            action = pred.data.cpu().item()
            print (action, prob)

            if (selected_action == None or prob > constants.ACTION_LOOKAHEAD_PROB_THRESHOLD):
                selected_action, selected_prob, selected_future_state, selected_index, selected_actions = action, prob, future_state, i, actions
            else:
                break # stop from proceeding

            # if selected_action == None:
            #     selected_action, selected_prob, selected_future_state, selected_index = action, prob, future_state, i
            # if (prob < constants.ACTION_LOOKAHEAD_PROB_THRESHOLD):
            #     break
            # if (action == 1 or action == 2):
            #    selected_action, selected_prob, selected_future_state, selected_index = action, prob, future_state, i

        # if (selected_index >= 3):
        #     for i in range(path[0], path[selected_index-3]):
        #         self.sptm.add_shortcut(i, path[selected_index], selected_prob)
        # return selected_action, selected_prob, selected_future_state, selected_actions

        return selected_actions, selected_future_state
