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
    def __init__(self):
        self.placeRecognition = PlaceRecognition()
        self.sptm = SPTM(self.placeRecognition)
        self.navigation = Navigation()

        self.normalize = transforms.Normalize(
            # mean=[121.50361069 / 127., 122.37611083 / 127., 121.25987563 / 127.],
            mean=[127. / 255., 127. / 255., 127. / 255.],
            std=[1 / 255., 1 / 255., 1 / 255.]
        )

        self.preprocess = transforms.Compose([
            transforms.Resize(227),
            transforms.CenterCrop(227),
            transforms.ToTensor(),
            self.normalize
        ])


    def load_place_weights(self, place_checkpoint_path):
        self.placeRecognition.load_weights(place_checkpoint_path)

    def load_navigation_weights(self, navigation_checkpoint_path):
        self.navigation.load_weights(navigation_checkpoint_path)

    def cuda(self):
        self.placeRecognition.cuda()
        self.navigation.cuda()

    def dry_run_test(self, args):
        goal_variable = None
        source_variable = None
        source_picked = False
        goal_index = 0

        with open(os.path.join(args.datapath, "teach.txt"), 'r') as reader:
            for image_path in reader:
                print (image_path)
                image_path = image_path.strip()
                image = Image.open(os.path.join(args.datapath, image_path)).convert('RGB')
                image_tensor = self.preprocess(image)

#               plt.figure()
#               plt.imshow(image_tensor.cpu().numpy().transpose((1, 2, 0)))
#               plt.show()

                image_tensor.unsqueeze_(0)
                image_variable = Variable(image_tensor).cuda()
                self.sptm.append_keyframe(image_variable)

        with open(os.path.join(args.datapath, "repeat.txt"), 'r') as reader:
            sequence = deque(maxlen=constants.SEQUENCE_LENGTH)
            for image_path in reader:
                print (image_path)
                image_path = image_path.strip()
                image = Image.open(os.path.join(args.datapath, image_path)).convert('RGB')
                image_tensor = self.preprocess(image)

#               plt.figure()
#               plt.imshow(image_tensor.cpu().numpy().transpose((1, 2, 0)))
#               plt.show()

                image_tensor.unsqueeze_(0)
                image_variable = Variable(image_tensor).cuda()
                sequence.append(image_variable)

                if (len(sequence) == constants.SEQUENCE_LENGTH):
                    self.sptm.relocalize(sequence)

#        self.sptm.build_graph()
#        goal, goal_index = self.sptm.find_closest(goal_variable)
#        source, source_index = self.sptm.find_closest(source_variable)
#        if (source != None and goal != None):
#            print (source_index, goal_index)
#            path = self.sptm.find_shortest_path(source_index, goal_index)
#            print (path)
