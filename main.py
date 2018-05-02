import argparse
import io
import os
from PIL import Image
import numpy as np
import time
import math
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torch.autograd import Variable

from place_recognition import PlaceRecognition
from sptm import SPTM

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--datapath', default='datapath', type=str, help='path st_lucia dataset')
    parser.add_argument('--env', default='Pendulum-v0', type=str, help='open-ai gym environment')
    parser.add_argument('--hidden1', default=400, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=10000, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma') 
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu') 
    parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--trajectory_length', default=5, type=int, help='')
    parser.add_argument('--validate_steps', default=2000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='') 
    parser.add_argument('--train_iter', default=20000000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--checkpoint', default="checkpoints", type=str, help='Checkpoint path')

    args = parser.parse_args()


    normalize = transforms.Normalize(
        #mean=[121.50361069 / 127., 122.37611083 / 127., 121.25987563 / 127.],
        mean=[127. / 255., 127. / 255., 127. / 255.],
        std=[1 / 255., 1 / 255., 1 / 255.]
    )

    preprocess = transforms.Compose([
        transforms.Resize(227),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        normalize
    ])

    goal_variable = None
    source_variable = None
    source_picked = False
    goal_index = 0

    placeRecognition = PlaceRecognition()
    if args.checkpoint is not None:
        placeRecognition.load_weights(args.checkpoint)

    if torch.cuda.is_available():
        placeRecognition.cuda()

    sptm = SPTM(placeRecognition)
    with open(os.path.join(args.datapath, "index.txt"), 'r') as reader:
        for index in reader:
            index = index.strip()
            with open(os.path.join(args.datapath, index, "index.txt"), 'r') as image_reader:
                for image_path in image_reader:
                    print (image_path)
                    image_path = image_path.strip()
                    image = Image.open(os.path.join(args.datapath, index, image_path)).convert('RGB')
                    image_tensor = preprocess(image)

                    if source_picked == False:
                        source_image = Image.open(os.path.join(args.datapath, index, image_path)).convert('RGB')
                        source_tensor = preprocess(source_image)
                        source_tensor.unsqueeze_(0)
                        source_variable = Variable(source_tensor).cuda()
                        source_picked = True

                    if goal_index == 50:
                        goal_image = Image.open(os.path.join(args.datapath, index, image_path)).convert('RGB')
                        goal_tensor = preprocess(goal_image)
                        goal_tensor.unsqueeze_(0)
                        goal_variable = Variable(goal_tensor).cuda()
                    goal_index = goal_index + 1

#                    plt.figure()
#                    plt.imshow(image_tensor.cpu().numpy().transpose((1, 2, 0)))
#                    plt.show()

                    image_tensor.unsqueeze_(0)
                    image_variable = Variable(image_tensor).cuda()
                    sptm.append_keyframe(image_variable)

    sptm.build_graph()
    goal, goal_index = sptm.find_closest(goal_variable)
    source, source_index = sptm.find_closest(source_variable)
    if (source != None and goal != None):
        print (source_index, goal_index)
        path = sptm.find_shortest_path(source_index, goal_index)
        print (path)

    # sptm.relocalize([ goal_variable ])
