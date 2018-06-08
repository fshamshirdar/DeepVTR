import argparse
import io
import os
from PIL import Image
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import torch

from data_collector import DataCollector
from agent import Agent
from airsim_agent import AirSimAgent
from place_recognition import PlaceRecognition
from navigation import Navigation
import constants

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--mode', default='train', type=str, help='support option: airsim_collect/train_place/train_nav/eval_place/eval_nav/airsim_agent/bebop_agent')
    parser.add_argument('--datapath', default='dataset', type=str, help='path to dataset')
    parser.add_argument('--env', default='Pendulum-v0', type=str, help='open-ai gym environment')
    parser.add_argument('--collect_index', default=0, type=int, help='collect intial index')
    parser.add_argument('--checkpoint_path', type=str, help='Checkpoint path')
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
    parser.add_argument('--place_checkpoint', type=str, help='Place Checkpoint path')
    parser.add_argument('--navigation_checkpoint', type=str, help='Navigation Checkpoint path')
    parser.add_argument('--teach_dump', type=str, help='Teach dump commands file')

    args = parser.parse_args()

    placeRecognition = PlaceRecognition()
    navigation = Navigation()
    if args.place_checkpoint is not None:
        placeRecognition.load_weights(args.place_checkpoint)
    if args.navigation_checkpoint is not None:
        navigation.load_weights(args.navigation_checkpoint)

#    agent = Agent(placeRecognition, navigation)

    if torch.cuda.is_available():
        placeRecognition.cuda()
        navigation.cuda()

    if args.mode == 'airsim_collect':
        dataCollector = DataCollector(args.datapath)
        dataCollector.collect(args.collect_index)
    elif args.mode == 'train_place':
        placeRecognition.train(args.datapath, args.checkpoint_path, args.train_iter)
    elif args.mode == 'eval_place':
        placeRecognition.eval(args.datapath)
    elif args.mode == 'train_nav':
        navigation.train(args.datapath, args.checkpoint_path, args.train_iter)
    elif args.mode == 'eval_nav':
        navigation.eval(args.datapath)
    elif args.mode == 'airsim_agent':
        airSimAgent = AirSimAgent(placeRecognition, navigation, teachCommandsFile=args.teach_dump)
        airSimAgent.run()
    elif args.mode == 'bebop_agent':
        bebopAgent = BebopAgent(placeRecognition, navigation, teachCommandsFile=args.teach_dump)
        bebopAgent.run()
    else:
        pass
