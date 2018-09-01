import argparse
import io
import os
from PIL import Image
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import torch

from airsim_data_collector import AirSimDataCollector
from vizdoom_data_collector import VizDoomDataCollector
from agent import Agent
from dqn_agent import DQNAgent
from dqn_agent_single import DQNAgentSingle
from airsim_agent import AirSimAgent
from vizdoom_agent import VizDoomAgent
from multi_vizdoom_agent import MultiVizDoomAgent
from multi_airsim_agent import MultiAirSimAgent
# from bebop_agent import BebopAgent
# from pioneer_agent import PioneerAgent
from place_recognition import PlaceRecognition
from navigation import Navigation
# from placenav import PlaceNavigation
import constants

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--mode', default='train', type=str, help='support option: airsim_collect/vizdoom_collect/train_place/train_nav/eval_place/eval_nav/test_nav/train_placenav/eval_placenav/dqn_agent/dqn_agent_single/airsim_agent/bebop_agent/pioneer_agent/vizdoom_agent/multi_vizdoom_agent/multi_airsim_agent')
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
    parser.add_argument('--placenav_checkpoint', type=str, help='PlaceNav Checkpoint path')
    parser.add_argument('--teach_dump', type=str, help='Teach dump commands file')
    parser.add_argument('--wad', type=str, default='vizdoom/Train/D3_battle_navigation_split.wad_manymaps_test.wad', help='WAD path')
    parser.add_argument('--dump_memory_path', type=str, help='Dump memory path')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    if args.mode == 'airsim_collect':
        dataCollector = AirSimDataCollector(args.datapath)
        dataCollector.collect(args.collect_index)
    if args.mode == 'vizdoom_collect':
        dataCollector = VizDoomDataCollector(args.datapath, args.wad)
        dataCollector.collect(args.collect_index)
    elif args.mode == 'train_place':
        placeRecognition = PlaceRecognition(args.place_checkpoint, use_cuda)
        placeRecognition.train(args.datapath, args.checkpoint_path, args.train_iter)
    elif args.mode == 'eval_place':
        placeRecognition = PlaceRecognition(args.place_checkpoint, use_cuda)
        placeRecognition.eval(args.datapath)
    elif args.mode == 'train_nav':
        navigation = Navigation(args.navigation_checkpoint, use_cuda)
        navigation.train(args.datapath, args.checkpoint_path, args.train_iter)
    elif args.mode == 'eval_nav':
        navigation = Navigation(args.navigation_checkpoint, use_cuda)
        navigation.eval(args.datapath)
    elif args.mode == 'test_nav':
        navigation = Navigation(args.navigation_checkpoint, use_cuda)
        navigation.test('current.png', 'future.png')
#    elif args.mode == 'train_placenav':
#        placeRecognition = PlaceRecognition(args.place_checkpoint, use_cuda)
#        placeNav = PlaceNavigation(placeRecognition, args.placenav_checkpoint, use_cuda)
#        placeNav.train(args.datapath, args.checkpoint_path, args.train_iter)
#    elif args.mode == 'eval_placenav':
#        placeRecognition = PlaceRecognition(args.place_checkpoint, use_cuda)
#        placeNav = PlaceNavigation(placeRecognition, args.placenav_checkpoint, use_cuda)
#        placeNav.train(args.datapath, args.checkpoint_path, args.train_iter)
    elif args.mode == 'dqn_agent':
        placeRecognition = PlaceRecognition(args.place_checkpoint, use_cuda)
        navigation = Navigation(args.navigation_checkpoint, use_cuda)
        dqnAgent = DQNAgent(placeRecognition, navigation, args.checkpoint_path, args.train_iter, args.teach_dump)
        dqnAgent.run()
    elif args.mode == 'dqn_agent_single':
        placeRecognition = PlaceRecognition(args.place_checkpoint, use_cuda)
        navigation = Navigation(args.navigation_checkpoint, use_cuda)
        dqnAgentSingle = DQNAgentSingle(placeRecognition, navigation, args.checkpoint_path, args.train_iter, args.dump_memory_path)
        dqnAgentSingle.run()
    elif args.mode == 'airsim_agent':
        placeRecognition = PlaceRecognition(args.place_checkpoint, use_cuda)
        navigation = Navigation(args.navigation_checkpoint, use_cuda)
        airSimAgent = AirSimAgent(placeRecognition, navigation, teachCommandsFile=args.teach_dump)
        airSimAgent.run()
    elif args.mode == 'bebop_agent':
        placeRecognition = PlaceRecognition(args.place_checkpoint, use_cuda)
        navigation = Navigation(args.navigation_checkpoint, use_cuda)
        bebopAgent = BebopAgent(placeRecognition, navigation, teachCommandsFile=args.teach_dump)
        bebopAgent.run()
    elif args.mode == 'pioneer_agent':
        placeRecognition = PlaceRecognition(args.place_checkpoint, use_cuda)
        navigation = Navigation(args.navigation_checkpoint, use_cuda)
        pioneerAgent = PioneerAgent(placeRecognition, navigation, teachCommandsFile=args.teach_dump)
        pioneerAgent.run()
    elif args.mode == 'vizdoom_agent':
        placeRecognition = PlaceRecognition(args.place_checkpoint, use_cuda)
        navigation = Navigation(args.navigation_checkpoint, use_cuda)
        vizDoomAgent = VizDoomAgent(placeRecognition, navigation, args.wad, game_args=[], teachCommandsFile=args.teach_dump)
        vizDoomAgent.run()
    elif args.mode == 'multi_vizdoom_agent':
        placeRecognition = PlaceRecognition(args.place_checkpoint, use_cuda)
        navigation = Navigation(args.navigation_checkpoint, use_cuda)
        multiVizDoomAgent = MultiVizDoomAgent(placeRecognition, navigation, args.wad)
        multiVizDoomAgent.run()
    elif args.mode == 'multi_airsim_agent':
        placeRecognition = PlaceRecognition(args.place_checkpoint, use_cuda)
        navigation = Navigation(args.navigation_checkpoint, use_cuda)
        multiAirSimAgent = MultiAirSimAgent(placeRecognition, navigation)
        multiAirSimAgent.run()
    else:
        pass
