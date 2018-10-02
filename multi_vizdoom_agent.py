import time
import random
import math
import pickle
import numpy as np
# from multiprocessing import Process
from PIL import Image
import cv2
from collections import deque
import torch
from vizdoom import *

from multi_agent import MultiAgent
import constants

def wait():
    input('key')

class VizDoomSingleAgent:
    def __init__(self, placeRecognition, trail, navigation, wad, seed=0, game_args=[]):
        self.placeRecognition = placeRecognition
        self.trail = trail
        self.navigation = navigation
        self.agent_state = constants.MULTI_AGENT_STATE_SEARCH
        self.wad = wad
        self.seed = seed
        self.game_args = game_args
        self.temporary_trail = []
        self.goal = None
        self.init = None
        self.cycle = 0
        self.episode_num = 0
        self.successful_episode_num = 0
        self.current_state = None
        self.previous_action = 0
        self.last_matched = []
        self.random_ongoing_actions = []
        self.random_movement_chance = constants.MULTI_AGENT_INITIAL_RANDOM_MOVEMENT_CHANCE
        self.game = self.initialize_game()
        self.placeRecognition.model.eval()
        # self.placeRecognition.siamesenet.eval()
        self.navigation.model.eval()

    def initialize_game(self):
        game = DoomGame()
        game.load_config(constants.VIZDOOM_DEFAULT_CONFIG)
        for args in self.game_args:
            game.add_game_args(args)
        game.set_doom_scenario_path(self.wad)
        game.set_seed(self.seed)
        game.init()
        return game

    def replay_episode(self, lmp):
        if (lmp == None):
            return

        self.reset_episode()
        actions = []
        with open(lmp, 'rb') as f:
            actions = pickle.load(f)

        print ("loading actions: ", len(actions))
        for action in actions:
            current_frame = self.game.get_state().screen_buffer.transpose([1, 2, 0])
            pose = self.game.get_state().game_variables
            next_state = self.step(action)
            self.temporary_trail.append({'state': current_frame, 'position': pose, 'action': action})

        self.trail.append_waypoints(self.temporary_trail, self.cycle)
        self.successful_episode_num += 1
        print ("memory size: ", self.trail.len())

    def get_distance_to_obstacles(self, depth_buf):
        min_left = 255
        min_right = 255
        i = int(depth_buf.shape[0]/2)
        for j in range(int(depth_buf.shape[1]/2)):
            if (depth_buf[i, j] < min_left):
                min_left = depth_buf[i, 0]
        
        for j in range(int(depth_buf.shape[1]/2), depth_buf.shape[1]):
            if (depth_buf[i, j] < min_right):
                min_right = depth_buf[i, j]

        return min_left, min_right

    def set_seed(self, seed):
        self.seed = seed

    def set_map(self, selected_map):
        self.game.set_doom_map(selected_map)

    def reset_episode(self, recording_path=None):
        self.game.set_seed(self.seed)
        # self.game.new_episode('./saved.lmp')
        if (recording_path is None):
            self.game.new_episode()
        else:
            self.game.new_episode(recording_path)
        state = self.game.get_state().screen_buffer.transpose([1, 2, 0])
        pose = self.game.get_state().game_variables
        self.init = state
        self.previous_action = -1
        self.current_state = state
        self.temporary_trail = [{'state': state, 'position': pose, 'action': constants.ACTION_MOVE_FORWARD}]
        self.last_matched = []
        self.episode_num += 1
        return state

    def step(self, action, repeat=1):
        self.game.make_action(constants.VIZDOOM_ACTIONS_LIST[action], repeat)
        state = self.game.get_state().screen_buffer.transpose([1, 2, 0])
        self.game.make_action(constants.VIZDOOM_STAY_IDLE, 4)
        # time.sleep(0.1)
        return state

    def goal_reached(self, pose):
        # goal_location = (609.92807007, 247.60987854, 0)
        # goal_location = (-97.92807007, -220.60987854, 0)
        goal_location = (715.92807007, -408.60987854, 0)
        distance = math.sqrt((pose[0] - goal_location[0]) ** 2 +
                             (pose[1] - goal_location[1]) ** 2 + 
                             (pose[2] - goal_location[2]) ** 2)

        return (distance < constants.VIZDOOM_GOAL_DISTANCE_THRESHOLD)

    def global_random_search(self):
        repeat = 1
        permitted_actions = [i for i in range(0, constants.LOCO_NUM_CLASSES)]
        # permitted_actions.remove(constants.ACTION_MOVE_BACKWARD)
        permitted_actions.remove(constants.ACTION_MOVE_FORWARD)
        # permitted_actions.remove(constants.ACTION_MOVE_LEFT)
        # permitted_actions.remove(constants.ACTION_MOVE_RIGHT)
        action = random.choice(permitted_actions)

        ongoing_random_action = [action for i in range(random.randint(1, 5))]
        ongoing_forward_action = [constants.ACTION_MOVE_FORWARD for i in range(random.randint(3, 7))]
        self.random_ongoing_actions += ongoing_random_action
        self.random_ongoing_actions += ongoing_forward_action
        
        return action, repeat

    def local_random_search(self):
        repeat = 1
        permitted_actions = [i for i in range(0, constants.LOCO_NUM_CLASSES)]
        # permitted_actions.remove(constants.ACTION_MOVE_BACKWARD)
        permitted_actions.remove(constants.ACTION_MOVE_FORWARD)
        # permitted_actions.remove(constants.ACTION_MOVE_LEFT)
        # permitted_actions.remove(constants.ACTION_MOVE_RIGHT)
        action = random.choice(permitted_actions)

        ongoing_random_action = [action for i in range(random.randint(1, 5))]
        ongoing_forward_action = [constants.ACTION_MOVE_FORWARD for i in range(random.randint(1, 7))]
        self.random_ongoing_actions += ongoing_random_action
        self.random_ongoing_actions += ongoing_forward_action
        
        return action, repeat

    def trail_following(self, future_state):
        cv2.imshow('image', future_state)
        cv2.waitKey(10)

        repeat = 1
        actions = self.navigation.forward(self.current_state, None, future_state)
        actions = torch.squeeze(actions)
        sorted_actions, indices = torch.sort(actions, descending=True)
        action = indices[0]
        if ((self.previous_action == constants.ACTION_MOVE_FORWARD and action == constants.ACTION_MOVE_BACKWARD) or
            (self.previous_action == constants.ACTION_MOVE_BACKWARD and action == constants.ACTION_MOVE_FORWARD) or
            (self.previous_action == constants.ACTION_TURN_RIGHT and action == constants.ACTION_TURN_LEFT) or
            (self.previous_action == constants.ACTION_TURN_LEFT and action == constants.ACTION_TURN_RIGHT) or
            (self.previous_action == constants.ACTION_MOVE_RIGHT and action == constants.ACTION_MOVE_LEFT) or
            (self.previous_action == constants.ACTION_MOVE_LEFT and action == constants.ACTION_MOVE_RIGHT)):
            action = indices[1]

        # print ("actions: ", actions)
        # wait()

        return action, repeat

    def search(self):
        repeat = 1
        pose = self.game.get_state().game_variables
        depth_buf = self.game.get_state().depth_buffer
        left_dist, right_dist = self.get_distance_to_obstacles(depth_buf)
        stabilization_episode = (self.successful_episode_num % constants.MULTI_AGENT_PATH_STABILIZATION_RATE == 0)
        if (stabilization_episode):
            print ("stabilization episde")

        if (len(self.random_ongoing_actions) > 0):
            # print (self.random_ongoing_actions)
            self.last_matched = []
            action = self.random_ongoing_actions.pop(0)
            repeat = 1
            # wait()
        else:
            # if (self.cycle % 5 == 0):
            #     self.last_matched = []

            self.last_matched = []
            future_state, score, velocity, self.last_matched = self.trail.find_closest_waypoint(self.current_state, backward=False, last_matched=self.last_matched)
            # future_state, score, velocity, self.last_matched = self.trail.find_best_waypoint(self.current_state, backward=False, last_matched=self.last_matched)
            # future_state, score, velocity, self.last_matched = self.trail.find_most_similar_waypoint(self.current_state, backward=False, last_matched=self.last_matched)

            if (future_state is None):
                action, repeat = self.global_random_search()
            elif (stabilization_episode == False and self.successful_episode_num % constants.MULTI_AGENT_PATH_STABILIZATION_RATE > 0 and random.random() < self.random_movement_chance):
                action, repeat = self.local_random_search()
            else:
                action, repeat = self.trail_following(future_state)
                print ("matched: ", score, action)
                # wait()

        if (left_dist < 3):
            action = constants.ACTION_TURN_RIGHT
            repeat = 1
        elif (right_dist < 3):
            action = constants.ACTION_TURN_LEFT
            repeat = 1

        next_state = self.step(action, repeat=repeat)

        self.previous_action = action
        self.current_state = next_state
        self.temporary_trail.append({'state': next_state, 'position': pose, 'action': action})

    def update(self, cycle):
        self.cycle = cycle
        if (self.agent_state == constants.MULTI_AGENT_STATE_SEARCH):
            if (len(self.temporary_trail) > 700): # TODO: temporary
                print ("got too long, resetting")
                self.reset_episode()

            self.search()
            pose = self.game.get_state().game_variables
            if (self.goal_reached(pose)):
                self.random_movement_chance -= constants.MULTI_AGENT_RANDOM_MOVEMENT_CHANCE_GAMMA
                self.successful_episode_num += 1
                print ("random chance: ", self.random_movement_chance)
                print ("memory size: ", self.trail.len())
                # self.agent_state = constants.MULTI_AGENT_STATE_HOME
                print ("goal reached, trail len: ", len(self.temporary_trail))
                self.trail.update_waypoints()
                self.trail.append_waypoints(self.temporary_trail, self.cycle)
                # self.game.close()
                self.reset_episode()
                return True
        return False

    def record(self, recording_path):
        self.reset_episode()
        actions = []
        while (True):
            if (len(self.temporary_trail) > 700): # TODO: temporary
                print ("got too long, resetting")
                self.reset_episode()
                actions = []

            self.search()
            actions.append(self.previous_action)
            pose = self.game.get_state().game_variables
            if (self.goal_reached(pose)):
                print ("goal reached, trail len: ", len(self.temporary_trail))
                self.game.close()
                with open(recording_path+'_actions.txt', 'wb') as f:
                    pickle.dump(actions, f)
                return

class MultiVizDoomAgent(MultiAgent):
    def __init__(self, placeRecognition, navigation, wad, lmp):
        super(MultiVizDoomAgent, self).__init__(placeRecognition, navigation)
        self.placeRecognition = placeRecognition
        self.navigation = navigation
        self.wad = wad
        self.lmp = lmp
        self.seed = self.new_seed()
        self.cycle = 0
        self.agents = []
        self.placeRecognition.model.eval()
        self.navigation.model.eval()
        for i in range(constants.MULTI_NUM_AGENTS):
            self.agents.append(VizDoomSingleAgent(self.placeRecognition, self.trail, self.navigation, self.wad, self.seed, game_args=[]))

    def new_seed(self):
        self.seed = 100 # random.randint(1, 1234567890)
        return self.seed

    def record(self):
        if (self.lmp is None):
            print ("lmp is empty")
            return
        selected_map = 'map02'
        self.agents[0].set_map(selected_map)
        self.agents[0].record(self.lmp)

    def run(self):
        # selected_map = (constants.VIZDOOM_MAP_NAME_TEMPLATE % random.randint(constants.VIZDOOM_MIN_RANDOM_TEXTURE_MAP_INDEX, constants.VIZDOOM_MAX_RANDOM_TEXTURE_MAP_INDEX))
        # selected_map = (constants.VIZDOOM_MAP_NAME_TEMPLATE % random.randint(constants.VIZDOOM_MIN_TEST_MAP_INDEX, constants.VIZDOOM_MAX_TEST_MAP_INDEX))
        selected_map = 'map02'
        for i in range(constants.MULTI_NUM_AGENTS):
            self.agents[i].set_map(selected_map)
            if (i == 0): # only once
                self.agents[i].replay_episode(self.lmp)
            self.agents[i].reset_episode()

        self.trail.draw_waypoints()
        while (True):
            self.cycle += 1
            for i in range(constants.MULTI_NUM_AGENTS):
                reached_goal = self.agents[i].update(self.cycle)
                if (reached_goal):
                    self.trail.draw_waypoints()
                    # self.agents[i].random_movement_chance = float(input("random: "))
            # self.trail.update_waypoints()
