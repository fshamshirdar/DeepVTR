import time
import random
import numpy as np
from collections import deque
import torch
from vizdoom import *

from agent import Agent
import constants

class VizDoomAgent(Agent):
    def __init__(self, placeRecognition, navigation, wad, game_args=[], teachCommandsFile=None):
        super(VizDoomAgent, self).__init__(placeRecognition, navigation)
        self.game = self.initialize_game(wad, game_args)
        self.goal = None
        self.init = None
        self.teachCommandsFile = teachCommandsFile
        self.place_recognition.model.eval()
        self.navigation.model.eval()
        self.new_seed()

    def initialize_game(self, wad, game_args):
        game = DoomGame()
        game.load_config(constants.VIZDOOM_DEFAULT_CONFIG)
        for args in game_args:
            game.add_game_args(args)
        game.set_doom_scenario_path(wad)
        game.set_seed(self.new_seed())
        game.init()
        return game

    def new_seed(self):
        self.seed = random.randint(1, 1234567890)
        return self.seed

    def reset_map(self):
        self.new_seed()
        self.game.set_doom_map(constants.VIZDOOM_MAP_NAME_TEMPLATE % random.randint(constants.VIZDOOM_MIN_RANDOM_TEXTURE_MAP_INDEX,
                                                                                    constants.VIZDOOM_MAX_RANDOM_TEXTURE_MAP_INDEX))
        self.game.set_seed(self.seed)
        self.game.new_episode()
        state = self.game.get_state().screen_buffer.transpose([1, 2, 0])
        return state

    def reset_episode(self):
        self.game.set_seed(self.seed)
        self.game.new_episode()
        state = self.game.get_state().screen_buffer.transpose([1, 2, 0])
        return state

    def step(self, action, repeat=4):
        self.game.make_action(constants.VIZDOOM_ACTIONS_LIST[action], repeat)
        state = self.game.get_state().screen_buffer.transpose([1, 2, 0])
        return state

    def random_walk(self):
        state = self.reset_map()
        self.init = state
        print ("state: ", self.game.get_state().game_variables)
        for i in range(constants.AIRSIM_AGENT_TEACH_LEN):
            action = random.randint(0, constants.LOCO_NUM_CLASSES-1)
            next_state = self.step(action)
            print ("state: ", self.game.get_state().game_variables)
            print ("random walk: index %d action %d" % (i, action))
            rep, _ = self.sptm.append_keyframe(state, action, done)
            self.goal = state
            state = next_state
            if done:
                break

    def commanded_walk(self):
        action_file = open(self.teachCommandsFile)
        if action_file == None:
            return None

        state = self.reset_map()
        print ("state: ", self.game.get_state().game_variables)
        self.init = state
        i = 0
        actions = [int(val) for val in action_file.read().split('\n') if val.isdigit()]
        for action in actions:
            next_state = self.step(action)
            print ("state: ", self.game.get_state().game_variables)
            print ("commanded walk: index %d action %d" % (i, action))
            rep, _ = self.sptm.append_keyframe(state, action, False)
            self.goal = state
            state = next_state
            i = i+1
            time.sleep(0.1)

    def teach(self):
        if (self.teachCommandsFile == None):
            self.random_walk()
        else:
            self.commanded_walk()
        
    def repeat(self):
        self.sptm.build_graph()
        goal, goal_index, similarity = self.sptm.find_closest(self.goal)
        if (goal_index < 0):
            print ("cannot find goal")
            return

        current_state = self.reset_episode()
        print ("state: ", self.game.get_state().game_variables)
        previous_state = current_state
        previous_action = -1
        self.sptm.clear_sequence()
        while (True):
            matched_index, similarity_score, best_velocity = self.sptm.relocalize(current_state)
            path = self.sptm.find_shortest_path(matched_index, goal_index)
            print (matched_index, similarity_score, path)
            if (len(path) < 2): # achieved the goal
                break


            action, future_state = self.navigate(current_state, path, previous_action)

            from PIL import Image
            current_image = Image.fromarray(current_state)
            future_image = Image.fromarray(future_state)
            current_image.save("current.png", "PNG")
            future_image.save("future.png", "PNG")
            next_state = self.step(action)
            print ("state: ", self.game.get_state().game_variables)
            previous_state = current_state
            current_state = next_state
            previous_action = action
            time.sleep(0.2)

    def run(self):
        # init_position, init_orientation = [10, 0, -6], [0, 0, 0]
        # self.env.set_initial_pose(init_position, init_orientation)
        # self.env.set_mode(constants.AIRSIM_MODE_TEACH)
        time.sleep(1)
        print ("Running teaching phase")
        self.teach()

        # print ("Running repeating backward phase")
        # self.env.set_mode(constants.AIRSIM_MODE_REPEAT)
        # time.sleep(1)
        # self.repeat_backward()

        # init_position, init_orientation = [10, 2, -6], [0, 0, 0]
        # self.env.set_initial_pose(init_position, init_orientation)
        # self.env.set_mode(constants.AIRSIM_MODE_REPEAT)
        time.sleep(1)
        print ("Running repeating phase")
        self.repeat()

        # init_position, init_orientation = [10, 4, -6], [0, 0, 0]
        # self.env.set_initial_pose(init_position, init_orientation)
        # self.env.set_mode(constants.AIRSIM_MODE_REPEAT)
        time.sleep(1)
        print ("Running repeating phase")
        self.repeat()
