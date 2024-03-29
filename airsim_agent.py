import time
import random
import numpy as np
from collections import deque
import pickle
import torch
import gym
import gym_airsim

from agent import Agent
import constants

class AirSimAgent(Agent):
    def __init__(self, placeRecognition=None, navigation=None, teachCommandsFile=None):
        super(AirSimAgent, self).__init__(placeRecognition, navigation)
        self.env = gym.make('AirSim-v1')
        self.env.reset()
        self.goal = None
        self.init = None
        self.teachCommandsFile = teachCommandsFile
        self.place_recognition.model.eval()
        self.navigation.model.eval()
        self.positions = []
        self.num_steps = 0

    def random_walk(self):
        state = self.env.reset()
        self.init = state
        for i in range(constants.AIRSIM_AGENT_TEACH_LEN):
            action = random.randint(0, constants.LOCO_NUM_CLASSES-1)
            next_state, _, done, position = self.env.step(action)
            print ("random walk: index %d action %d" % (i, action))
            rep, _ = self.sptm.append_keyframe(state, action, done, position)
            self.goal = state
            state = next_state
            if done:
                break

    def commanded_walk(self):
        action_file = open(self.teachCommandsFile)
        if action_file == None:
            return None

        state = self.env.reset()
        self.init = state
        i = 0
        actions = [int(val) for val in action_file.read().split('\n') if val.isdigit()]
        for action in actions:
            next_state, _, done, position = self.env.step(action)
            print ("commanded walk: index %d action %d" % (i, action))
            rep, _ = self.sptm.append_keyframe(state, action, done, position)
            self.goal = state
            state = next_state
            i = i+1
            if done:
                break

    def teach(self):
        # while (True):
        #     action = eval(input("Enter a number: "))
        #     next_state, _, done, position = self.env.step(action)
        if (self.teachCommandsFile == None):
            self.random_walk()
        else:
            self.commanded_walk()
        
    def repeat(self):
        self.sptm.build_graph(with_shortcuts=constants.SHORTCUT_ENABLE)
        goal, goal_index, similarity = self.sptm.find_closest(self.goal)
        if (goal_index < 0):
            print ("cannot find goal")
            return

        current_state = self.env.reset()
        previous_state = current_state
        previous_action = -1
        self.sptm.clear_sequence()
        self.num_steps = 0
        temporality_enabled = constants.TEMPORALITY_ENABLE
        while (True):
            matched_index, similarity_score, best_velocity = self.sptm.relocalize(current_state, temporality_enabled)
            path = self.sptm.find_shortest_path(matched_index, goal_index)
            print (matched_index, similarity_score, path)
            if (len(path) < 2): # achieved the goal
                break

            if (similarity_score > 0.7):
                action, future_state = self.navigate(current_state, path, previous_action)
                temporality_enabled = constants.TEMPORALITY_ENABLE
            else:
                action = 1
                future_state = current_state
                self.sptm.clear_sequence()
                temporality_enabled = False

            from PIL import Image
            current_image = Image.fromarray(current_state)
            future_image = Image.fromarray(future_state)
            current_image.save("current.png", "PNG")
            future_image.save("future.png", "PNG")
            next_state, _, done, position = self.env.step(action)
            previous_state = current_state
            current_state = next_state
            previous_action = action
            self.positions.append(position)
            self.num_steps += 1
            if (done or self.num_steps > 5000):
                break

    def run(self):
        init_position, init_orientation = [10, 0, -6], [0, 0, 0]
        self.env.set_initial_pose(init_position, init_orientation)

        if (self.sptm.load("experiment1.dump") == False):
            self.env.set_mode(constants.AIRSIM_MODE_TEACH)
            time.sleep(1)
            print ("Running teaching phase")
            self.teach()
            self.sptm.save("experiment1.dump")
        else:
            self.goal = self.sptm.memory[-1].state

        # print ("Running repeating backward phase")
        # self.env.set_mode(constants.AIRSIM_MODE_REPEAT)
        # time.sleep(1)
        # self.repeat_backward()

        for i in range(0, 20):
            self.positions = []
            init_position, init_orientation = [10, 0, -6], [0, 0, 0]
            self.env.set_initial_pose(init_position, init_orientation)
            self.env.set_mode(constants.AIRSIM_MODE_REPEAT)
            time.sleep(1)
            print ("Running repeating phase")
            self.repeat()

            try:
                f = open("positions{}.txt".format(i), 'wb')
                pickle.dump(self.positions, f)
            except IOError:
                print ("Could not open file!")
