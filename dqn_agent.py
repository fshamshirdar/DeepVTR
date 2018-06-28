import time
import math
import random
import numpy as np
from collections import deque
import torch
import gym
import gym_airsim

from replay_memory import SequentialMemory
from agent import Agent
import constants

class DQNAgent(Agent):
    def __init__(self, placeRecognition=None, navigation=None):
        super(DQNAgent, self).__init__(placeRecognition, navigation)
        self.env = gym.make('AirSim-v1')
        self.env.reset()
        self.goal = None
        self.init = None
        self.path = []

        self.memory = SequentialMemory(limit=constants.DQN_MEMORY_SIZE, window_length=1)

    def random_walk(self):
        state = self.env.reset()
        self.init = state
        for i in range(constants.DQN_LOCO_TEACH_LEN):
            action = random.randint(0, constants.LOCO_NUM_CLASSES-1)
            action = 0
            next_state, _, done, info = self.env.step(action)
            position = (info['x_pos'], info['y_pos'], info['z_pos'])
            rep, _ = self.sptm.append_keyframe(state, action, done, position=position)
            self.path.append(position)
            self.goal = state
            state = next_state
            if done:
                break

    def teach(self):
        self.random_walk()
        
    def repeat(self):
        self.sptm.build_graph()
        goal, goal_index, similarity = self.sptm.find_closest(self.goal)
        if (goal_index < 0):
            print ("cannot find goal")
            return

        sequence = deque(maxlen=constants.SEQUENCE_LENGTH)

        current_state = self.env.reset()
        previous_state = current_state
        sequence.append(current_state)
        while (True):
            matched_index, similarity_score, best_velocity = self.sptm.relocalize(sequence)
            path = self.sptm.find_shortest_path(matched_index, goal_index)
            print (matched_index, similarity_score, path)
            if (len(path) < 2): # achieved the goal
                return

            closest_state = self.sptm.memory[matched_index].state
            future_state = self.sptm.memory[path[1]].state
            actions = self.navigation.forward(current_state, closest_state, future_state)
            prob, pred = torch.max(actions.data, 1)
            prob = prob.data.cpu().item()
            action = pred.data.cpu().item()
            print ("action %d" % action)

#            m = Categorical(actions)
#            action = m.sample()

            # from PIL import Image
            # current_image = Image.fromarray(current_state)
            # future_image = Image.fromarray(future_state)
            # current_image.save("current.png", "PNG")
            # future_image.save("future.png", "PNG")
            next_state, _, done, info = self.env.step(action)
            position = (info['x_pos'], info['y_pos'], info['z_pos'])
            reward = (constants.DQN_REWARD_DISTANCE_OFFSET - self.compute_reward(position))
            print ("reward {}".format(reward))
            previous_state = current_state
            current_state = next_state
            sequence.append(current_state)
            if (done):
                return

    def compute_reward(self, current_position):
        if (len(self.path) < 1):
            return 0
        distances = [self.calculate_distance(current_position, position) for position in self.path]
        distances.sort()
        return distances[0]

    def calculate_distance(self, start_coordinates, current_coordinates):
        distance = math.sqrt((start_coordinates[0] - current_coordinates[0]) ** 2 +
                             (start_coordinates[1] - current_coordinates[1]) ** 2 + 
                             (start_coordinates[2] - current_coordinates[2]) ** 2)
        # abs_angle_difference = math.fabs(start_coordinates[3] - current_coordinates[3])
        # angle = min(abs_angle_difference, 360.0 - abs_angle_difference)
        return distance

    def run(self):
        init_position, init_orientation = [10, 0, -6], [0, 0, 0]
        self.env.set_initial_pose(init_position, init_orientation)
        self.env.set_mode(constants.AIRSIM_MODE_TEACH)
        time.sleep(1)
        print ("Running teaching phase")
        self.teach()

        init_position, init_orientation = [10, 0, -6], [0, 0, 0]
        self.env.set_initial_pose(init_position, init_orientation)
        self.env.set_mode(constants.AIRSIM_MODE_REPEAT)
        time.sleep(1)
        print ("Running repeating phase")
        self.repeat()
