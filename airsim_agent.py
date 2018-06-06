import time
import random
import numpy as np
from collections import deque
import torch
import gym
import gym_airsim

from agent import Agent
from AirSimClient import AirSimClientBase # needed for write_png
import constants

class AirSimAgent(Agent):
    def __init__(self, placeRecognition=None, navigation=None):
        super(AirSimAgent, self).__init__(placeRecognition, navigation)
        self.env = gym.make('AirSim-v1')
        self.env.reset()
        self.goal = None
        self.init = None

    def random_walk(self):
        action = random.randint(0, constants.LOCO_NUM_CLASSES-1)
        action = 0
        next_state, _, done, _ = self.env.step(action)
        return next_state, action, done

    def teach(self):
        state = self.env.reset()
        self.init = state
        for i in range(constants.AIRSIM_AGENT_TEACH_LEN):
            next_state, action, done = self.random_walk()
            print ("index %d action %d" % (i, action))
            rep, _ = self.sptm.append_keyframe(state, action, done)
            self.goal = state
            state = next_state
            if done:
                break 

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
            matched_index, similarity_score = self.sptm.relocalize(sequence)
            path = self.sptm.find_shortest_path(matched_index, goal_index)
            print (matched_index, similarity_score, path)
            if (len(path) < 2): # achieved the goal
                break
            future_state = self.sptm.memory[path[1]].state

            from PIL import Image
            current_image = Image.fromarray(current_state)
            future_image = Image.fromarray(future_state)
            current_image.save("current.png", "PNG")
            future_image.save("future.png", "PNG")
            actions = self.navigation.forward(previous_state, current_state, future_state)
            print (actions)
            prob, pred = torch.max(actions.data, 1)
            action = pred.data.cpu().item()
            print ("action %d" % action)
            next_state, _, done, _ = self.env.step(action)
            previous_state = current_state
            current_state = next_state
            sequence.append(current_state)

    def repeat_backward(self):
        self.sptm.build_graph()
        goal, goal_index, similarity = self.sptm.find_closest(self.init)
        if (goal_index < 0):
            print ("cannot find goal")
            return

        sequence = deque(maxlen=constants.SEQUENCE_LENGTH)

        current_state = self.env.reset()
        previous_state = current_state
        sequence.append(current_state)
        while (True):
            matched_index, similarity_score = self.sptm.relocalize(sequence, backward=True)
            path = self.sptm.find_shortest_path(matched_index, goal_index)
            print (matched_index, similarity_score, path)
            if (len(path) < 2): # achieved the goal
                break
            future_state = self.sptm.memory[path[1]].state

            from PIL import Image
            current_image = Image.fromarray(current_state)
            future_image = Image.fromarray(future_state)
            current_image.save("current.png", "PNG")
            future_image.save("future.png", "PNG")
            actions = self.navigation.forward(previous_state, current_state, future_state)
            print (actions)
            prob, pred = torch.max(actions.data, 1)
            action = pred.data.cpu().item()
            print ("action %d" % action)
            next_state, _, done, _ = self.env.step(action)
            previous_state = current_state
            current_state = next_state
            sequence.append(current_state)

    def run(self):
        init_position, init_orientation = [10, 0, -6], [0, 0, 0]
        self.env.set_initial_pose(init_position, init_orientation)
        time.sleep(1)
        print ("Running teaching phase")
        self.teach()
        init_position, init_orientation = [10, 1, -8], [0, 0, 0.5]
        self.env.set_initial_pose(init_position, init_orientation)
        time.sleep(1)
        print ("Running repeating phase")
        self.repeat()
