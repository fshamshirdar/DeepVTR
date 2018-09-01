import time
import random
import math
import numpy as np
# from multiprocessing import Process
from collections import deque
import torch
import gym
import gym_airsim

from multi_agent import MultiAgent
import constants

class AirSimSingleAgent:
    def __init__(self, placeRecognition, trail, navigation):
        self.env = gym.make('AirSim-v1') 
        self.placeRecognition = placeRecognition
        self.trail = trail
        self.navigation = navigation
        self.agent_state = constants.MULTI_AGENT_STATE_SEARCH
        self.temporary_trail = []
        self.goal = None
        self.init = self.env.reset()
        self.cycle = 0
        self.sequence = deque(maxlen=constants.SEQUENCE_LENGTH)
        self.current_state = None
        self.previous_action = 0
        self.placeRecognition.model.eval()
        self.navigation.model.eval()
        self.test_actions = [0, 0, 0, 0, 3, 0, 1, 2, 1, 1, 1, 1, 1, 1, 1, 0, 4, 0, 5, 0, 0]
        self.test_action_id = 0

    def reset_episode(self):
        self.test_action_id = 0
        init_position, init_orientation = [10, 0, -6], [0, 0, 0]
        self.env.set_initial_pose(init_position, init_orientation)
        self.env.set_mode(constants.AIRSIM_MODE_REPEAT)
        time.sleep(1)
        state = self.env.reset()
        self.init = state
        self.current_state = state
        self.temporary_trail = [state]
        self.sequence = deque(maxlen=constants.SEQUENCE_LENGTH)
        return state

    def goal_reached(self, pose):
#        return (self.test_action_id >= len(self.test_actions)) # TODO: temporary
        return (math.fabs(pose['x_pos'] - 12.6) < 0.5 and
                math.fabs(pose['y_pos'] - 1.8) < 0.5 and
                math.fabs(pose['yaw'] - 1.2) < 0.5)

    def search(self):
        print ("pose: ", self.env.get_position_orientation())
        # index, score, velocity = self.trail.find_closest_waypoint(self.sequence)
        index, score, velocity = self.trail.find_best_waypoint(self.sequence)
        # index, score, velocity = self.trail.find_most_similar_waypoint(self.sequence)
        # index = -1
        if (index == -1): # or random.random() < constants.MULTI_AGENT_RANDOM_MOVEMENT_CHANCE): # TODO
            # action = random.randint(0, constants.LOCO_NUM_CLASSES-1)
            action = self.test_actions[self.test_action_id]
            self.test_action_id += 1
            next_state, _, done, info = self.env.step(action)
            print (action)
        else:
            # if (index+1 >= self.trail.len()):
            #     future_state = self.trail.waypoints[index].state
            # else:
            #     future_state = self.trail.waypoints[index+1].state
            future_state = self.trail.waypoints[index].state
            actions = self.navigation.forward(self.current_state, self.trail.waypoints[index].state, future_state)
            actions = torch.squeeze(actions)
            sorted_actions, indices = torch.sort(actions, descending=True)
            action = indices[0]
            if ((self.previous_action == 0 and action == 3) or
                (self.previous_action == 3 and action == 0) or
                (self.previous_action == 1 and action == 2) or
                (self.previous_action == 2 and action == 1) or
                (self.previous_action == 4 and action == 5) or
                (self.previous_action == 5 and action == 4)):
                action = indices[1]
            print ("matched: ", index, score, actions)
            next_state, _, done, info = self.env.step(action)

            from PIL import Image
            # current_image = Image.fromarray(self.current_state)
            future_image = Image.fromarray(future_state)
            # current_image.save("current.png", "PNG")
            future_image.save("future.png", "PNG")

        self.previous_action = action
        self.current_state = next_state
        if (self.trail.len() > 0):
            next_rep = self.placeRecognition.forward(next_state)
            self.sequence.append(next_rep)
        self.temporary_trail.append(next_state)
        return info

    def update(self, cycle):
        self.cycle = cycle
        if (self.agent_state == constants.MULTI_AGENT_STATE_SEARCH):
            if (len(self.temporary_trail) > 100): # TODO: temporary
                print ("got too long, resetting")
                self.reset_episode()

            pose = self.search()
            if (self.goal_reached(pose)):
                # self.agent_state = constants.MULTI_AGENT_STATE_HOME
                steps_to_goal = len(self.temporary_trail)
                print ("goal reached, trail len: ", len(self.temporary_trail))
                for state in self.temporary_trail:
                    self.trail.append_waypoint(input=state, created_at=self.cycle, steps_to_goal=steps_to_goal)
                    steps_to_goal -= 1
                self.reset_episode()

class MultiAirSimAgent(MultiAgent):
    def __init__(self, placeRecognition, navigation):
        super(MultiAirSimAgent, self).__init__(placeRecognition, navigation)
        self.placeRecognition = placeRecognition
        self.navigation = navigation
        self.cycle = 0
        self.agents = []
        self.placeRecognition.model.eval()
        self.navigation.model.eval()
        for i in range(constants.MULTI_NUM_AGENTS):
            self.agents.append(AirSimSingleAgent(self.placeRecognition, self.trail, self.navigation))

    def run(self):
        for i in range(constants.MULTI_NUM_AGENTS):
            self.agents[i].reset_episode()

        while (True):
            self.cycle += 1
            for i in range(constants.MULTI_NUM_AGENTS):
                self.agents[i].update(self.cycle)
            self.trail.update_waypoints()
