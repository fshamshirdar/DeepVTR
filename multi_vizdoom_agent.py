import time
import random
import math
import numpy as np
# from multiprocessing import Process
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
        self.temporary_trail = []
        self.goal = None
        self.init = None
        self.seed = seed
        self.cycle = 0
        self.current_state = None
        self.previous_action = 0
        self.last_matched = []
        self.game = self.initialize_game(wad, game_args)
        self.placeRecognition.model.eval()
        # self.placeRecognition.siamesenet.eval()
        self.navigation.model.eval()

    def initialize_game(self, wad, game_args):
        game = DoomGame()
        game.load_config(constants.VIZDOOM_DEFAULT_CONFIG)
        for args in game_args:
            game.add_game_args(args)
        game.set_doom_scenario_path(wad)
        game.set_seed(self.seed)
        game.init()
        return game

    def get_distance_to_obstacles(self, depth_buf):
#        min_left = 255
#        min_right = 255
#        for i in range(depth_buf.shape[0]):
#            if (depth_buf[i, 0] < min_left):
#                min_left = depth_buf[i, 0]
#            if (depth_buf[i, depth_buf.shape[1]-1] < min_right):
#                min_right = depth_buf[i, depth_buf.shape[1]-1]
#        return min_left, min_right

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

    def reset_episode(self):
        self.game.set_seed(self.seed)
        self.game.new_episode()
        state = self.game.get_state().screen_buffer.transpose([1, 2, 0])
        self.init = state
        self.current_state = state
        self.temporary_trail = [state]
        self.last_matched = []
        self.trail.clear_sequence()
        return state

    def step(self, action, repeat=4):
        self.game.make_action(constants.VIZDOOM_ACTIONS_LIST[action], repeat)
        state = self.game.get_state().screen_buffer.transpose([1, 2, 0])
        # time.sleep(0.1)
        return state

    def goal_reached(self, pose):
        # goal_location = (609.92807007, 247.60987854, 0)
        # goal_location = (-97.92807007, -220.60987854, 0)
        goal_location = (700.92807007, -388.60987854, 0)
        distance = math.sqrt((pose[0] - goal_location[0]) ** 2 +
                             (pose[1] - goal_location[1]) ** 2 + 
                             (pose[2] - goal_location[2]) ** 2)

        return (distance < constants.VIZDOOM_GOAL_DISTANCE_THRESHOLD)

    def search(self):
        # self.last_matched = []
        # print ("pose: ", self.game.get_state().game_variables)
        index, score, velocity, self.last_matched = self.trail.find_closest_waypoint(self.current_state, backward=False, last_matched=self.last_matched)
        # index, score, velocity, self.last_matched = self.trail.find_best_waypoint(self.current_state, backward=False, last_matched=self.last_matched)
        # index, score, velocity, self.last_matched = self.trail.find_most_similar_waypoint(self.current_state, backward=False, last_matched=self.last_matched)
        # index = -1

        depth_buf = self.game.get_state().depth_buffer
        left_dist, right_dist = self.get_distance_to_obstacles(depth_buf)
        if (index == -1):
            # action = random.choice([0, 0, 0, 0, 1, 2, 3, 4, 5])
            if (self.cycle % 5 == 0):
                # action = random.randint(0, constants.LOCO_NUM_CLASSES-1)
                permitted_actions = [i for i in range(0, constants.LOCO_NUM_CLASSES)]
                permitted_actions.remove(constants.ACTION_MOVE_BACKWARD)
                if (self.previous_action == constants.ACTION_MOVE_FORWARD and constants.ACTION_MOVE_BACKWARD in permitted_actions):
                    permitted_actions.remove(constants.ACTION_MOVE_BACKWARD)
                elif (self.previous_action == constants.ACTION_MOVE_BACKWARD and constants.ACTION_MOVE_FORWARD in permitted_actions):
                    permitted_actions.remove(constants.ACTION_MOVE_FORWARD)
                elif (self.previous_action == constants.ACTION_TURN_RIGHT and constants.ACTION_TURN_LEFT in permitted_actions):
                    permitted_actions.remove(constants.ACTION_TURN_LEFT)
                elif (self.previous_action == constants.ACTION_TURN_LEFT and constants.ACTION_TURN_RIGHT in permitted_actions):
                    permitted_actions.remove(constants.ACTION_TURN_RIGHT)
                elif (self.previous_action == constants.ACTION_MOVE_RIGHT and constants.ACTION_MOVE_LEFT in permitted_actions):
                    permitted_actions.remove(constants.ACTION_MOVE_LEFT)
                elif (self.previous_action == constants.ACTION_MOVE_LEFT and constants.ACTION_MOVE_RIGHT in permitted_actions):
                    permitted_actions.remove(constants.ACTION_MOVE_RIGHT)
                action = random.choice(permitted_actions)
            elif (self.cycle % 5 == 3):
                action = 0
            else:
                action = self.previous_action

            if (left_dist < 6):
                action = constants.ACTION_TURN_RIGHT
            elif (right_dist < 6): 
                action = constants.ACTION_TURN_LEFT

            # wait()
        elif (random.random() < constants.MULTI_AGENT_RANDOM_MOVEMENT_CHANCE):
            permitted_actions = [i for i in range(0, constants.LOCO_NUM_CLASSES)]
            permitted_actions.remove(constants.ACTION_MOVE_BACKWARD)
            action = random.choice(permitted_actions)
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
            if ((self.previous_action == constants.ACTION_MOVE_FORWARD and action == constants.ACTION_MOVE_BACKWARD) or
                (self.previous_action == constants.ACTION_MOVE_BACKWARD and action == constants.ACTION_MOVE_FORWARD) or
                (self.previous_action == constants.ACTION_TURN_RIGHT and action == constants.ACTION_TURN_LEFT) or
                (self.previous_action == constants.ACTION_TURN_LEFT and action == constants.ACTION_TURN_RIGHT) or
                (self.previous_action == constants.ACTION_MOVE_RIGHT and action == constants.ACTION_MOVE_LEFT) or
                (self.previous_action == constants.ACTION_MOVE_LEFT and action == constants.ACTION_MOVE_RIGHT)):
                action = indices[1]
            print ("matched: ", index, score, actions)

            # from PIL import Image
            # future_image = Image.fromarray(future_state)
            # future_image.save("future.png", "PNG")
            # time.sleep(0.1)
            wait()

        # if (left_dist < 5):
        #     action = constants.ACTION_TURN_RIGHT
        # elif (right_dist < 5): 
        #     action = constants.ACTION_TURN_LEFT

        next_state = self.step(action, repeat=1)

        self.previous_action = action
        self.current_state = next_state
        self.temporary_trail.append(next_state)

    def update(self, cycle):
        self.cycle = cycle
        if (self.agent_state == constants.MULTI_AGENT_STATE_SEARCH):
            if (len(self.temporary_trail) > 700): # TODO: temporary
                print ("got too long, resetting")
                self.reset_episode()

            self.search()
            pose = self.game.get_state().game_variables
            if (self.goal_reached(pose)):
                # self.agent_state = constants.MULTI_AGENT_STATE_HOME
                steps_to_goal = len(self.temporary_trail)
                print ("goal reached, trail len: ", len(self.temporary_trail))
                for state in self.temporary_trail:
                    self.trail.append_waypoint(input=state, created_at=self.cycle, steps_to_goal=steps_to_goal)
                    steps_to_goal -= 1
                    from PIL import Image
                    image = Image.fromarray(state)
                    image.save("image_%d.png" % steps_to_goal, "PNG")
                self.reset_episode()

class MultiVizDoomAgent(MultiAgent):
    def __init__(self, placeRecognition, navigation, wad):
        super(MultiVizDoomAgent, self).__init__(placeRecognition, navigation)
        self.placeRecognition = placeRecognition
        self.navigation = navigation
        self.wad = wad
        self.seed = self.new_seed()
        self.cycle = 0
        self.agents = []
        self.placeRecognition.model.eval()
        self.navigation.model.eval()
        for i in range(constants.MULTI_NUM_AGENTS):
            self.agents.append(VizDoomSingleAgent(self.placeRecognition, self.trail, self.navigation, self.wad, self.seed, game_args=[]))

    def new_seed(self):
        self.seed = random.randint(1, 1234567890)
        return self.seed

    def run(self):
        # p1 = Process(target=self.agent1.run())
        # p1.start()
        # self.agent2.run()

        # selected_map = (constants.VIZDOOM_MAP_NAME_TEMPLATE % random.randint(constants.VIZDOOM_MIN_RANDOM_TEXTURE_MAP_INDEX, constants.VIZDOOM_MAX_RANDOM_TEXTURE_MAP_INDEX))
        selected_map = (constants.VIZDOOM_MAP_NAME_TEMPLATE % random.randint(constants.VIZDOOM_MIN_TEST_MAP_INDEX, constants.VIZDOOM_MAX_TEST_MAP_INDEX))
        for i in range(constants.MULTI_NUM_AGENTS):
            self.agents[i].set_map(selected_map)
            self.agents[i].reset_episode()

        while (True):
            self.cycle += 1
            for i in range(constants.MULTI_NUM_AGENTS):
                self.agents[i].update(self.cycle)
            self.trail.update_waypoints()
