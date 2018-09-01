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
        self.sequence = deque(maxlen=constants.SEQUENCE_LENGTH)
        self.current_state = None
        self.previous_action = 0
        self.game = self.initialize_game(wad, game_args)
        self.placeRecognition.model.eval()
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
        return state

    def step(self, action, repeat=4):
        self.game.make_action(constants.VIZDOOM_ACTIONS_LIST[action], repeat)
        state = self.game.get_state().screen_buffer.transpose([1, 2, 0])
        # time.sleep(0.1)
        return state

    def goal_reached(self, pose):
        return (math.fabs(pose[0] - 630.0) < 50. and
                math.fabs(pose[1] - 550.0) < 50. and
                math.fabs(pose[3] - 130.) < 30.)

    def search(self):
        index, score, velocity = self.trail.find_closest_waypoint(self.sequence)
        # index, score, velocity = self.trail.find_best_waypoint(self.sequence)
        # index, score, velocity = self.trail.find_most_similar_waypoint(self.sequence)
        # index = -1
        if (index == -1): # or random.random() < constants.MULTI_AGENT_RANDOM_MOVEMENT_CHANCE):
            action = random.randint(0, constants.LOCO_NUM_CLASSES-1)
            next_state = self.step(action)
        else:
            if (index+1 >= self.trail.len()):
                future_state = self.trail.waypoints[index].state
            else:
                future_state = self.trail.waypoints[index+1].state
            # future_state = self.trail.waypoints[index].state
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
            next_state = self.step(action, repeat=2)

            from PIL import Image
            # current_image = Image.fromarray(self.current_state)
            future_image = Image.fromarray(self.trail.waypoints[index].state)
            # current_image.save("current.png", "PNG")
            future_image.save("future.png", "PNG")

        self.previous_action = action
        self.current_state = next_state
        if (self.trail.len() > 0):
            next_rep = self.placeRecognition.forward(next_state).data.cpu()
            self.sequence.append(next_rep)
        self.temporary_trail.append(next_state)

    def update(self, cycle):
        self.cycle = cycle
        if (self.agent_state == constants.MULTI_AGENT_STATE_SEARCH):
            if (len(self.temporary_trail) > 200): # TODO: temporary
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
                self.reset_episode()

class MultiVizDoomAgent(MultiAgent):
    def __init__(self, placeRecognition, navigation, wad, teachCommandsFile=None):
        super(MultiVizDoomAgent, self).__init__(placeRecognition, navigation)
        self.placeRecognition = placeRecognition
        self.navigation = navigation
        self.wad = wad
        self.seed = self.new_seed()
        self.cycle = 0
        self.agents = []
        self.placeRecognition.model.eval()
        self.navigation.model.eval()
        for i in range(constants.VIZDOOM_MULTI_NUM_AGENTS):
            self.agents.append(VizDoomSingleAgent(self.placeRecognition, self.trail, self.navigation, self.wad, self.seed, game_args=[]))

    def new_seed(self):
        self.seed = 100 # random.randint(1, 1234567890)
        return self.seed

    def run(self):
        # p1 = Process(target=self.agent1.run())
        # p1.start()
        # self.agent2.run()

        selected_map = (constants.VIZDOOM_MAP_NAME_TEMPLATE % random.randint(constants.VIZDOOM_MIN_RANDOM_TEXTURE_MAP_INDEX, constants.VIZDOOM_MAX_RANDOM_TEXTURE_MAP_INDEX))
        for i in range(constants.VIZDOOM_MULTI_NUM_AGENTS):
            self.agents[i].set_map(selected_map)
            self.agents[i].reset_episode()

        while (True):
            self.cycle += 1
            for i in range(constants.VIZDOOM_MULTI_NUM_AGENTS):
                self.agents[i].update(self.cycle)
            self.trail.update_waypoints()
