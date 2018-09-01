from PIL import Image
import os
import os.path
import random
import math

from vizdoom import *

import constants

class VizDoomDataCollector:
    def __init__(self, datapath, wad):
        self.base_path = datapath
        self.seed = self.new_seed()
        self.game = self.initialize_game(wad, game_args=[])

    def new_seed(self):
        self.seed = random.randint(1, 1234567890)
        return self.seed

    def initialize_game(self, wad, game_args):
        game = DoomGame()
        game.load_config(constants.VIZDOOM_DEFAULT_CONFIG)
        for args in game_args:
            game.add_game_args(args)
        game.set_doom_scenario_path(wad)
        game.set_seed(self.seed)
        game.init()
        return game

    def reset_map(self):
        selected_map = (constants.VIZDOOM_MAP_NAME_TEMPLATE % random.randint(constants.VIZDOOM_MIN_RANDOM_TEXTURE_MAP_INDEX, constants.VIZDOOM_MAX_RANDOM_TEXTURE_MAP_INDEX))
        self.game.set_doom_map(selected_map)
        return self.reset_episode()

    def reset_episode(self):
        self.game.set_seed(self.seed)
        self.game.new_episode()
        state = self.game.get_state().screen_buffer.transpose([1, 2, 0])
        return state

    def step(self, action, repeat=4):
        self.game.make_action(constants.VIZDOOM_ACTIONS_LIST[action], repeat)
        self.game.make_action(constants.VIZDOOM_STAY_IDLE, repeat * 2)
        state = self.game.get_state().screen_buffer.transpose([1, 2, 0])
        # time.sleep(0.1)
        return state

    def random_walk(self):
        action = random.randint(0, constants.LOCO_NUM_CLASSES-1)
        state = self.step(action)
        return state, action, False

    def play_round(self, index):
        if not os.path.exists(os.path.join(self.base_path, str(index))):
            os.makedirs(os.path.join(self.base_path, str(index)))
        actionFile = open(os.path.join(self.base_path, str(index), 'action.txt'), 'w', 1)

        states = []
        actions = []

        current_state = self.reset_map()
        for i in range(5): # needed for skipping first outliers
            current_state, _, _ = self.random_walk()

        for i in range(constants.DATA_COLLECTION_PLAYING_ROUNG_LENGTH):
            future_state, action, done = self.random_walk()
            if (done == True):
                break

            # AirSimClientBase.write_png(os.path.join(self.base_path, str(index), str(i)+".png"), current_state)
            image = Image.fromarray(current_state)
            image.save(os.path.join(self.base_path, str(index), str(i)+".png"), "PNG")
            actionFile.write("%d\n" % action)

            current_state = future_state

            ## since we are already storing the file the line above
            # states.append(current_state) 
            actions.append(action)

        actionFile.close()
        return states, actions

    def collect(self, index):
        for i in range(index, index+constants.DATA_COLLECTION_ROUNDS):
            _, actions = self.play_round(i)
            print ("Round %d: collected %d images" % (i, len(actions)))
