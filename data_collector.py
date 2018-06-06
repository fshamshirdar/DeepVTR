from PIL import Image
import os
import os.path
import random
import math

import gym
import gym_airsim

from AirSimClient import AirSimClientBase # needed for write_png

import constants

class DataCollector:
    def __init__(self, datapath):
        self.base_path = datapath
        self.env = gym.make('AirSim-v1')
        self.env.reset()

    def random_walk(self):
        action = random.randint(0, constants.LOCO_NUM_CLASSES-1)
        state, _, done, _ = self.env.step(action)
        return state, action, done

    def play_round(self, index):
        if not os.path.exists(os.path.join(self.base_path, str(index))):
            os.makedirs(os.path.join(self.base_path, str(index)))
        actionFile = open(os.path.join(self.base_path, str(index), 'action.txt'), 'w', 1)

        states = []
        actions = []

        current_state = self.env.reset()
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
            self.env.random_initial_pose()
            _, actions = self.play_round(i)
            print ("Round %d: collected %d images" % (i, len(actions)))
