import logging
import random
import math

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Tuple, Box, Discrete, MultiDiscrete, Dict
from gym.spaces.box import Box

from gym_airsim.envs.my_airsim_client import *
from gym_airsim.envs.cv_airsim_client import *
from AirSimClient import *
import constants

logger = logging.getLogger(__name__)

class AirSimEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
#       self.state = (10, 10, 0, 0)
#       self.action_space = spaces.Box(low=-1, high=1, shape=(1))
        self.observation_space = spaces.Box(low=0, high=255, shape=(30, 100, 1))
        self.state = np.zeros((30, 100, 1), dtype=np.uint8) 
        self.action_space = spaces.Discrete(3)
        self.mode = constants.AIRSIM_MODE_DATA_COLLECTION
        self._seed()

        # self.client = myAirSimClient()
        self.client = CVAirSimClient()
        self.steps = 0

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        self.steps += 1
        collided = self.client.take_action(action)
        position = self.client.getPosition()
        
        info = {"x_pos" : position.x_val, "y_pos" : position.y_val, "z_pos": position.z_val}
        self.state = self.client.getImage()
        reward = 0
        done = collided

        return self.state, reward, done, info

    def _reset(self):
        self.client._reset()
        self.steps = 0

        position = self.client.getPosition()
        self.state = self.client.getImage()
        
        return self.state

    def _render(self, mode='human', close=False):
        return

    def set_mode(self, mode):
        self.mode = mode
        self.client.set_mode(mode)

    def random_initial_pose(self):
        height = random.uniform(constants.DATA_COLLECTION_MIN_HEIGHT, constants.DATA_COLLECTION_MAX_HEIGHT)
        pose = [random.uniform(-150.0, 150.0), random.uniform(-150.0, 150.0), -height]
        orientation = [0.0, 0.0, random.uniform(-math.pi, math.pi)]
        self.client.set_pose(pose, orientation)

        time.sleep(1)
        collision = self.client.getCollisionInfo()
        if collision.has_collided:
            return self.random_initial_pose()

        """
        height = random.uniform(constants.DATA_COLLECTION_MIN_HEIGHT, constants.DATA_COLLECTION_MAX_HEIGHT)
        # pose = [random.uniform(-400.0, 400.0), random.uniform(-400.0, 400.0), -height]
        # orientation = [0.0, 0.0, random.uniform(-math.pi, math.pi)]
        # self.client.set_pose(pose, orientation)

        self.client.set_height(height)

        #time.sleep(1)
        #collision = self.client.getCollisionInfo()
        #if collision.has_collided:
        #    return self.random_initial_pose()
        """

    def set_height(self, height):
        self.client.set_height(height)

    def set_initial_pose(self, pose, orientation):
        self.client.set_pose(pose, orientation)

    def get_position_orientation(self):
        pos = self.client.getPosition()
        orientation = self.client.getOrientation()
        return pos, orientation
