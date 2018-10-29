import time
import random
import numpy as np
from collections import deque
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

    def random_walk(self):
        state = self.env.reset()
        self.init = state
        for i in range(constants.AIRSIM_AGENT_TEACH_LEN):
            action = random.randint(0, constants.LOCO_NUM_CLASSES-1)
            next_state, _, done, _ = self.env.step(action)
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

        state = self.env.reset()
        self.init = state
        i = 0
        actions = [int(val) for val in action_file.read().split('\n') if val.isdigit()]
        for action in actions:
            next_state, _, done, _ = self.env.step(action)
            print ("commanded walk: index %d action %d" % (i, action))
            rep, _ = self.sptm.append_keyframe(state, action, done)
            self.goal = state
            state = next_state
            i = i+1
            if done:
                break

    def teach(self):
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
        while (True):
            matched_index, similarity_score, best_velocity = self.sptm.relocalize(current_state)
            path = self.sptm.find_shortest_path(matched_index, goal_index)
            print (matched_index, similarity_score, path)
            if (len(path) < 2): # achieved the goal
                break

            if (constants.ACTION_LOOKAHEAD_ENABLED):
                action, prob, future_state = self.path_lookahead(previous_state, current_state, path)
            else:
                future_state = self.sptm.memory[path[1]].state
                # actions = self.navigation.forward(previous_state, current_state, future_state)
                # actions = self.navigation.forward(current_state, self.sptm.memory[matched_index].state, future_state)

                actions = self.navigation.forward(current_state, None, future_state)
                actions = torch.squeeze(actions)
                sorted_actions, indices = torch.sort(actions, descending=True)
                action = indices[0]
                if ((previous_action == constants.ACTION_MOVE_FORWARD and action == constants.ACTION_MOVE_BACKWARD) or
                    (previous_action == constants.ACTION_MOVE_BACKWARD and action == constants.ACTION_MOVE_FORWARD) or
                    (previous_action == constants.ACTION_TURN_RIGHT and action == constants.ACTION_TURN_LEFT) or
                    (previous_action == constants.ACTION_TURN_LEFT and action == constants.ACTION_TURN_RIGHT) or
                    (previous_action == constants.ACTION_MOVE_RIGHT and action == constants.ACTION_MOVE_LEFT) or
                    (previous_action == constants.ACTION_MOVE_LEFT and action == constants.ACTION_MOVE_RIGHT)):
                    action = indices[1]

                # prob, pred = torch.max(actions.data, 0)
                # prob = prob.data.cpu().item()
                # action = pred.data.cpu().item()
                print ("actions: ", actions, action)

                # select based on probability distribution
                # action = np.random.choice(np.arange(0, constants.LOCO_NUM_CLASSES), p=actions.data.cpu().numpy())
                # prob = actions[action].data.cpu().item()

            from PIL import Image
            current_image = Image.fromarray(current_state)
            future_image = Image.fromarray(future_state)
            current_image.save("current.png", "PNG")
            future_image.save("future.png", "PNG")
            next_state, _, done, _ = self.env.step(action)
            previous_state = current_state
            current_state = next_state
            previous_action = action
            if (done):
                break

    def repeat_backward(self):
        self.sptm.build_graph()
        goal, goal_index, similarity = self.sptm.find_closest(self.init)
        if (goal_index < 0):
            print ("cannot find goal")
            return

        future_state = self.env.reset()
        self.sptm.clear_sequence()
        while (True):
            matched_index, similarity_score, best_velocity = self.sptm.relocalize(future_state, backward=True)
            path = self.sptm.find_shortest_path(matched_index, goal_index)
            print (matched_index, similarity_score, path)
            if (len(path) < 2): # achieved the goal
                break
            current_state = self.sptm.memory[path[1]].state
            if (len(path) > 2):
                previous_state = self.sptm.memory[path[2]].state
            else:
                previous_state = current_state

            from PIL import Image
            current_image = Image.fromarray(current_state)
            future_image = Image.fromarray(future_state)
            current_image.save("current.png", "PNG")
            future_image.save("future.png", "PNG")
            actions = self.navigation.forward(previous_state, current_state, future_state)
            print (actions)
            prob, pred = torch.max(actions.data, 1)
            action = pred.data.cpu().item()
            if (action == 0):
                action = -1
            elif (action == 1):
                action = 2
            elif (action == 2):
                action = 1
            print ("action %d" % action)
            next_state, _, done, _ = self.env.step(action)
            future_state = next_state
            if (done):
                break

    def run(self):
        init_position, init_orientation = [10, 0, -6], [0, 0, 0]
        self.env.set_initial_pose(init_position, init_orientation)
        self.env.set_mode(constants.AIRSIM_MODE_TEACH)
        time.sleep(1)
        print ("Running teaching phase")
        self.teach()

        # print ("Running repeating backward phase")
        # self.env.set_mode(constants.AIRSIM_MODE_REPEAT)
        # time.sleep(1)
        # self.repeat_backward()

        init_position, init_orientation = [10, 0, -6], [0, 0, 0]
        self.env.set_initial_pose(init_position, init_orientation)
        self.env.set_mode(constants.AIRSIM_MODE_REPEAT)
        time.sleep(1)
        print ("Running repeating phase")
        self.repeat()

        init_position, init_orientation = [10, 0, -6], [0, 0, 0]
        self.env.set_initial_pose(init_position, init_orientation)
        self.env.set_mode(constants.AIRSIM_MODE_REPEAT)
        time.sleep(1)
        print ("Running repeating phase")
        self.repeat()
