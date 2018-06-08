import time
import random
import numpy as np
from collections import deque
import torch
import cv2

import rospy
import ros_numpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from agent import Agent
from AirSimClient import AirSimClientBase # needed for write_png
import constants

class BebopAgent(Agent):
    def __init__(self, placeRecognition=None, navigation=None, teachCommandsFile=None):
        super(BebopAgent, self).__init__(placeRecognition, navigation)
        rospy.init_node('bebop_agent')
        self.imageSubscriber = rospy.Subscriber("bebop/image_raw", Image, self.image_callback)
        self.commandPublisher = rospy.Publisher('bebop/cmd_vel', Twist, queue_size=10)
        self.bridge = CvBridge()

        self.state = constants.BEBOP_MODE_TEACH_MANUALLY
        self.latest_image = None

        self.goal = None
        self.init = None
        self.teachCommandsFile = teachCommandsFile

    def path_lookahead(self, previous_state, current_state, path):
        selected_action, selected_prob, selected_future_state = None, None, None
        i = 1
        for i in range(1, len(path)):
            future_state = self.sptm.memory[path[i]].state
            actions = self.navigation.forward(previous_state, current_state, future_state)
            prob, pred = torch.max(actions.data, 1)
            prob = prob.data.cpu().item()
            action = pred.data.cpu().item()
            print (action, prob)

            if selected_action == None:
                selected_action, selected_prob, selected_future_state = action, prob, future_state
            if (prob < constants.ACTION_LOOKAHEAD_PROB_THRESHOLD):
                break
            selected_action, selected_prob, selected_future_state = action, prob, future_state

        if (i > 8):
            self.sptm.add_shortcut(path[0], path[i], selected_prob)
        return selected_action, selected_prob, selected_future_state

    def repeat_backward(self):
        self.sptm.build_graph()
        goal, goal_index, similarity = self.sptm.find_closest(self.init)
        if (goal_index < 0):
            print ("cannot find goal")
            return

        sequence = deque(maxlen=constants.SEQUENCE_LENGTH)

        future_state = self.env.reset()
        sequence.append(future_state)
        while (True):
            matched_index, similarity_score = self.sptm.relocalize(sequence, backward=True)
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
            sequence.append(future_state)
            if (done):
                break

    def image_callback(self, image):
        self.latest_image = image

    def take_action(self, action):
        cmd_vel = Twist()
        duration = 1.
        if action == 0:
            cmd_vel.linear.x = constants.BEBOP_STRAIGHT_SPEED
        elif action == 1:
            cmd_vel.angular.z = constants.BEBOP_YAW_SPEED
        elif action == 2:
            cmd_vel.angular.z = -constants.BEBOP_YAW_SPEED

        self.actionPublisher.publish(cmd_vel)
        time.sleep(duration)

        cmd_stop_vel = Twist()
        self.commandPublisher.publish(cmd_stop_vel)
        time.sleep(duration / 3.) # let it stop completely

    def run(self):
        if (self.teachCommandsFile != None):
            teach_action_file = open(self.teachCommandsFile)
            teach_actions = [int(val) for val in teach_action_file.read().split('\n') if val.isdigit()]

        teach_index = 0
        goal_index = 0
        sequence = deque(maxlen=constants.SEQUENCE_LENGTH)
        previous_state = None

        rate = rospy.Rate(10) # 10hz
        while not rospy.is_shutdown():
            if (self.state == constants.BEBOP_MODE_TEACH_MANUALLY):
                if (self.latest_image == None):
                    print ("waiting for an image")
                    rate.sleep()
                    continue
                if (teach_index > 10):
                    self.state = constants.BEBOP_MODE_IDLE
                    print ("teaching is done")
                    continue

                state = ros_numpy.numpify(self.latest_image)
                cv_image = state[...,::-1] # rgb to bgr
                cv2.imshow("image", cv_image)
                cv2.waitKey(10)
                rep, _ = self.sptm.append_keyframe(state, -1, False)
                self.goal = state
                time.sleep(constants.BEBOP_ACTION_DURATION)
                teach_index = teach_index+1

            elif (self.state == constants.BEBOP_MODE_TEACH):
                if (self.latest_image == None):
                    print ("waiting for an image")
                    rate.sleep()
                    continue
                if teach_index >= len(teach_actions):
                    self.state = constants.BEBOP_MODE_IDLE
                    continue
                """
                try:
                    cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
                    self.latest_image = None
                except CvBridgeError as e:
                    print(e)
                    rate.sleep()
                    continue

                state = np.asarray(cv_image)
                """
                state = ros_numpy.numpify(self.latest_image)
                cv_image = state[...,::-1] # rgb to bgr
                cv2.imshow("image", cv_image)
                cv2.waitKey()
                action = teach_actions[teach_index]
                rep, _ = self.sptm.append_keyframe(state, action, False)
                print ("commanded walk: index %d action %d" % (teach_index, action))
                self.take_action(action)
                self.goal = state
                teach_index = teach_index+1

            elif (self.state == constants.BEBOP_MODE_PROCESS):
                self.sptm.build_graph()
                goal, goal_index, similarity = self.sptm.find_closest(self.goal)
                if (goal_index < 0):
                    print ("cannot find goal")
                    self.state = constants.BEBOP_MODE_IDLE

            elif (self.state == constants.BEBOP_MODE_REPEAT):
                if (self.latest_image == None):
                    print ("waiting for an image")
                    rate.sleep()
                    continue
                try:
                    cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
                    self.latest_image = None
                except CvBridgeError as e:
                    print(e)
                    rate.sleep()
                    continue
                current_state = np.asarray(cv_image)
                if (previous_state == None):
                    previous_state = current_state

                sequence.append(current_state)
                matched_index, similarity_score = self.sptm.relocalize(sequence)
                path = self.sptm.find_shortest_path(matched_index, goal_index)
                print (matched_index, similarity_score, path)
                if (len(path) < 2): # achieved the goal
                    self.state = constants.BEBOP_MODE_IDLE

                if (False and constants.ACTION_LOOKAHEAD_ENABLED): # Disabled now
                    action, prob, future_state = self.path_lookahead(previous_state, current_state, path)
                else:
                    future_state = self.sptm.memory[path[1]].state
                    actions = self.navigation.forward(previous_state, current_state, future_state)
                    print (actions)
                    prob, pred = torch.max(actions.data, 1)
                    prob = prob.data.cpu().item()
                    action = pred.data.cpu().item()
                    print ("action %d" % action)

                from PIL import Image
                current_image = Image.fromarray(current_state)
                future_image = Image.fromarray(future_state)
                current_image.save("current.png", "PNG")
                future_image.save("future.png", "PNG")
                self.take_action(action)
                previous_state = current_state
            else: # IDLE
                if (self.latest_image == None):
                    print ("waiting for an image")
                    rate.sleep()
                    continue
                current_state = ros_numpy.numpify(self.latest_image)
                sequence.append(current_state)
                matched_index, similarity_score = self.sptm.relocalize(sequence)
                print (matched_index, similarity_score)
                cv_image = self.sptm.memory[matched_index].state[...,::-1] # rgb to bgr
                cv2.imshow("image", cv_image)
                cv2.waitKey(10)
                time.sleep(constants.BEBOP_ACTION_DURATION)

            rate.sleep()
