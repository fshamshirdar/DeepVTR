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
from sensor_msgs.msg import Joy
from std_msgs.msg import Empty
# from cv_bridge import CvBridge, CvBridgeError

from agent import Agent
import constants

class PioneerAgent(Agent):
    def __init__(self, placeRecognition=None, navigation=None, teachCommandsFile=None):
        super(PioneerAgent, self).__init__(placeRecognition, navigation)
        rospy.init_node('pioneer_agent')
        self.joySubscriber = rospy.Subscriber("/joy", Joy, self.joy_callback)
        self.imageSubscriber = rospy.Subscriber("/camera/rgb", Image, self.image_callback)
        self.commandPublisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.closestMatchPublisher = rospy.Publisher('/closest_match', Image, queue_size=10)
        self.futureMatchPublisher = rospy.Publisher('/future_match', Image, queue_size=10)
        # self.bridge = CvBridge()

        self.set_state(constants.ROBOT_MODE_IDLE)
        self.latest_image = None
        self.joy = None

        self.goal = None
        self.goal_index = 0
        self.init = None
        self.teachCommandsFile = teachCommandsFile

    def image_callback(self, image):
        self.latest_image = image

    def joy_callback(self, joy):
        self.joy = joy
        if (joy.buttons[constants.JOY_BUTTONS_TEACH_ID]): # X
            if (self.state == constants.ROBOT_MODE_TEACH_MANUALLY):
                self.set_state(constants.ROBOT_MODE_IDLE)
                self.process()
            else:
                self.set_state(constants.ROBOT_MODE_TEACH_MANUALLY)
        elif (joy.buttons[constants.JOY_BUTTONS_REPEAT_ID]): # B
            self.set_state(constants.ROBOT_MODE_REPEAT)
        elif (joy.buttons[constants.JOY_BUTTONS_IDLE_ID]): # LB
            self.set_state(constants.ROBOT_MODE_IDLE)
        elif (joy.buttons[constants.JOY_BUTTONS_MANUAL_CONTROL_ID]): # RB
            self.set_state(constants.ROBOT_MODE_IDLE)
            cmd_vel = Twist()
            cmd_vel.linear.x = joy.axes[3]
            cmd_vel.linear.y = joy.axes[2]
            cmd_vel.linear.z = joy.axes[5]
            cmd_vel.angular.z = joy.axes[0]
            self.commandPublisher.publish(cmd_vel)
        elif (joy.buttons[constants.JOY_BUTTONS_CLEAR_MEMORY_ID]): # LT
            self.set_state(constants.ROBOT_MODE_IDLE)
            self.sptm.clear()
            print ("Memory has been cleared.")

    def process(self):
        self.sptm.build_graph()
        goal, self.goal_index, similarity = self.sptm.find_closest(self.goal)
        if (self.goal_index < 0):
            print ("Cannot find goal")

    def set_state(self, state):
        self.state = state
        print ("State: {}".format(constants.ROBOT_MODE_STRINGS[self.state]))

    def hover(self):
        cmd_stop_vel = Twist()
        self.commandPublisher.publish(cmd_stop_vel)
        time.sleep(constants.PIONEER_ACTION_STOP_DURATION) # let it stop completely

    def take_action(self, action):
        cmd_vel = Twist()
        if action == 0:
            cmd_vel.linear.x = constants.PIONEER_STRAIGHT_SPEED
        elif action == 1:
            cmd_vel.angular.z = -constants.PIONEER_YAW_SPEED
        elif action == 2:
            cmd_vel.angular.z = constants.PIONEER_YAW_SPEED

        start = time.time()
        duration = constants.PIONEER_ACTION_DURATION
        while duration > time.time() - start:
            self.commandPublisher.publish(cmd_vel)
            time.sleep(constants.PIONEER_ACTION_FREQ)

    def run(self):
        if (self.teachCommandsFile != None):
            teach_action_file = open(self.teachCommandsFile)
            teach_actions = [int(val) for val in teach_action_file.read().split('\n') if val.isdigit()]

        teach_index = 0
        previous_state = None

        rate = rospy.Rate(10) # 10hz
        while not rospy.is_shutdown():
            if (self.state == constants.ROBOT_MODE_TEACH_MANUALLY):
                if (self.latest_image == None):
                    print ("waiting for an image")
                    rate.sleep()
                    continue
                # if (teach_index > 10):
                #     self.state = constants.ROBOT_MODE_IDLE
                #     print ("teaching is done")
                #     continue

                current_state = ros_numpy.numpify(self.latest_image)

                # start debug
                # cv_image = current_state[...,::-1] # rgb to bgr
                # cv2.imshow("image", cv_image)
                # cv2.waitKey(10)
                # end debug

                rep, _ = self.sptm.append_keyframe(current_state, -1, False)
                self.goal = current_state
                time.sleep(constants.PIONEER_ACTION_STOP_DURATION)
                teach_index = teach_index+1
                print ("Teaching manually: index %d stored in memory" % (teach_index))

            elif (self.state == constants.ROBOT_MODE_TEACH):
                if (self.latest_image == None):
                    print ("waiting for an image")
                    rate.sleep()
                    continue
                if teach_index >= len(teach_actions):
                    self.state = constants.ROBOT_MODE_IDLE
                    continue
                """
                try:
                    cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
                    self.latest_image = None
                except CvBridgeError as e:
                    print(e)
                    rate.sleep()
                    continue

                current_state = np.asarray(cv_image)
                """
                current_state = ros_numpy.numpify(self.latest_image)

                # start debug
                cv_image = current_state[...,::-1] # rgb to bgr
                cv2.imshow("image", cv_image)
                cv2.waitKey(10)
                # end debug

                action = teach_actions[teach_index]
                rep, _ = self.sptm.append_keyframe(current_state, action, False)
                print ("commanded walk: index %d action %d" % (teach_index, action))
                self.take_action(action)
                self.goal = current_state
                teach_index = teach_index+1

            elif (self.state == constants.ROBOT_MODE_REPEAT):
                if (self.latest_image == None):
                    print ("waiting for an image")
                    rate.sleep()
                    continue

                self.hover()
                current_state = ros_numpy.numpify(self.latest_image)
                if (previous_state is None):
                    previous_state = current_state

                matched_index, similarity_score, best_velocity = self.sptm.relocalize(current_state)

                # start debug
                closest_match = self.sptm.memory[matched_index].state
                closest_match_message = ros_numpy.msgify(Image, closest_match, encoding='rgb8')
                self.closestMatchPublisher.publish(closest_match_message)
                # end debug

                path = self.sptm.find_shortest_path(matched_index, self.goal_index)
                print (matched_index, similarity_score, best_velocity, path)
                if (len(path) < 2): # achieved the goal
                    print ("Goal reached")
                    self.set_state(constants.ROBOT_MODE_IDLE)
                    continue

                if (constants.ACTION_LOOKAHEAD_ENABLED):
                    action, prob, future_state = self.path_lookahead(previous_state, current_state, path)
                else:
                    future_state = self.sptm.memory[path[1]].state
                    actions = self.navigation.forward(previous_state, current_state, future_state)
                    print (actions)
                    prob, pred = torch.max(actions.data, 1)
                    prob = prob.data.cpu().item()
                    action = pred.data.cpu().item()
                    print ("action %d" % action)

                # start debug
                future_match_message = ros_numpy.msgify(Image, future_state, encoding='rgb8')
                self.futureMatchPublisher.publish(future_match_message)
                # end debug

                # start debug
                # from PIL import Image
                # current_image = Image.fromarray(current_state)
                # future_image = Image.fromarray(future_state)
                # current_image.save("current.png", "PNG")
                # future_image.save("future.png", "PNG")
                # end debug

                self.take_action(action)
                previous_state = current_state
            else: # IDLE
                if (self.latest_image == None):
                    print ("waiting for an image")
                    rate.sleep()
                    continue
                current_state = ros_numpy.numpify(self.latest_image)
                matched_index, similarity_score, best_velocity = self.sptm.relocalize(current_state)
                print (matched_index, similarity_score, best_velocity)

                # start debug
                if (matched_index >= 0):
                    closest_match = self.sptm.memory[matched_index].state
                    closest_match_message = ros_numpy.msgify(Image, closest_match, encoding='rgb8')
                    self.closestMatchPublisher.publish(closest_match_message)
                # end debug

                # start debug
                # cv_image = self.sptm.memory[matched_index].state[...,::-1] # rgb to bgr
                # cv2.imshow("image", cv_image)
                # cv2.waitKey(10)
                # end debug

                time.sleep(constants.PIONEER_ACTION_STOP_DURATION)

            # self.latest_image = None
            rate.sleep()
