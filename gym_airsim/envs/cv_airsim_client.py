import numpy as np
import time
import math
import random
import cv2
from pylab import array, arange, uint8 
from PIL import Image

from AirSimClient import *
import constants

class CVAirSimClient(MultirotorClient):
    def __init__(self):        
        MultirotorClient.__init__(self)
        MultirotorClient.confirmConnection(self)

#        AirSimClientBase.wait_key('Press any key to set camera-0 gimble to 15-degree pitch')
#        self.setCameraOrientation(0, AirSimClientBase.toQuaternion(0, 0, 0)); #radians

        self.mode = constants.AIRSIM_MODE_DATA_COLLECTION
        self.pose = [0, 0, -6]
        self.orientation = [0, 0, 0]
        self.update_pose()

    def take_action(self, action):
        if action == -1: # backward
            if (self.mode == constants.AIRSIM_MODE_DATA_COLLECTION):
                speed = random.uniform(constants.DATA_COLLECTION_MIN_SPEED, constants.DATA_COLLECTION_MAX_SPEED)
            elif (self.mode == constants.AIRSIM_MODE_TEACH):
                speed = random.uniform(constants.DATA_COLLECTION_MIN_SPEED, constants.DATA_COLLECTION_MAX_SPEED)
            elif (self.mode == constants.AIRSIM_MODE_REPEAT):
                speed = constants.DATA_COLLECTION_MIN_SPEED

            vx = math.cos(self.orientation[2]) * speed
            vy = math.sin(self.orientation[2]) * speed
 
            self.pose[0] -= vx
            self.pose[1] -= vy
        elif action == 0:
            if (self.mode == constants.AIRSIM_MODE_DATA_COLLECTION):
                speed = random.uniform(constants.DATA_COLLECTION_MIN_SPEED, constants.DATA_COLLECTION_MAX_SPEED)
            elif (self.mode == constants.AIRSIM_MODE_TEACH):
                speed = random.uniform(constants.DATA_COLLECTION_MIN_SPEED, constants.DATA_COLLECTION_MAX_SPEED)
            elif (self.mode == constants.AIRSIM_MODE_REPEAT):
                speed = constants.DATA_COLLECTION_MIN_SPEED

            vx = math.cos(self.orientation[2]) * speed
            vy = math.sin(self.orientation[2]) * speed
 
            self.pose[0] += vx
            self.pose[1] += vy
        elif action == 1:
            if (self.mode == constants.AIRSIM_MODE_DATA_COLLECTION):
                angle = random.uniform(constants.DATA_COLLECTION_MIN_ANGLE, constants.DATA_COLLECTION_MAX_ANGLE)
            elif (self.mode == constants.AIRSIM_MODE_TEACH):
                angle = random.uniform(constants.DATA_COLLECTION_MIN_ANGLE, constants.DATA_COLLECTION_MAX_ANGLE)
            elif (self.mode == constants.AIRSIM_MODE_REPEAT):
                # angle = (constants.DATA_COLLECTION_MAX_ANGLE + constants.DATA_COLLECTION_MIN_ANGLE) / 2.
                angle = constants.DATA_COLLECTION_MIN_ANGLE

            self.orientation[2] += angle
        elif action == 2:
            if (self.mode == constants.AIRSIM_MODE_DATA_COLLECTION):
                angle = random.uniform(constants.DATA_COLLECTION_MIN_ANGLE, constants.DATA_COLLECTION_MAX_ANGLE)
            elif (self.mode == constants.AIRSIM_MODE_TEACH):
                angle = random.uniform(constants.DATA_COLLECTION_MIN_ANGLE, constants.DATA_COLLECTION_MAX_ANGLE)
            elif (self.mode == constants.AIRSIM_MODE_REPEAT):
                # angle = (constants.DATA_COLLECTION_MAX_ANGLE + constants.DATA_COLLECTION_MIN_ANGLE) / 2.
                angle = constants.DATA_COLLECTION_MIN_ANGLE

            self.orientation[2] -= angle 
        else:
            print ("wrong action: %d" % action)

        self.update_pose()

        time.sleep(0.5)
        # collision = self.getCollisionInfo()
        # return collision.has_collided
        return False

    def set_mode(self, mode):
        self.mode = mode

    def getImage(self):
        responses = self.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
        image = img1d.reshape(response.height, response.width, 4)  
        image = image[:, :, :3]

        factor = 0.5
        maxIntensity = 255.0 # depends on dtype of image data

        # Decrease intensity such that dark pixels become much darker, bright pixels become slightly dark 
        image = (maxIntensity)*(image/maxIntensity)**factor
        image = array(image, dtype=uint8)

        # from PIL import Image
        # image = Image.fromarray(image)
        # image.show()
        # image.save("img1.png","PNG")

        return image

    def getImageOld(self):
        responses = self.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
        image = img1d.reshape(response.height, response.width, 4)  

        factor = 0.5
        maxIntensity = 255.0 # depends on dtype of image data

        # Decrease intensity such that dark pixels become much darker, bright pixels become slightly dark 
        image = (maxIntensity)*(image/maxIntensity)**factor
        image = array(image, dtype=uint8)

        # cv2.imshow("Test", image)
        # cv2.waitKey(30)

        image = np.flipud(image)

        return image

    def getScreenDepthVis(self, track):
        responses = self.simGetImages([ImageRequest(0, AirSimImageType.DepthPerspective, True, False)])
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        image = np.invert(np.array(Image.fromarray(img2d.astype(np.uint8), mode='L')))

        factor = 10
        maxIntensity = 255.0 # depends on dtype of image data

        # Decrease intensity such that dark pixels become much darker, bright pixels become slightly dark 
        newImage1 = (maxIntensity)*(image/maxIntensity)**factor
        newImage1 = array(newImage1,dtype=uint8)

        small = cv2.resize(newImage1, (0,0), fx=0.39, fy=0.38)

        cut = small[20:40,:]

        info_section = np.zeros((10,cut.shape[1]),dtype=np.uint8) + 255
        info_section[9,:] = 0

        line = np.int((((track - -180) * (100 - 0)) / (180 - -180)) + 0)

        if line != (0 or 100):
            info_section[:,line-1:line+2]  = 0
        elif line == 0:
            info_section[:,0:3]  = 0
        elif line == 100:
            info_section[:,info_section.shape[1]-3:info_section.shape[1]]  = 0
        total = np.concatenate((info_section, cut), axis=0)

        # cv2.imshow("Test", total)
        # cv2.waitKey(30)

        total = total.reshape(total.shape[0], total.shape[1], -1)

        return total

    def set_height(self, z):
        self.z = -1 * z

    def set_pose(self, pose, orientation):
        self.pose = pose
        self.orientation = orientation
        self.update_pose()

    def update_pose(self):
        self.simSetPose(Pose(Vector3r(*self.pose), AirSimClientBase.toQuaternion(*self.orientation)), True)

    def _reset(self):
        return True
