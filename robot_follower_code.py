#!/usr/bin/env python3


# A very basic TurtleBot script that moves TurtleBot forward indefinitely. Press CTRL + C to stop.  To run:
# On TurtleBot:
# roslaunch turtlebot_bringup minimal.launch
# On work station:
# python goforward.py

import rospy
from geometry_msgs.msg import Twist
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2
from realsense_depth import *




#### Intialization for rospy ####
rospy.init_node('rotate', anonymous=False)
print("node made")
# tell user how to stop TurtleBot
rospy.loginfo("To stop TurtleBot CTRL + C")



# Create a publisher which can "talk" to TurtleBot and tell it to move
# Tip: You may need to change cmd_vel_mux/input/navi to /cmd_vel if you're not using TurtleBot2
cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)

# TurtleBot will stop if we don't keep telling it to move.  How often should we tell it to move? 10 HZ
# 0.1 second = 10 hz
r = rospy.Rate(5);

# Twist is a datatype for velocity
move_cmd_right = Twist()
move_cmd_right.linear.x = 0.0
move_cmd_right.angular.z = -0.3

# turn left
move_cmd_left = Twist()
move_cmd_left.linear.x = 0.0
move_cmd_left.angular.z = 0.3
#
# move stop
move_cmd_stop = Twist()
move_cmd_stop.linear.x = 0
move_cmd_stop.angular.z = 0

# forward
move_cmd_forward = Twist()
move_cmd_forward.linear.x = 0.1
move_cmd_forward.angular.z = 0

###  person detector initialization ###
# deep learning config file
# USE ABSOLUTE PATH HERE WHEN CREATE A LAUNCH FILE
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model, config_file)

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# create class lables
classLabels = []  # empty list
# USE ABSOLUTE PATH HERE 
file_name = 'labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

print(classLabels)

# Video capturing
font_scale = 2
font = cv2.FONT_HERSHEY_PLAIN

# webcam capture
# cam = cv2.VideoCapture(0)

# change this to the realsense camera
dc = DepthCamera()

# Global command for robot
string_command = ""

# as long as you haven't ctrl + c keeping doing...
while not rospy.is_shutdown():
    # cam is 640x480 px
    ret, depth_frame, colour_frame = dc.get_frame()


    # Actual object detection aspect
    ClassIndex, confidence, bbox = model.detect(colour_frame, confThreshold=0.55)
    # print(ClassIndex)

    if (len(ClassIndex) != 0):
        for ClassInd, conf in zip(ClassIndex.flatten(), confidence.flatten()):

            # get only first box in bbox
            followBox = bbox[0]

            if (ClassInd <= len(classLabels)):
                if (classLabels[ClassInd - 1] == 'person'):
                    cv2.rectangle(colour_frame, followBox, (255, 0, 0), 2)
                    cv2.putText(colour_frame, classLabels[ClassInd - 1], (followBox[0] + 10, followBox[1] + 40), font,
                                fontScale=font_scale, color=(0, 255, 0))

                    # print centre position of bounding box
                    centre_width = followBox[0] + round(followBox[2] / 2)

                    # draw circle for box
                    cv2.circle(colour_frame, (centre_width, 240), 3, (0, 255, 0), 3)
                    box = followBox
                    point = (round((box[2] / 2) + box[0]), round((box[3] / 2) + box[1]))
                    # print(point)
                    cv2.circle(colour_frame, (round((box[2] / 2) + box[0]), round((box[3] / 2) + box[1])), 3,
                               (0, 255, 255), 3)
                    # depth perception is measured at the yellow point

                    distance = depth_frame[point[1], point[0]]  # when working with arrays, we put y coordinate before x coordinate
                    print(distance)

                    # run function to find difference between bbox centre and cam centre
                    # cam is 640px wide
                    # turn left if less than 320, turn right if greater than 340
                    threshold_modifier = 80
                    distance_thres = 600
                    if distance <= distance_thres:
                        string_command = "stop..."
                        cmd_vel.publish(move_cmd_stop)
                        r.sleep()

                    elif distance > distance_thres:
                        if centre_width <= 320 - threshold_modifier:
                            string_command = "rotating left..."
                            cmd_vel.publish(move_cmd_left)
                            r.sleep()

                        elif centre_width >= 320 + threshold_modifier:
                            string_command = "rotating right..."
                            cmd_vel.publish(move_cmd_right)
                            r.sleep()

                        else:
                            string_command = "forward"
                            cmd_vel.publish(move_cmd_forward)
                            r.sleep()

    # Draw centre circle on frame
    cv2.circle(colour_frame, (320, 240), 3, (0, 0, 255), 3)

    print(string_command)
    cv2.imshow("Webcam", colour_frame)
    cv2.waitKey(10)  # delay in ms, might need to change this?




