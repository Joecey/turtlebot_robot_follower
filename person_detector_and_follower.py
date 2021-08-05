#!/usr/bin/env python3

# File to act as person detector and follower (prototype atm)
# Using python and opencv
import cv2
import pyrealsense2
from realsense_depth import *
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2
from realsense_depth import *

# deep learning config file
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model, config_file)

model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

# create class labels
classLabels = [] # empty list
file_name = 'labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

print(classLabels)

# Video capturing
font_scale = 2
font = cv2.FONT_HERSHEY_PLAIN

#change this to the realsense camera
dc = DepthCamera()

# realsense
# dc = DepthCamera()

# Global command for robot
string_command = ""

while True:
    # cam is 640x480 px
    ret, depth_frame, colour_frame = dc.get_frame()

    # realsense
    # ret, depth_frame, frame = dc.get_frame()

    # Actual object detection aspect
    ClassIndex, confidence, bbox = model.detect(colour_frame, confThreshold=0.55)
    # print(ClassIndex)

    if (len(ClassIndex)!=0):
        for ClassInd, conf in zip(ClassIndex.flatten(), confidence.flatten()):

            # get only first box in bbox
            followBox = bbox[0]

            if(ClassInd<=len(classLabels)):
                if(classLabels[ClassInd-1] == 'person'):
                    cv2.rectangle(colour_frame, followBox , (255,0,0), 2)
                    cv2.putText(colour_frame, classLabels[ClassInd-1], (followBox [0]+10,followBox [1]+40), font, fontScale = font_scale, color=(0,255,0))

                    # print centre position of bounding box
                    centre_width = followBox [0] + round(followBox [2]/2)
                    # print(followBox)
                    # print(centre_width)
                    # draw circle for box
                    cv2.circle(colour_frame, (centre_width, 240), 3, (0, 255, 0), 3) # red circle in centre

                    box = followBox
                    point = (round((box[2] / 2) + box[0]), round((box[3] / 2) + box[1]))
                    # print(point)
                    cv2.circle(colour_frame, (round((box[2] / 2) + box[0]), round((box[3] / 2) + box[1])), 3, (0, 255, 255), 3)
                    # depth perception is measured at the yellow point
                    
                    distance = depth_frame[point[1], point[0]]  # when working with arrays, we put y coordinate before x coordinate
                    print(distance)



    # Draw centre circle on frame
    cv2.circle(colour_frame, (320, 240), 3, (0,0,255), 3)

    # run function to find difference between bbox centre and cam centre
    # cam is 640px wide
    # turn left if less than 320, turn right if greater than 340
    threshold_modifier = 40
    if centre_width <= 320 - threshold_modifier:
       # string_command = "rotating left..."
        pass
    elif centre_width >= 320 + threshold_modifier:
       # string_command = "rotating right..."
        pass
    else:
       # string_command = "moving forward..."
        pass

    # print string_command i.e. robot command
    print(string_command)

    cv2.imshow("RealSense Camera", colour_frame)
    cv2.waitKey(5) # delay in ms, might need to change this?

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#dc.release()
#cv2.destroyWindow()
