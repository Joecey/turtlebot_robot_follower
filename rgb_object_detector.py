#!/usr/bin/env python3

# File to act as object detection and avoidance
# Using python and opencv
import rospy
import cv2
import pyttsx3
import pyrealsense2
import matplotlib.pyplot as plt
import numpy as np

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

# Video capturing
font_scale = 2
font = cv2.FONT_HERSHEY_PLAIN

cam = cv2.VideoCapture(0)

# set confidence value
confidence = 70

while True:
    check, colour_frame = cam.read()

    # Actual object detection aspect
    ClassIndex, confidence, bbox = model.detect(colour_frame, confThreshold=0.55)
    # print(ClassIndex)


    if (len(ClassIndex)!=0):
        for ClassInd, conf in zip(ClassIndex.flatten(), confidence.flatten()):

            # get only first box in bbox
            followBox = bbox[0]

            if(ClassInd<=len(classLabels)):
                if(classLabels[ClassInd-1] == 'cell phone'):
                    cv2.rectangle(colour_frame, followBox , (255,0,0), 2)
                    cv2.putText(colour_frame, classLabels[ClassInd-1], (followBox [0]+10,followBox [1]+40), font, fontScale = font_scale, color=(0,255,0))


    cv2.imshow("rgb_object_detector", colour_frame)
    cv2.waitKey(5) # delay in ms, might need to change this?

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cam.release()
cv2.destroyWindow()
