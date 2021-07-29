# File to act as person detector and follower (prototype atm)
# Using python and opencv
import cv2
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

# create class lables
classLabels = [] # empty list
file_name = 'Labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

print(classLabels)

# Video capturing
font_scale = 2
font = cv2.FONT_HERSHEY_PLAIN

cam = cv2.VideoCapture(0)

while True:
    # cam is 640x480 px
    check, frame = cam.read()


    # Actual object detection aspect
    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)
    # print(ClassIndex)

    if (len(ClassIndex)!=0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if(ClassInd<=len(classLabels)):
                if(classLabels[ClassInd-1] == 'person'):
                    cv2.rectangle(frame, boxes, (255,0,0), 2)
                    cv2.putText(frame, classLabels[ClassInd-1], (boxes[0]+10,boxes[1]+40), font, fontScale = font_scale, color=(0,255,0))

                    # print centre position of bounding box
                    centre_width = boxes[0] + round(boxes[2]/2)
                    # print(boxes)
                    # print(centre_width)
                    # testing branch changes

                    # draw circle for box
                    cv2.circle(frame, (centre_width, 240), 3, (0, 255, 0), 3)


    # Draw centre circle on frame
    cv2.circle(frame, (320, 240), 3, (0,0,255), 3)


    cv2.imshow("Webcam", frame)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyWindow()
