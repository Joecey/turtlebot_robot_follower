# testing depth perception camera on realsense
import cv2
import pyrealsense2
from realsense_depth import *
from person_detector_and_follower import *

print("let's see if this works woo")

# intialize camera intel realsense
dc = DepthCamera()

while True:
    ret, depth_frame, colour_frame = dc.get_frame()

    # show distance
    # point = (400, 300)
    box = followBox
    point = ((box[2]/2)+box[0], (box[3]/2)+box[1])
    print(point)
    #cv2.circle(colour_frame, box, 4, (0, 0, 255))
    #should actually print out the box instead to show where we're measuring
    distance = depth_frame[point[1], point[0]]  #when working with arrays, we put y coordinate before x coordinate
    print(distance)

    cv2.imshow("colour", colour_frame)
    cv2.imshow("depth", depth_frame)
    cv2.waitKey(1)