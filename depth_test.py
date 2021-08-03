# testing depth perception camera on realsense
import cv2
import pyrealsense2
from realsense_depth import *
print("cool")

# intialize camera intel realsense
dc = DepthCamera()

while True:
    ret, depth_frame, colour_frame = dc.get_frame()

    # show distnace
    point = (400, 300)
    cv2.circle(colour_frame, point, 4, (0, 0, 255))
    distance = depth_frame[point[1], point[0]]
    print(distance)

    cv2.imshow("colour", colour_frame)
    cv2.imshow("depth", depth_frame)
    cv2.waitKey(1)