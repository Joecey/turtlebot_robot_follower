# testing depth perception camera on realsense
import cv2
import pyrealsense2
from realsense_depth import *
print("epic")

# intialize camera intel realsense
dc = DepthCamera()

while True:
    ret, depth_frame, color_frame = dc.get_frame()

    # show distnace
    point = (400, 300)
    cv2.circle(color_frame, point, 4, (0, 0, 255))
    distance = depth_frame[point[1], point[0]]
    print(distance)

    cv2.imshow("colour", color_frame)
    cv2.imshow("depth", depth_frame)
    cv2.waitKey(1)