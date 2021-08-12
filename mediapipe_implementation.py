# This code is a testing ground for mediapipe implementation
# in conjuction with OpenCV

# install mediapipe with !pip install mediapipe opencv-python

import cv2 as cv
import mediapipe as mp
import numpy as np
print("Package Imported")

# Load mediapipe model
mp_drawing = mp.solutions.drawing_utils # draw detections from holistic model
mp_holistic = mp.solutions.holistic # import holistic model

# setup webcam with detection model
cap = cv.VideoCapture(0)

# define width and height
cap.set(3, 640) # width id
cap.set(4, 480) # height id
cap.set(10, 100) # brightness id

# implement holistic model

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        success, vid = cap.read()      # success is bool

        # Recolour feed to RGB
        image = cv.cvtColor(vid, cv.COLOR_BGR2RGB)

        # Make detections
        results = holistic.process(image)

        # show various landmarks
        #print(results.face_landmarks)
        print(results.right_hand_landmarks)

        # EXAMPLE, face, pose, left_hand, right_hand

        # Draw face landmarks
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        # Draw face landmarks
        
        # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)

        # Draw right hand landmarks
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # # Draw left hand landmarks
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # # Draw pose landmarks
        # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # raw feed
        # cv.imshow("Webcam Feed", vid)

        # render drawn image
        cv.imshow("Detections", image)

        # adds delay and waits for key press q to break loop
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
