# This code is a testing ground for mediapipe implementation
# in conjuction with OpenCV

# install mediapipe with !pip install mediapipe opencv-python

# import rospy if needed to run in ros (however, I don't think it will be needed?)

import cv2 as cv    # I don't know why I defined cv2 as cv, probably due to brain damage - Joe
import mediapipe as mp
import numpy as np
print("Package Imported")

# Load mediapipe model
mp_drawing = mp.solutions.drawing_utils # draw detections from holistic model
mp_hands = mp.solutions.hands # import holistic model

# setup webcam with detection model
cap = cv.VideoCapture(0)

# define width and height
cap.set(3, 640) # width id
cap.set(4, 480) # height id
cap.set(10, 100) # brightness id

# stack images function
# You can copy this function to stack multiple image sources in the one window - Joe
def stackImages(scale,imgArray):

    rows = len(imgArray)

    cols = len(imgArray[0])

    rowsAvailable = isinstance(imgArray[0], list)

    width = imgArray[0][0].shape[1]

    height = imgArray[0][0].shape[0]

    if rowsAvailable:

        for x in range ( 0, rows):

            for y in range(0, cols):

                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:

                    imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)

                else:

                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)

                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv.cvtColor( imgArray[x][y], cv.COLOR_GRAY2BGR)

        imageBlank = np.zeros((height, width, 3), np.uint8)

        hor = [imageBlank]*rows

        hor_con = [imageBlank]*rows

        for x in range(0, rows):

            hor[x] = np.hstack(imgArray[x])

        ver = np.vstack(hor)

    else:

        for x in range(0, rows):

            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:

                imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)

            else:

                imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)

            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)


        hor= np.hstack(imgArray)

        ver = hor

    return ver

# grab information of right or left hand
# Dunno how useful this will be, but it's there

# Here's some videos I found useful as a starting point/to understand the code
# https://www.youtube.com/watch?v=EgjwKM3KzGU
# https://www.youtube.com/watch?v=vQZ4IvB07ec&t=1827s

def get_label(index, hand, results):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            # process results
            label = classification.classification[0].index
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score,2))

            # extract coordinates
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
                [640,480]).astype(int))

            output = text,coords

    return output

# implement hand model
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        success, vid = cap.read()      # success is bool

        # create copy of video to display
        raw = vid.copy()

        # Recolour feed to RGB
        image = cv.cvtColor(vid, cv.COLOR_BGR2RGB)

        # Make detections
        results = hands.process(image)

        # show various landmarks
        #print(results.face_landmarks)
        #print(results.pose_landmarks)
        # print(results.multi_hand_landmarks)


        # Draw face landmarks
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        # Draw hand landmarks, rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),)

            # print specific landmarks (follow this link for labels: https://user-images.githubusercontent.com/37477845/102242918-ed328c80-3f3d-11eb-907c-61ba05678d54.png)
            # Ctrl + click to open link, see example below

            # get the wrist (type the landmark name) of the first hand listed
            # print(results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST])

            # extract "percentage" coordinates of landmarks, then convert them to pixel coordinates
            # relative to the shape of our image (640 x 480)
            # IFT = Index finger tip; TT = Thumb Tip
            coordsIFT = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                          hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y)),
                [640, 480]).astype(int))

            coordsTT = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                          hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y)),
                [640, 480]).astype(int))


            cv.circle(image, coordsIFT, 10, (255,255,255), thickness=2)
            cv.circle(image, coordsTT, 10, (255, 255, 255), thickness=2)

            # draw rectangle to join the two points
            cv.line(image, coordsIFT, coordsTT, (255,255,255), thickness=2)

            # calculate distance between index and thumb and display as percentage
            # distance between two points
            minDist = 10
            maxDist = 170

            # using distance between two points formula
            temp = ((coordsTT[0] - coordsIFT[0]) ** 2) + ((coordsTT[1] - coordsIFT[1]) ** 2)
            distance = np.sqrt(temp)

            # calculate as percentage
            if distance > maxDist:
                distPer = 100.0

            elif distance < minDist :
                distPer = 0.0

            else:
                distPer = round((((distance - minDist)/ maxDist) * 100), 1)

            # display result next to thumb
            cv.putText(image, (str(distPer) + "%"), coordsTT, cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 1,
                       color=(255,255,255), thickness=2)

        # raw feed
        # cv.imshow("Webcam Feed", vid)

        # render drawn image
        cv.imshow("Detections", image)

        # Display raw video and holistic model side by side (using stack function)
        # imgStack = stackImages(0.9, ([raw, image]))
        # cv.imshow("Raw and Detections", imgStack)

        # adds delay and waits for key press q to break loop
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
