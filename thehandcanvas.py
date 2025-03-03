import cv2
import mediapipe as mp
from screeninfo import get_monitors

for monitor in get_monitors():
    deviceWidth = monitor.width
    deviceHeight = monitor.height

# setting up the target camera
camera = cv2.VideoCapture(0)

# setting up hand tracking solution from mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence = 0.5)

# setting up the drawing utility from mediapipe
mpDraw = mp.solutions.drawing_utils

while True:
    # capturing the frame
    success, frame = camera.read()
    frame = cv2.flip(frame, 1)

    canvas = cv2.imread('image.png')

	# converting the frame into RGB
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# getting coordinates of palm landmarks
    palm_data = hands.process(frameRGB)
    multi_hand_landmarks = palm_data.multi_hand_landmarks

	# drawing landmark points on the palm
    if multi_hand_landmarks != None:
        for single_hand_landmarks in multi_hand_landmarks:
            mpDraw.draw_landmarks(canvas, single_hand_landmarks, mpHands.HAND_CONNECTIONS)
    
    cv2.namedWindow("Canvas", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Canvas", deviceWidth, deviceHeight)

    # displaying every frame one by one
    cv2.imshow("Canvas", canvas)
    # cv2.imshow("Frame", frame)
    cv2.waitKey(1)  