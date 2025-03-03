import cv2
from handtracking_module import HandTrace

hands = HandTrace()

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()
    frame = cv2.flip(frame, 1)
    canvas = cv2.imread('image.png')

    hands.drawHandConnections(detection_frame=frame, display_frame=canvas)

    cv2.imshow('Canvas', canvas)
    cv2.waitKey(1)