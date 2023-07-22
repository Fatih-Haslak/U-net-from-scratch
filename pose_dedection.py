
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose 
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) 

cap = cv2.VideoCapture(0)
if cap.isOpened() == False:
    print("Error opening video stream or file")
    raise TypeError
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)         
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow("deneme",image)
    cv2.waitKey(1)
pose.close()
cap.release()
