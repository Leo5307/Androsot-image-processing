import cv2
import numpy as np
from cv2 import aruco


points_path = "./points.npy"

points = np.load(points_path)
p1 = np.float32(points)
p2 = np.float32([[0,0],[400,0],[0,800],[400,800]])
m = cv2.getPerspectiveTransform(p1,p2)

frame = cv2.imread('2024-03-25_17-24-09.jpg')

frame = cv2.warpPerspective(frame, m, (400, 800))
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
aruco_dict = cv2.aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids,borderColor=(0, 255, 0))

# Display the result
cv2.imshow("Frame Markers", frame_markers)
# cv2.imshow('oxxostudio', output)
cv2.waitKey(0)