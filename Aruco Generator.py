import cv2
import cv2.aruco as aruco
import numpy as np

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
marker_image = aruco.generateImageMarker(aruco_dict, 2, 700)  # assume size of aruco is 700 pixels

#Gray 2 BGR
marker_color = cv2.cvtColor(marker_image, cv2.COLOR_GRAY2BGR)

#Blue (B=255, G=0, R=0)
marker_color[marker_image == 0] = [0, 0, 255]  # BGR格式

cv2.imshow('Blue-White Aruco Marker', marker_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('1.png', marker_color)
