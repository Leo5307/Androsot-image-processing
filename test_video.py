import cv2
from cv2 import aruco
import numpy as np
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap = cv2.VideoCapture('2024-03-25_17-24-36.mp4') # 讀取電腦中的影片
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
 print("Cannot open camera")
 exit()
while True:
 # Capture frame-by-frame
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True 
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    points_path = "./points.npy"
    points = np.load(points_path)
    p1 = np.float32(points)
    p2 = np.float32([[0,0],[400,0],[0,800],[400,800]])
    m = cv2.getPerspectiveTransform(p1,p2)
    # frame = cv2.warpPerspective(frame, m, (400, 800))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # mask = cv2.inRange(gray,0, 75)
    # gray[mask != 0] = 255
    mask = cv2.inRange(gray,55, 140)
    gray[mask != 0] = 0
    mask = cv2.inRange(gray,190, 230)
    gray[mask != 0] = 200

    mask = cv2.inRange(gray,230, 255)
    gray[mask != 0] = 0
    # gray = cv2.equalizeHist(gray)


    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    frame_markers = aruco.drawDetectedMarkers(gray.copy(), corners, ids,borderColor=(0, 255, 0))

    # Display the result
    cv2.imshow("Frame Markers", frame_markers)

    if cv2.waitKey(1) == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()