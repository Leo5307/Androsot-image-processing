import cv2
import numpy as np
import time
import cv2.aruco as aruco
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)  # 60fps

while True:
 # Capture frame-by-frame
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True 
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

    frame_markers = frame.copy()
    marker_corners = np.int32(corners)
    for i in range(marker_corners.shape[0]):
        each_marker_corners = marker_corners[i]
        each_marker_corners = each_marker_corners[0]
        print(each_marker_corners.shape)
        print("start")
        box = np.int32(each_marker_corners)
        print(box)
        min_x = np.min(box[:, 0])
        max_x = np.max(box[:, 0])
        min_y = np.min(box[:, 1])
        max_y = np.max(box[:, 1])

        #左上角和右下角坐標
        x1,y1 = (min_x, min_y)
        x2,y2 = (max_x, max_y)

        cropped_frame = frame[x1:x2,y1:y2]
        cv2.rectangle(frame_markers, (x1,y1), (x2,y2), (0, 255, 0), 2)
        print("start detect color")
        if cropped_frame.size == 0:
            continue

        hsv_image = cv2.cvtColor( cropped_frame, cv2.COLOR_BGR2HSV)
        #定義範圍
        red_lower_1= np.array([0, 50, 50])
        red_upper_1 = np.array([10, 255, 255])
        red_lower_2 = np.array([150, 50, 50])
        red_upper_2 = np.array([180, 255, 255])
        green_lower = np.array([50, 50, 50])
        green_upper = np.array([70, 255, 255])
        blue_lower = np.array([110, 50, 50])
        blue_upper = np.array([130, 255, 255])

        mask_red = cv2.inRange(hsv_image, red_lower_1, red_upper_1) +cv2.inRange(hsv_image, red_lower_2, red_upper_2)
        mask_green = cv2.inRange(hsv_image, green_lower, green_upper)
        mask_blue = cv2.inRange(hsv_image, blue_lower, blue_upper)   

        num_red = np.sum(mask_red)
        num_green = np.sum(mask_green)
        num_blue = np.sum(mask_blue)#會有 三個都是0的情況 不知道為什麼 可能範圍沒選對

        #找出最常見的顏色
        if num_blue == num_green == num_red:
            print("No color")
            continue
        else:
            max_color = np.argmax(np.array([num_blue, num_green, num_red]))
            print(num_blue, num_green, num_red)
            colors = {0:'Blue',1:'Green',2:'Red'}
            print("Detect color", colors[max_color])
        cv2.rectangle(frame_markers, (x1,y1), (x2,y2), (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Frame Markers", frame_markers)

    if cv2.waitKey(1) == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

