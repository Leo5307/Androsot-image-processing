import numpy as np
import time
import cv2
import cv2.aruco as aruco
import math
from Aruco_distance_estimation_multi_target import my_estimatePoseSingleMarkers,calc_Aruco_distance,rotation_mtx2euler_angle,get_Aruco_information

ARUCO_SIZE = 0.09#0.053
CAMERA_HEIGHT = 1.1
class Robot():
    def __init__(self,id,x,y,degree): 
        self.id = id
        self.x = x
        self.y = y
        self.degree = degree

class Robot_Team():
    def __init__(self,color):
        self.color = color
        self.Robot_list = []

    def get_information(self):
        print("Team:",self.color,"\n")
        for robot in self.Robot_list:
            print("robot:",robot.id," x:",robot.x," y:",robot.y," degree",robot.degree)

if __name__ == '__main__':
    calibration_matrix_path = "./calibration_matrix.npy"
    distortion_coefficients_path = "./distortion_coefficients.npy"
    camera_matrix = np.load(calibration_matrix_path)
    dist_matrix = np.load(distortion_coefficients_path)

    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)  # 设置帧率为60帧/秒
    font = cv2.FONT_HERSHEY_SIMPLEX #font for displaying text (below)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()
        h1, w1 = frame.shape[:2]
        #糾正畸變
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_matrix, (h1, w1), 1, (h1, w1))
        dst1 = cv2.undistort(frame, camera_matrix, dist_matrix, None, newcameramtx)
        x, y, w1, h1 = roi
        dst1 = dst1[y:y + h1, x:x + w1]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        parameters =  cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        #使用aruco.detectMarkers()函數可以檢測到marker，返回ID和标志板的4个角点坐標
        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
        frame,id_list,distance_list,distance_x_list,degree_list = get_Aruco_information(frame=frame,corners=corners,ids=ids)
        red_team = Robot_Team(color='red')
        blue_team = Robot_Team(color='blue')

        for index,id in enumerate(id_list):
            if id % 2 == 0:
                #even,blue team
                blue_team.Robot_list.append(Robot(id=id,x=distance_x_list[index],y=distance_list[index],degree=degree_list[index]))

            else:
                 #odd,blue team
                red_team.Robot_list.append(Robot(id=id,x=distance_x_list[index],y=distance_list[index],degree=degree_list[index]))

        red_team.get_information()
        blue_team.get_information()
        


        # frame_resized = cv2.resize(frame, (0, 0), fx=1000, fy=800)
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 1200, 800)
        cv2.imshow("frame",frame)

        key = cv2.waitKey(1)

        if key == 27:
            print('esc break...')
            cap.release()
            cv2.destroyAllWindows()
            break

        if key == ord(' '):
            filename = str(time.time())[:10] + ".jpg"
            cv2.imwrite(filename, frame)