import numpy as np
import time
import cv2
import cv2.aruco as aruco
import math
from Robot_pos_estimator import my_estimatePoseSingleMarkers,calc_Aruco_distance,rotation_mtx2euler_angle,get_Aruco_information
from color_masking import colorMasking
ARUCO_SIZE = 0.08#0.053
CAMERA_HEIGHT = 1.5
CAMERA_DISTANCE = 1
FIELD_LENGTH = 3.6
FIELD_WIDTH = 2.6

class Robot():
    def __init__(self,id,aruco_id_list,aruco_x_list,aruco_y_list,aruco_degree_list):
        self.id = id 
        self.aruco_id_list = aruco_id_list
        self.aruco_x_list = aruco_x_list
        self.aruco_y_list = aruco_y_list
        self.aruco_degree_list = aruco_degree_list
        self.x = 0
        self.y = 0
        self.degree = 0
    
    def robot_pose_estimation(self):
        for index,id in enumerate(self.aruco_id_list):
            if(id == 1 or id == 0):#,3,5,7,9,clockwise
                self.x = FIELD_LENGTH* 100 - (self.aruco_x_list[index] - CAMERA_DISTANCE*100)
                self.y = (FIELD_WIDTH* 100/2 - self.aruco_y_list[index])
                self.degree = self.aruco_degree_list[index]

    def update_robot_position(self,id,aruco_id_list,aruco_x_list,aruco_y_list,aruco_degree_list):
        self.id = id
        self.aruco_id_list = aruco_id_list
        self.aruco_x_list = aruco_x_list
        self.aruco_y_list = aruco_y_list
        self.aruco_degree_list = aruco_degree_list
        self.robot_pose_estimation()

class Robot_Team():
    def __init__(self,color,length):
        self.color = color
        self.team_length = length
        self.Robot_list = []
        for i in range(length):
            aruco_id_list,aruco_x_list,aruco_y_list,aruco_degree_list = [],[],[],[]
            self.Robot_list.append(Robot(i,aruco_id_list,aruco_x_list,aruco_y_list,aruco_degree_list))

    def update_position(self,aruco_id_list,aruco_x_list,aruco_y_list,aruco_degree_list):
        for robot_id in range(self.team_length):
            slice_aruco_id_list,slice_aruco_x_list,slice_aruco_y_list,slice_aruco_degree_list = [],[],[],[]
            for index,aruco_id in enumerate(aruco_id_list):
                if aruco_id >= 10*(robot_id) and aruco_id < 10*(robot_id + 1):
                    slice_aruco_id_list.append(aruco_id_list[index])
                    slice_aruco_x_list.append(aruco_x_list[index])
                    slice_aruco_y_list.append(aruco_y_list[index])
                    slice_aruco_degree_list.append(aruco_degree_list[index])

                self.Robot_list[robot_id].update_robot_position(robot_id,slice_aruco_id_list,slice_aruco_x_list,slice_aruco_y_list,slice_aruco_degree_list)
    def get_information(self):
        print("Team:",self.color," num:",len(self.Robot_list),"\n")
        for robot in self.Robot_list:
            print("robot:",robot.id," x:",robot.x," y:",robot.y," degree",robot.degree)


class Ball():
    def __init__(self):
        self.x = 0
        self.y = 0

    def update_position(self):
        mask = colorMasking(frame,"yellow")
        mask = cv2.medianBlur(mask, 5)  # 中值濾波
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 抓出輪廓
        datacenter = []  # 儲存球心座標
        dataradius = []  # 儲存半徑座標
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)  # 取出輪廓外接的最小正矩形座標
            if w * h > 100 and w * h < 2000 and w < 5 * h and h < 5 * w:  # 過濾面積小於150的矩形
                (xx, yy), radius = cv2.minEnclosingCircle(cnt)  # 找出最小外接圓
                ball_x = int(xx)
                ball_y = int(yy)
                center = [ball_x,ball_y]
                radius = int(radius)
                
                if radius > 20:
                    cv2.circle(frame, center, radius, (0,255,0), 3)  # 畫圓
                    cv2.putText(frame, "Ball", (x, y - 5), font, 0.7, (0,255,0), 2)  # 螢幕上寫出"Ball"
                    datacenter.append(center)  # 儲存球心座標
                    dataradius.append(radius)  # 儲存半徑座標
                    
                    # print(center)
        #只取第一個
        if (datacenter):#這裡可以做一個要求只在范圍內的過濾
            ballcenter = datacenter[0]
            print(ballcenter)
            transform_ball_x = int((ballcenter[0] / 1920) * FIELD_LENGTH * 100)
            transform_ball_y = int((ballcenter[1] / 1080) * FIELD_WIDTH * 100)
            [self.x,self.y] = [transform_ball_x,transform_ball_y]
   
            
def gamma_correction(frame, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
    	# apply gamma correction using the lookup table
	return cv2.LUT(frame, table)
 
if __name__ == '__main__':
    calibration_matrix_path = "./calibration_matrix.npy"
    distortion_coefficients_path = "./distortion_coefficients.npy"
    camera_matrix = np.load(calibration_matrix_path)
    dist_matrix = np.load(distortion_coefficients_path)

    cap = cv2.VideoCapture(1,cv2.CAP_DSHOW) 
    cap = cv2.VideoCapture('2024-03-27_19-19-03.mp4') # 讀取電腦中的影片

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)  # 设置帧率为60帧/秒
    font = cv2.FONT_HERSHEY_SIMPLEX #font for displaying text (below)
    red_team = Robot_Team(color='red',length = 2)
    blue_team = Robot_Team(color='blue',length = 2)
    ball = Ball()

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
        # frame = dst1
        points_path = "./points.npy"
        points = np.load(points_path)
        p1 = np.float32(points)
        p2 = np.float32([[0,0],[1080,0],[0,1920],[1080,1920]])
        m = cv2.getPerspectiveTransform(p1,p2)
        # frame = cv2.warpPerspective(frame, m, (1080, 1920))
        # frame = gamma_correction(frame,0.2)
        frame = gamma_correction(frame, gamma=0.5)
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
        parameters =  cv2.aruco.DetectorParameters()
        parameters.perspectiveRemoveIgnoredMarginPerCell = 0.3
        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizeMax = 60
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        #使用aruco.detectMarkers()函數可以檢測到marker，返回ID和标志板的4个角点坐標
        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
        frame,id_list,distance_list,distance_x_list,degree_list,degree_list_x,degree_list_y,degree_list_z\
              = get_Aruco_information(camera_matrix=camera_matrix,dist_matrix=dist_matrix,frame=frame,corners=corners,ids=ids)

        red_id_list,blue_id_list,red_x_list,blue_x_list,red_y_list,blue_y_list,red_degree_list,blue_degree_list = [],[],[],[],[],[],[],[]
        for index,id in enumerate(id_list):
            if id % 2 == 0:
                #even,blue team
                blue_id_list.append(id)
                blue_x_list.append(distance_list[index])
                blue_y_list.append(distance_x_list[index])
                blue_degree_list.append(degree_list[index])

            else:
                red_id_list.append(id)
                red_x_list.append(distance_list[index])
                red_y_list.append(distance_x_list[index])
                red_degree_list.append(degree_list[index])

        blue_team.update_position(blue_id_list,blue_x_list,blue_y_list,blue_degree_list)
        red_team.update_position(red_id_list,red_x_list,red_y_list,red_degree_list)

        # red_team.get_information()
        blue_team.get_information()

        # frame_resized = cv2.resize(frame, (0, 0), fx=1000, fy=800)
        # frame = cv2.warpPerspective(frame, m, (1080, 1920))
        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ball.update_position()
        cv2.putText(frame,'coordinates:'+ str([blue_team.Robot_list[0].x,blue_team.Robot_list[0].y]) ,(0, 110), font, 1, (0, 255, 0), 2,cv2.LINE_AA)
        cv2.putText(frame,'deg_z:'+ str(blue_team.Robot_list[0].degree) ,(0, 150), font, 1, (0, 255, 0), 2,cv2.LINE_AA)
        cv2.putText(frame,'ballcenter:'+ str([ball.x,ball.y]) ,(0, 190), font, 1, (0, 255, 0), 2,cv2.LINE_AA)
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