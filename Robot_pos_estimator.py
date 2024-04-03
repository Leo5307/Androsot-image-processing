# ref:https://github.com/ZengWenJian123/aruco_positioning_2D
import numpy as np
import time
import cv2
import cv2.aruco as aruco
import math
ARUCO_SIZE = 0.078#0.053
CAMERA_HEIGHT = 1.5
#ref:https://stackoverflow.com/questions/76802576/how-to-estimate-pose-of-single-marker-in-opencv-python-4-8-0
def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    
    for c in corners:
        #c 是image pt marker pt是 object pt
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R.flatten())
        tvecs.append(t.flatten())
        # print("t:",tvecs)
        # print(tvecs[1])
        #nada 通常是一個布林值，表示求解过程是否成功
        trash.append(nada)
    rvecs = np.array(rvecs)
    rvecs = rvecs.reshape(-1,1,3)
    tvecs = np.array(tvecs)
    tvecs = tvecs.reshape(-1,1,3)
    # print(tvecs.shape)
    return rvecs, tvecs, trash

def rotation_mtx2euler_angle(Rotation_matrix):
    R = Rotation_matrix
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular=sy< 1e-6#解決Singular問題
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.asin(-R[2, 0])
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    #轉成角度
    rx = x * 180.0 / 3.141592653589793
    ry = y * 180.0 / 3.141592653589793
    rz = z * 180.0 / 3.141592653589793

    return rx,ry,rz
def calc_Aruco_distance(tvec):
    ###### 距離估計 #####
    distance = (tvec[0][0][2]) * 100  #change to cm
    distance_x = (tvec[0][0][0]) * 100  
    distance_y = (tvec[0][0][1]) * 100  
    distance_z = (tvec[0][0][2]) * 100  
    if (distance_z):#判斷值是否正確
        tmp = distance_z*distance_z - (CAMERA_HEIGHT*100)**2
        distance = np.sqrt(tmp)
    else:
        distance = -99999#不能計算
    return distance,distance_x,distance_y,distance_z

def robot_pos_estimation(id,rvecs,tvecs):
    # print("id = ",id)
    rotation_bias = []
    t_vecs_bias = [0,0,0]
    # new_rvecs = np.ndarray((len(rvecs),3))
    rvecs = rvecs[0]
    tvecs = tvecs[0]
    # print("rvecs",rvecs)
    R,_ = cv2.Rodrigues(rvecs[0])
    # print(R)
    inv_R = np.linalg.inv(R)
    
    if (id == 0 or id == 1):
        t_vecs_bias = np.dot(inv_R,[[0],[0],[0]]).flatten()
        rotation_bias = [[1,0,0],
                         [0,1,0],
                         [0,0,1]]

    elif (id == 2 or id == 3):#front
        t_vecs_bias = np.dot(inv_R,[[0],[0],[0]]).flatten()
        rotation_bias_1 = [[np.cos(np.deg2rad(180)),-np.sin(np.deg2rad(180)),0],
                        [np.sin(np.deg2rad(180)),np.cos(np.deg2rad(180)),0],
                        [0,0,1]]
        
        rotation_bias_2 = [[1,0,0],
                         [0,np.cos(np.deg2rad(90)),-np.sin(np.deg2rad(90))],
                         [0,np.sin(np.deg2rad(90)),np.cos(np.deg2rad(90))]]
        
        rotation_bias = np.dot(rotation_bias_1,rotation_bias_2)
    elif (id == 4 or id == 5):#left
        t_vecs_bias = np.dot(inv_R,[[0],[0],[0]]).flatten()
        
        rotation_bias_1 = [[np.cos(np.deg2rad(90)),0,np.sin(np.deg2rad(90))],
                        [0,1,0],
                        [-np.sin(np.deg2rad(90)),0,np.cos(np.deg2rad(90))]]
        rotation_bias_2 = [[1,0,0],
                         [0,np.cos(np.deg2rad(-90)),-np.sin(np.deg2rad(-90))],
                         [0,np.sin(np.deg2rad(-90)),np.cos(np.deg2rad(-90))]]
             
        rotation_bias = np.dot(rotation_bias_1,rotation_bias_2)
    elif(id == 8 or id == 9):#back
        # print("go id == 9")
        # t_vecs_bias = np.dot(inv_R,[[0],[0.05],[-0.05]]).flatten()
        t_vecs_bias = np.dot(inv_R,[[0],[0],[0]]).flatten()
        rotation_bias = [[1,0,0],[0,np.cos(np.deg2rad(-90)),-np.sin(np.deg2rad(-90))],[0,np.sin(np.deg2rad(-90)),np.cos(np.deg2rad(-90))]]

        
    elif(id == 10 or id == 11):#right
        # print("go id == 9")
        t_vecs_bias = np.dot(inv_R,[[0],[0],[0]]).flatten()
        
        rotation_bias_1 = [[1,0,0],
                    [0,np.cos(np.deg2rad(-90)),-np.sin(np.deg2rad(-90))],
                    [0,np.sin(np.deg2rad(-90)),np.cos(np.deg2rad(-90))]]
        
        rotation_bias_2 = [[np.cos(np.deg2rad(-90)),-np.sin(np.deg2rad(-90)),0],
                        [np.sin(np.deg2rad(-90)),np.cos(np.deg2rad(-90)),0],
                        [0,0,1]]

        rotation_bias = np.dot(rotation_bias_1,rotation_bias_2)

    new_tvecs = tvecs + t_vecs_bias 
    new_rvecs,_ = cv2.Rodrigues(np.dot(R,rotation_bias))
    new_rvecs = np.array(new_rvecs)
    new_rvecs = new_rvecs.reshape(-1,1,3)
    new_tvecs = new_tvecs.reshape(-1,1,3)
        
    return new_rvecs,new_tvecs

def display(frame,id_list,degree_list_x,degree_list_y,degree_list_z,distance_list):

    font = cv2.FONT_HERSHEY_SIMPLEX #font for displaying text (below)
    cv2.putText(frame,'ids:'+ str(id_list) ,(0, 110), font, 1, (0, 255, 0), 2,cv2.LINE_AA)
    cv2.putText(frame,'deg_x:'+ str(degree_list_x) ,(0, 150), font, 1, (0, 255, 0), 2,cv2.LINE_AA)
    cv2.putText(frame,'deg_y:'+ str(degree_list_y) ,(0, 190), font, 1, (0, 255, 0), 2,cv2.LINE_AA)
    cv2.putText(frame,'deg_z:'+ str(degree_list_z) ,(0, 230), font, 1, (0, 255, 0), 2,cv2.LINE_AA)
    cv2.putText(frame,'distance:'+ str(distance_list) ,(0, 270), font, 1, (0, 255, 0), 2,cv2.LINE_AA)

def get_Aruco_information(camera_matrix,dist_matrix,frame,corners,ids):
    distance_list = []
    distance_x_list = []
    degree_list = []
    degree_list_x = []
    degree_list_y = []
    degree_list_z = []

    id_list = []
    font = cv2.FONT_HERSHEY_SIMPLEX #font for displaying text (below)
    #    如果找不到id
    if ids is not None:
            for index,i in enumerate(ids):
                rvec, tvec, _ = my_estimatePoseSingleMarkers(corners[index], ARUCO_SIZE, camera_matrix, dist_matrix)
                (rvec-tvec).any() # get rid of that nasty numpy value array error
                # print("fr",rvec)
                new_rvec,new_tvec = robot_pos_estimation(ids[index][0],rvec,tvec)
                
                for i in range(new_rvec.shape[0]):
                    #X: red, Y: green, Z: blue
                    cv2.drawFrameAxes(frame, camera_matrix, dist_matrix, new_rvec, new_tvec, 0.03)
                    frame = aruco.drawDetectedMarkers(frame.copy(), corners, ids,borderColor=(0, 255, 0))
                R = np.zeros((3,3),dtype=np.float64)
                cv2.Rodrigues(new_rvec,R)
                rx,ry,rz = rotation_mtx2euler_angle(Rotation_matrix=R)
                ###### 距離估計 #####
                distance,distance_x,distance_y,distance_z = calc_Aruco_distance(tvec=new_tvec)

                distance_list.append(round(distance_z,2))
                distance_x_list.append(round(distance_x,2))
                degree_list.append(round(rz,2))
                degree_list_x.append(round(rx,2))
                degree_list_y.append(round(ry,2))
                degree_list_z.append(round(rz,2))
                id_list.append(ids[index][0])

    else:
        cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)    
    return frame,id_list,distance_list,distance_x_list,degree_list,degree_list_x,degree_list_y,degree_list_z

if __name__ == '__main__':
    calibration_matrix_path = "./calibration_matrix.npy"
    distortion_coefficients_path = "./distortion_coefficients.npy"
    camera_matrix = np.load(calibration_matrix_path)
    dist_matrix = np.load(distortion_coefficients_path)

    cap = cv2.VideoCapture(1,cv2.CAP_DSHOW) 
    # cap = cv2.VideoCapture('2024-03-27_19-19-03.mp4') # 讀取電腦中的影片
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)  # 设置帧率为60帧/秒
    font = cv2.FONT_HERSHEY_SIMPLEX #font for displaying text (below)

    #num = 0
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
        parameters =  cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        #使用aruco.detectMarkers()函數可以檢測到marker，返回ID和标志板的4个角点坐標
        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

        frame,id_list,distance_list,distance_x_list,degree_list,degree_list_x,degree_list_y,degree_list_z\
              = get_Aruco_information(camera_matrix=camera_matrix,dist_matrix=dist_matrix,frame=frame,corners=corners,ids=ids)
        
        # display(frame,id_list,degree_list_x,degree_list_y,degree_list_z,distance_list)
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