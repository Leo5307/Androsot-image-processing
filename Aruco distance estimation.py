# ref:https://github.com/ZengWenJian123/aruco_positioning_2D
import numpy as np
import time
import cv2
import cv2.aruco as aruco
import math
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
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    rvecs = np.array(rvecs)
    rvecs = rvecs.reshape(-1,1,3)
    tvecs = np.array(tvecs)
    tvecs = tvecs.reshape(-1,1,3)
    return rvecs, tvecs, trash
#加载鱼眼镜头的yaml标定文件，检测aruco并且估算与标签之间的距离,获取偏航，俯仰，滚动

#加载相机纠正参数
# cv_file = cv2.FileStorage("yuyan.yaml", cv2.FILE_STORAGE_READ)
# camera_matrix = cv_file.getNode("camera_matrix").mat()
# dist_matrix = cv_file.getNode("dist_coeff").mat()
# cv_file.release()
calibration_matrix_path = "./calibration_matrix.npy"
distortion_coefficients_path = "./distortion_coefficients.npy"
camera_matrix = np.load(calibration_matrix_path)
dist_matrix = np.load(distortion_coefficients_path)


#默认cam参数
# dist=np.array(([[-0.58650416 , 0.59103816, -0.00443272 , 0.00357844 ,-0.27203275]]))
# newcameramtx=np.array([[189.076828   ,  0.    ,     361.20126638]
#  ,[  0 ,2.01627296e+04 ,4.52759577e+02]
#  ,[0, 0, 1]])
# mtx=np.array([[398.12724231  , 0.      ,   304.35638757],
#  [  0.       ,  345.38259888, 282.49861858],
#  [  0.,           0.,           1.        ]])



cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

font = cv2.FONT_HERSHEY_SIMPLEX #font for displaying text (below)

#num = 0
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
    h1, w1 = frame.shape[:2]
    # 读取摄像头画面
    # 纠正畸变
    #？？？校正後會把畫面變小
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_matrix, (h1, w1), 0, (h1, w1))
    # dst1 = cv2.undistort(frame, camera_matrix, dist_matrix, None, newcameramtx)
    # x, y, w1, h1 = roi
    # dst1 = dst1[y:y + h1, x:x + w1]
    # frame=dst1

    #灰度化，检测aruco标签，所用字典为6×6——250
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    #使用aruco.detectMarkers()函数可以检测到marker，返回ID和标志板的4个角点坐标
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

#    如果找不到id
    if ids is not None:
        #获取aruco返回的rvec旋转矩阵、tvec位移矩阵

        rvec, tvec, _ = my_estimatePoseSingleMarkers(corners, 0.053, camera_matrix, dist_matrix)
        # 估计每个标记的姿态并返回值rvet和tvec ---不同
        #rvec为旋转矩阵，tvec为位移矩阵
        # from camera coeficcients
        (rvec-tvec).any() # get rid of that nasty numpy value array error
        print(rvec,np.shape(rvec))
        print("===")
        print(tvec,np.shape(tvec))
        #print(rvec)



        #在画面上 标注auruco标签的各轴
        for i in range(rvec.shape[0]):
            cv2.drawFrameAxes(frame, camera_matrix, dist_matrix, rvec[i, :, :], tvec[i, :, :], 0.03)
            aruco.drawDetectedMarkers(frame.copy(), corners, ids,borderColor=(0, 255, 0))


        ###### 显示id标记 #####
        cv2.putText(frame, "Id: " + str(ids), (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)


        ###### 角度估计 #####
        #print(rvec)
        #考虑Z轴（蓝色）的角度
        #本来正确的计算方式如下，但是由于蜜汁相机标定的问题，实测偏航角度能最大达到104°所以现在×90/104这个系数作为最终角度
        deg=rvec[0][0][2]/math.pi*180
        #deg=rvec[0][0][2]/math.pi*180*90/104
        # 旋转矩阵到欧拉角
        R=np.zeros((3,3),dtype=np.float64)
        cv2.Rodrigues(rvec,R)
        sy=math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular=sy< 1e-6
        if not singular:#偏航，俯仰，滚动
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        # 偏航，俯仰，滚动换成角度
        rx = x * 180.0 / 3.141592653589793
        ry = y * 180.0 / 3.141592653589793
        rz = z * 180.0 / 3.141592653589793

        cv2.putText(frame,'deg_z:'+str(ry)+str('deg'),(0, 140), font, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        #print("偏航，俯仰，滚动",rx,ry,rz)


        ###### 距离估计 #####
        # distance = ((tvec[0][0][2] + 0.02)) * 100  # 单位是米
        distance = (tvec[0][0][2]) * 100  # 单位是米


        # 显示距离
        cv2.putText(frame, 'distance:' + str(round(distance, 4)) + str('m'), (0, 110), font, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

        ####真实坐标换算####（to do）
        # print('rvec:',rvec,'tvec:',tvec)
        # # new_tvec=np.array([[-0.01361995],[-0.01003278],[0.62165339]])
        # # 将相机坐标转换为真实坐标
        # r_matrix, d = cv2.Rodrigues(rvec)
        # r_matrix = -np.linalg.inv(r_matrix)  # 相机旋转矩阵
        # c_matrix = np.dot(r_matrix, tvec)  # 相机位置矩阵

    else:
        ##### DRAW "NO IDS" #####
        cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)


    # 显示结果画面
    cv2.imshow("frame",frame)

    key = cv2.waitKey(1)

    if key == 27:         # 按esc键退出
        print('esc break...')
        cap.release()
        cv2.destroyAllWindows()
        break

    if key == ord(' '):   # 按空格键保存
#        num = num + 1
#        filename = "frames_%s.jpg" % num  # 保存一张图像
        filename = str(time.time())[:10] + ".jpg"
        cv2.imwrite(filename, frame)