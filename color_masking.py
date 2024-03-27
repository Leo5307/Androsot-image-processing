
import cv2
import numpy as np
# HSV format(8 bytes) => [ 0~180, 0~255, 0~255]

# define range of blue color in HSV
lower_blue = np.array([110,100,50]) #lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])#upper_blue = np.array([130,255,255])

# define range of green color in HSV 
lower_green = np.array([50,50,100])
upper_green = np.array([70,255,255])

# define range of red color in HSV 
# lower_red = np.array([-10,178,128])
# upper_red = np.array([11,256,240])
lower_red_1= np.array([0, 50, 50])
upper_red_1 = np.array([10, 255, 255])
lower_red_2 = np.array([150, 50, 50])
upper_red_2 = np.array([180, 255, 255])

# red1 = np.array([-10,178,128, 11,256,240])
# red2 = np.array([168, 178, 131, 189, 252, 232])

# define range of white color in HSV 
lower_white = np.array([0,0,240])
upper_white = np.array([150,5,255])

# define range of yellow color in HSV 
lower_yellow = np.array([20,50,100])
upper_yellow = np.array([40,255,255])

def colorMasking(frame):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Threshold the HSV image to get only blue colors
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    # Threshold the HSV image to get only green colors
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    # Threshold the HSV image to get only red colors   
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red_1, upper_red_1) , cv2.inRange(hsv, lower_red_2, upper_red_2))
    # Threshold the HSV image to get only red colors   
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    # Threshold the HSV image to get only red colors   
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    return mask_blue, mask_green, mask_red, mask_white, mask_yellow


if __name__ == '__main__':
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    # cap = cv2.VideoCapture('2024-03-25_17-24-36.mp4') # 讀取電腦中的影片
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

        # Convert BGR to HSV
        mask_blue, mask_green, mask_red, mask_white, mask_yellow = colorMasking(frame=frame)
        res = cv2.bitwise_and(frame,frame, mask= mask_blue)

        cv2.imshow('frame',frame)
        cv2.imshow('mask',mask_blue)
        cv2.imshow('mask',res)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()