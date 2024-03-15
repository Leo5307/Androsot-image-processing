import numpy as np
import cv2, PIL
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import cv2

cap = cv2.VideoCapture(0)
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
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = cv2.imread("squares.png")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids,borderColor=(0, 255, 0))

# for i in range(len(ids)):
#     c = corners[i][0]
#     center = tuple(np.int32(c.mean(axis=0)))  # Calculate the center point of the marker
#     marker_id = str(ids[i][0])
    # cv2.putText(frame_markers, marker_id, center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the result
    cv2.imshow("Frame Markers", frame_markers)
        # Display the resulting frame
    # cv2.imshow('frame', gray)
    if cv2.waitKey(1) == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

# fig = plt.figure()
# nx = 4
# ny = 3
# for i in range(1, nx*ny+1):
#     ax = fig.add_subplot(ny,nx, i)
#     img = aruco.drawMarker(aruco_dict,i, 700)
#     plt.imshow(img, cmap = mpl.cm.gray, interpolation = "nearest")
#     ax.axis("off")

# plt.savefig("squares.png")
frame = cv2.imread("squares.png")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
aruco_dict = cv2.aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids,borderColor=(0, 255, 0))

# for i in range(len(ids)):
#     c = corners[i][0]
#     center = tuple(np.int32(c.mean(axis=0)))  # Calculate the center point of the marker
#     marker_id = str(ids[i][0])
    # cv2.putText(frame_markers, marker_id, center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Display the result
cv2.imshow("Frame Markers", frame_markers)
# cv2.imwrite('frame_markers.png', frame_markers)
cv2.waitKey(0)
cv2.destroyAllWindows()


def quad_area(data):
    l = data.shape[0]//2
    corners = data[["c1", "c2", "c3", "c4"]].values.reshape(l, 2,4)
    c1 = corners[:, :, 0]
    c2 = corners[:, :, 1]
    c3 = corners[:, :, 2]
    c4 = corners[:, :, 3]
    e1 = c2-c1
    e2 = c3-c2
    e3 = c4-c3
    e4 = c1-c4
    a = -.5 * (np.cross(-e1, e2, axis = 1) + np.cross(-e3, e4, axis = 1))
    return a

corners2 = np.array([c[0] for c in corners])

data = pd.DataFrame({"x": corners2[:,:,0].flatten(), "y": corners2[:,:,1].flatten()},
                   index = pd.MultiIndex.from_product(
                           [ids.flatten(), ["c{0}".format(i )for i in np.arange(4)+1]],
                       names = ["marker", ""] ))

data = data.unstack().swaplevel(0, 1, axis = 1).stack()
data["m1"] = data[["c1", "c2"]].mean(axis = 1)
data["m2"] = data[["c2", "c3"]].mean(axis = 1)
data["m3"] = data[["c3", "c4"]].mean(axis = 1)
data["m4"] = data[["c4", "c1"]].mean(axis = 1)
data["o"] = data[["m1", "m2", "m3", "m4"]].mean(axis = 1)
print(data)
