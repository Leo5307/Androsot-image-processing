import cv2
import numpy as np

# 初始化一個空列表來儲存點的位置
points = []

# 滑鼠回呼函數
def mouse_callback(event, x, y, flags, param):
    # 當左鍵點擊時
    if event == cv2.EVENT_LBUTTONDOWN:
        # 記錄並印出(x, y)座標
        print(f"Point: ({x}, {y})")
        points.append((x, y))

# 創建一個簡單的黑色圖像
image = cv2.imread("2024-03-25_17-24-09.jpg")

# 創建一個窗口
cv2.namedWindow('Image')

# 為窗口設定滑鼠回呼函數
cv2.setMouseCallback('Image', mouse_callback)

while True:
    # 顯示圖像
    cv2.imshow('Image', image)
    # 等待按鍵事件，如果是'q'則退出迴圈
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# 釋放所有OpenCV窗口
cv2.destroyAllWindows()

# 將記錄的點轉換為Numpy數組
points_array = np.array(points)

# 儲存數組為.npy格式
np.save('points.npy', points_array)

# 檢視儲存的點
print("Saved points:", points_array)
