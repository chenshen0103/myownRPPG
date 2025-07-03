import cv2
import time
print("程式開始，正在 import 函式庫...")
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt, detrend
from collections import deque
print("函式庫 import 完畢。")

print("正在初始化攝影機...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("錯誤: 無法開啟攝影機。")
    exit()
else:
    print("攝影機開啟成功！")

print("正在初始化 MediaPipe...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
print("MediaPipe 初始化完畢。")

print(f"準備進入主迴圈，攝影機狀態 (cap.isOpened()): {cap.isOpened()}")

while cap.isOpened():
    print("--- 迴圈開始 ---")
    
    print("正在執行 cap.read()... (如果程式在這裡停止，就是它崩潰了)")
    success, image = cap.read()
    print(f"cap.read() 執行完畢。成功狀態: {success}")

    if not success:
        print("讀取影像幀失敗，跳過此幀。")
        continue
    
    print("影像幀讀取成功，程式將在顯示一次後結束。")
    
    cv2.putText(image, "Debug Mode: Success!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.imshow('Final Debug Test', image)
    
    cv2.waitKey(2000) # 顯示影像 2 秒
    
    break # 執行一次就跳出迴圈

print("--- 迴圈已結束或未進入 ---")

print("正在釋放資源...")
face_mesh.close()
cap.release()
cv2.destroyAllWindows()
print("程式正常結束。")