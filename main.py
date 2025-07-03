import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt, detrend
import time
from collections import deque

# --- 參數設定 ---
BUFFER_SIZE = 150
FS = 30
LOW_BPM = 50
HIGH_BPM = 180
SNR_THRESHOLD = 2.5

# --- 初始化工具 ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

ROI_INDEXES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

# --- 主程式 ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("錯誤: 無法開啟攝影機。")
    exit()

signal_buffer = deque(maxlen=BUFFER_SIZE)
timestamps = deque(maxlen=BUFFER_SIZE)
bpm = 0
display_text = "Initializing..."
text_color = (255, 255, 0) # 初始顏色

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("忽略空的攝影機幀。")
        continue

    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    image.flags.writeable = True

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        h, w, _ = image.shape
        roi_points = np.array([(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in ROI_INDEXES])
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [roi_points], 255)
        
        green_channel = image[:, :, 1]
        roi_green_mean = cv2.mean(green_channel, mask=mask)[0]
        
        signal_buffer.append(roi_green_mean)
        timestamps.append(time.time())

        for point in roi_points:
            cv2.circle(image, tuple(point), 2, (0, 255, 0), -1)

        if len(signal_buffer) == BUFFER_SIZE:
            raw_signal = np.array(list(signal_buffer))
            
            detrended_signal = detrend(raw_signal)
            
            low_hz = LOW_BPM / 60.0
            high_hz = HIGH_BPM / 60.0
            nyquist_freq = 0.5 * FS
            
            if nyquist_freq > high_hz: # 確保奈奎斯特頻率足夠高
                b, a = butter(2, [low_hz/nyquist_freq, high_hz/nyquist_freq], btype='band')
                filtered_signal = filtfilt(b, a, detrended_signal)
                
                fft_data = np.abs(np.fft.fft(filtered_signal))
                freqs = np.fft.fftfreq(BUFFER_SIZE, 1.0/FS)
                
                valid_indices = np.where((freqs >= low_hz) & (freqs <= high_hz))
                valid_freqs = freqs[valid_indices]
                valid_fft_data = fft_data[valid_indices]
                
                if len(valid_freqs) > 0:
                    peak_index = np.argmax(valid_fft_data)
                    peak_freq = valid_freqs[peak_index]
                    peak_power = valid_fft_data[peak_index]
                    
                    noise_power = np.mean(valid_fft_data)
                    snr = peak_power / noise_power if noise_power > 0 else 0
                    
                    
                    
                    if snr > SNR_THRESHOLD:
                        bpm = peak_freq * 60
                        display_text = f"BPM: {bpm:.1f}"
                        text_color = (0, 255, 0)
                    else:
                        display_text = "No Pulse Detected"
                        text_color = (0, 0, 255)
                
                signal_buffer.clear()
                timestamps.clear()

    else:
        display_text = "No Face Detected"
        text_color = (0, 0, 255)

    cv2.putText(image, display_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)
    cv2.imshow('Real-time rPPG with Liveness Detection', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

face_mesh.close()
cap.release()
cv2.destroyAllWindows()