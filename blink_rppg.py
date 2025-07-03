import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt, detrend
import time
from collections import deque

# ---------- 參數設定 ----------
EYE_AR_THRESH = 0.22
EYE_AR_CONSEC_FRAMES = 2
BLINK_WINDOW_SECONDS = 5            # 延長眨眼時間
BUFFER_SIZE = 150
FS = 30
LOW_BPM = 50
HIGH_BPM = 180
SNR_THRESHOLD = 2.0                 # 放寬 SNR 門檻
MAX_FAIL_COUNT = 3                 # 誤判容錯：連續 3 次失敗才顯示 Failed

# ---------- 初始化 ----------
BLINK_COUNTER = 0
TOTAL_BLINKS = 0
last_blink_time = time.time()
collecting_rppg = False
fail_count = 0
bpm = 0

# MediaPipe 初始化
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

LEFT_EYE_INDEXES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDEXES = [33, 160, 158, 133, 153, 144]
ROI_INDEXES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
               400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

def eye_aspect_ratio(eye_landmarks_list, image_shape):
    h, w = image_shape
    coords = np.array([(lm.x * w, lm.y * h) for lm in eye_landmarks_list])
    A = np.linalg.norm(coords[1] - coords[5])
    B = np.linalg.norm(coords[2] - coords[4])
    C = np.linalg.norm(coords[0] - coords[3])
    ear = (A + B) / (2.0 * C)
    return ear

# ---------- 視訊 ----------
cap = cv2.VideoCapture(0)
signal_buffer = deque(maxlen=BUFFER_SIZE)
filtered_signal_buffer = deque(maxlen=BUFFER_SIZE)

liveness_result = "Initializing..."
liveness_color = (255, 255, 0)

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
graph_w = frame_w // 2

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    graph_canvas = np.zeros((frame_h, graph_w, 3), dtype=np.uint8)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    current_time = time.time()

    if results.multi_face_landmarks:
        all_landmarks = results.multi_face_landmarks[0].landmark
        left_eye_lms = [all_landmarks[i] for i in LEFT_EYE_INDEXES]
        right_eye_lms = [all_landmarks[i] for i in RIGHT_EYE_INDEXES]
        left_ear = eye_aspect_ratio(left_eye_lms, (frame_h, frame_w))
        right_ear = eye_aspect_ratio(right_eye_lms, (frame_h, frame_w))
        ear = (left_ear + right_ear) / 2.0

        if ear < EYE_AR_THRESH:
            BLINK_COUNTER += 1
        else:
            if BLINK_COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL_BLINKS += 1
                last_blink_time = current_time
                collecting_rppg = True
            BLINK_COUNTER = 0

        time_since_blink = current_time - last_blink_time
        collecting_rppg = time_since_blink < BLINK_WINDOW_SECONDS

        if collecting_rppg:
            roi_points = np.array([(int(all_landmarks[i].x * frame_w), int(all_landmarks[i].y * frame_h)) for i in ROI_INDEXES])
            overlay = image.copy()
            cv2.fillPoly(overlay, [roi_points], (0, 255, 0))
            image = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)

            mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
            cv2.fillPoly(mask, [roi_points], 255)
            green_channel = image[:, :, 1]
            roi_green_mean = cv2.mean(green_channel, mask=mask)[0]
            signal_buffer.append(roi_green_mean)

            if len(signal_buffer) == BUFFER_SIZE:
                raw_signal = np.array(list(signal_buffer))
                detrended_signal = detrend(raw_signal)
                low_hz, high_hz = LOW_BPM / 60.0, HIGH_BPM / 60.0
                nyquist_freq = 0.5 * FS

                if nyquist_freq > high_hz:
                    b, a = butter(2, [low_hz / nyquist_freq, high_hz / nyquist_freq], btype='band')
                    filtered_signal = filtfilt(b, a, detrended_signal)
                    filtered_signal_buffer.clear()
                    filtered_signal_buffer.extend(filtered_signal)

                    fft_data = np.abs(np.fft.fft(filtered_signal))
                    freqs = np.fft.fftfreq(BUFFER_SIZE, 1.0 / FS)
                    valid_indices = np.where((freqs >= low_hz) & (freqs <= high_hz))
                    valid_freqs, valid_fft_data = freqs[valid_indices], fft_data[valid_indices]

                    if len(valid_freqs) > 0:
                        peak_index = np.argmax(valid_fft_data)
                        peak_freq = valid_freqs[peak_index]
                        peak_power = valid_fft_data[peak_index]
                        noise_power = np.mean(valid_fft_data)
                        snr = peak_power / noise_power if noise_power > 0 else 0

                        if snr > SNR_THRESHOLD:
                            bpm = peak_freq * 60
                            fail_count = 0
                            liveness_result = "Liveness Passed"
                            liveness_color = (0, 255, 0)
                        else:
                            fail_count += 1
                            if fail_count >= MAX_FAIL_COUNT:
                                liveness_result = "Liveness Failed (Low SNR)"
                                liveness_color = (0, 0, 255)
                    signal_buffer.clear()
        else:
            liveness_result = "Please blink to start"
            liveness_color = (0, 0, 255)
            signal_buffer.clear()
            filtered_signal_buffer.clear()

        # 畫出眼睛輪廓與狀態
        for points in [left_eye_lms, right_eye_lms]:
            eye_points = np.array([(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in points], dtype=np.int32)
            cv2.polylines(image, [eye_points], True, (255, 255, 0), 1)
        cv2.putText(image, f"EAR: {ear:.2f}", (frame_w - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(image, f"Blinks: {TOTAL_BLINKS}", (frame_w - 180, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    else:
        liveness_result = "No Face Detected"
        liveness_color = (0, 0, 255)
        last_blink_time = current_time
        signal_buffer.clear()
        filtered_signal_buffer.clear()

    # 繪製波形圖
    if len(filtered_signal_buffer) > 0:
        signal_to_plot = np.array(list(filtered_signal_buffer))
        normalized_signal = cv2.normalize(signal_to_plot, None, alpha=0, beta=frame_h - 50, norm_type=cv2.NORM_MINMAX)
        points = [(int(i * (graph_w / BUFFER_SIZE)), int(frame_h - val - 25)) for i, val in enumerate(normalized_signal)]
        if len(points) > 1:
            cv2.polylines(graph_canvas, [np.array(points, dtype=np.int32)], False, (0, 255, 0), 2)

    # 顯示主畫面文字資訊
    cv2.putText(image, liveness_result, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, liveness_color, 3)
    if bpm > 0:
        cv2.putText(image, f"BPM: {bpm:.1f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    final_frame = np.hstack((image, graph_canvas))
    cv2.imshow('rPPG Liveness Detection Dashboard', final_frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

face_mesh.close()
cap.release()
cv2.destroyAllWindows()
