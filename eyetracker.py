import cv2
import math
import mediapipe as mp
import numpy as np
from collections import deque

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]
LEFT_EYE_LIDS = [159, 145]
RIGHT_EYE_LIDS = [384, 374]

ALERT_FRAMES = 30
off_center_count = 0

angle_history = deque(maxlen=5)
baseline_angle = None
vertical_baseline = None

cap = cv2.VideoCapture(2)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

def get_eye_metrics(face_landmarks, iris_idxs, corner_idxs, lid_idxs, w, h):
    iris_pts = []
    for i in iris_idxs:
        x = int(face_landmarks.landmark[i].x * w)
        y = int(face_landmarks.landmark[i].y * h)
        iris_pts.append((x, y))
    (cx, cy), _ = cv2.minEnclosingCircle(np.array(iris_pts, dtype=np.int32))
    cx, cy = int(cx), int(cy)

    corners = [(int(face_landmarks.landmark[i].x * w),
                int(face_landmarks.landmark[i].y * h)) for i in corner_idxs]
    y_top = face_landmarks.landmark[lid_idxs[0]].y * h
    y_bottom = face_landmarks.landmark[lid_idxs[1]].y * h
    x_center = (corners[0][0] + corners[1][0]) // 2
    y_center = int((y_top + y_bottom) / 2)

    dx = cx - x_center
    dy = cy - y_center
    theta = math.degrees(math.atan2(-dy, dx))

    # vertical ratio for up/down detection
    vertical_ratio = (cy - y_top) / (y_bottom - y_top + 1e-6)
    return theta, vertical_ratio

def classify_direction(theta, vertical_ratio, base_theta, base_vert):
    diff_h = theta - base_theta
    diff_v = vertical_ratio - base_vert

    # Horizontal thresholds
    if abs(diff_h) < 15:
        horiz = 'Center'
    elif diff_h > 25:
        horiz = 'Right'
    elif diff_h < -25:
        horiz = 'Left'
    else:
        horiz = 'Center'

    # Vertical thresholds
    if diff_v < -0.1:
        vert = 'Up'
    elif diff_v > 0.1:
        vert = 'Down'
    else:
        vert = 'Center'

    # Combine intelligently
    if vert != 'Center':
        return vert
    else:
        return horiz

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    results = face_mesh.process(rgb)
    directions = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            theta_L, vL = get_eye_metrics(face_landmarks, LEFT_IRIS, LEFT_EYE_CORNERS, LEFT_EYE_LIDS, w, h)
            theta_R, vR = get_eye_metrics(face_landmarks, RIGHT_IRIS, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS, w, h)
            theta = (theta_L + theta_R) / 2
            vert_ratio = (vL + vR) / 2

            angle_history.append(theta)
            smooth_theta = sum(angle_history) / len(angle_history)

            # Auto-calibrate
            if baseline_angle is None or vertical_baseline is None:
                baseline_angle = smooth_theta
                vertical_baseline = vert_ratio
                cv2.putText(frame, "Calibrating... Look Straight", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                continue

            d = classify_direction(smooth_theta, vert_ratio, baseline_angle, vertical_baseline)
            directions.append(d)

        if directions:
            gaze = max(set(directions), key=directions.count)
            cv2.putText(frame, f'Gaze: {gaze}', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

            if gaze != 'Center':
                off_center_count += 1
            else:
                off_center_count = 0

            if off_center_count > ALERT_FRAMES:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                cv2.putText(frame, 'FOCUS!', (w // 3, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)

    cv2.imshow("Eye Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
