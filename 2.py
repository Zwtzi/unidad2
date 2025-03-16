import cv2
import mediapipe as mp
import pandas as pd

def load_csv(file_path):
    return pd.read_csv(file_path)

def classify_posture(nose_y, left_shoulder_y, right_shoulder_y, neck_angle, torso_angle, data):
    for _, row in data.iterrows():
        if (abs(nose_y - row['Nose_Y']) < 0.03 and  # Tolerancia en datos normalizados
            abs(left_shoulder_y - row['Left_Shoulder_Y']) < 0.03 and
            abs(right_shoulder_y - row['Right_Shoulder_Y']) < 0.03 and
            abs(neck_angle - row['Neck_Angle']) < 5 and
            abs(torso_angle - row['Torso_Angle']) < 5):
            return row['Estado']
    return "Estado Desconocido"

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

data = load_csv('resultados_estudiantes.csv')
cap = cv2.VideoCapture(0)

cv2.namedWindow("Detección de postura", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        nose_y = landmarks[0].y
        left_shoulder_y = landmarks[11].y
        right_shoulder_y = landmarks[12].y
        neck_angle = abs(left_shoulder_y - right_shoulder_y) * 100  # Convertir a grados
        torso_angle = abs(landmarks[23].y - landmarks[24].y) * 100

        estado = classify_posture(nose_y, left_shoulder_y, right_shoulder_y, neck_angle, torso_angle, data)

        cv2.putText(frame, f"Estado: {estado}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Detección de postura", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
