import cv2
import mediapipe as mp
import pandas as pd
import os
import sys

# Bloqueo de archivo para evitar múltiples instancias
lock_file = "/tmp/posture_detection.lock"

if os.path.exists(lock_file):
    print("El script ya se está ejecutando.")
    sys.exit(1)

with open(lock_file, 'w') as lock:
    lock.write(str(os.getpid()))

try:
    def load_csv(file_path):
        return pd.read_csv(file_path)

    def classify_posture(nose_y, left_shoulder_y, right_shoulder_y):
        avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
        if nose_y < avg_shoulder_y - 30:
            return "Leyendo"
        elif avg_shoulder_y - 30 <= nose_y <= avg_shoulder_y + 30:
            return "Encorvado"
        elif nose_y > avg_shoulder_y + 30:
            return "Mirando"
        return "Desconocido"

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(0)

    data = load_csv('resultados_estudiantes.csv')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y * frame.shape[0]
            left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0]
            right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0]

            estado = classify_posture(nose_y, left_shoulder_y, right_shoulder_y)
            cv2.putText(frame, estado, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Detección de Postura', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

finally:
    if os.path.exists(lock_file):
        os.remove(lock_file)
