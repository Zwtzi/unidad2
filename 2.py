import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from collections import deque

# Configurar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Cargar datos del CSV
def load_csv(file_path):
    return pd.read_csv(file_path)

data = load_csv('resultados_estudiantes.csv')

# Cola para suavizar los valores (media móvil)
history_size = 5
nose_queue = deque(maxlen=history_size)
left_shoulder_queue = deque(maxlen=history_size)
right_shoulder_queue = deque(maxlen=history_size)
neck_angle_queue = deque(maxlen=history_size)
torso_angle_queue = deque(maxlen=history_size)

# Función para encontrar la postura más cercana usando distancia euclidiana
def find_closest_posture(nose_y, left_shoulder_y, right_shoulder_y, neck_angle, torso_angle, data):
    if data.empty:
        return "Estado Desconocido"

    # Calcular distancia euclidiana normalizando los ángulos
    data["distance"] = np.sqrt(
        (data["Nose_Y"] - nose_y) ** 2 +
        (data["Left_Shoulder_Y"] - left_shoulder_y) ** 2 +
        (data["Right_Shoulder_Y"] - right_shoulder_y) ** 2 +
        ((data["Neck_Angle"] - neck_angle) / 30) ** 2 +  # Normalización angular
        ((data["Torso_Angle"] - torso_angle) / 30) ** 2
    )

    # Obtener la postura con la menor distancia
    closest_row = data.loc[data["distance"].idxmin()]
    return closest_row["Estado"]

# Captura de video
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

        # Obtener coordenadas normalizadas
        nose_y = landmarks[0].y
        left_shoulder_y = landmarks[11].y
        right_shoulder_y = landmarks[12].y
        neck_angle = abs(left_shoulder_y - right_shoulder_y) * 100  # Aproximación a grados
        torso_angle = abs(landmarks[23].y - landmarks[24].y) * 100

        # Agregar valores a la cola para suavizado
        nose_queue.append(nose_y)
        left_shoulder_queue.append(left_shoulder_y)
        right_shoulder_queue.append(right_shoulder_y)
        neck_angle_queue.append(neck_angle)
        torso_angle_queue.append(torso_angle)

        # Promediar valores para suavizar detección
        smoothed_nose_y = np.mean(nose_queue)
        smoothed_left_shoulder_y = np.mean(left_shoulder_queue)
        smoothed_right_shoulder_y = np.mean(right_shoulder_queue)
        smoothed_neck_angle = np.mean(neck_angle_queue)
        smoothed_torso_angle = np.mean(torso_angle_queue)

        # Encontrar el estado más cercano
        estado = find_closest_posture(
            smoothed_nose_y, smoothed_left_shoulder_y, smoothed_right_shoulder_y,
            smoothed_neck_angle, smoothed_torso_angle, data
        )

        # Mostrar resultado en la ventana
        cv2.putText(frame, f"Estado: {estado}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Detección de postura", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
