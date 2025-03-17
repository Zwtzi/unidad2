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
    data = pd.read_csv(file_path)
    # Escalado automático para normalizar valores
    for col in ['Nose_Y', 'Left_Shoulder_Y', 'Right_Shoulder_Y', 'Neck_Angle', 'Torso_Angle']:
        data[f"{col}_scaled"] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
    return data

data = load_csv('daniel.csv')

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

    # Escalar los valores en tiempo real para que coincidan con el CSV escalado
    nose_y_scaled = (nose_y - data['Nose_Y'].min()) / (data['Nose_Y'].max() - data['Nose_Y'].min())
    left_shoulder_y_scaled = (left_shoulder_y - data['Left_Shoulder_Y'].min()) / (data['Left_Shoulder_Y'].max() - data['Left_Shoulder_Y'].min())
    right_shoulder_y_scaled = (right_shoulder_y - data['Right_Shoulder_Y'].min()) / (data['Right_Shoulder_Y'].max() - data['Right_Shoulder_Y'].min())
    neck_angle_scaled = (neck_angle - data['Neck_Angle'].min()) / (data['Neck_Angle'].max() - data['Neck_Angle'].min())
    torso_angle_scaled = (torso_angle - data['Torso_Angle'].min()) / (data['Torso_Angle'].max() - data['Torso_Angle'].min())

    # Calcular distancia euclidiana con mayor peso en cuello y torso
    data["distance"] = np.sqrt(
        0.2 * (data["Nose_Y_scaled"] - nose_y_scaled) ** 2 +
        0.2 * (data["Left_Shoulder_Y_scaled"] - left_shoulder_y_scaled) ** 2 +
        0.2 * (data["Right_Shoulder_Y_scaled"] - right_shoulder_y_scaled) ** 2 +
        0.3 * (data["Neck_Angle_scaled"] - neck_angle_scaled) ** 2 +
        0.3 * (data["Torso_Angle_scaled"] - torso_angle_scaled) ** 2
    )

    # Obtener la postura con la menor distancia
    closest_row = data.loc[data["distance"].idxmin()]
    return closest_row["Estado"], closest_row

# Captura de video
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('/dev/video2')  # Reemplaza '/dev/video2' según lo que te mostró el comando anterior

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

        # Dibuja el esqueleto en el frame
        mp.solutions.drawing_utils.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

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
        estado, valores = find_closest_posture(
            smoothed_nose_y, smoothed_left_shoulder_y, smoothed_right_shoulder_y,
            smoothed_neck_angle, smoothed_torso_angle, data
        )

        # Mostrar el estado y los valores escaneados
        cv2.putText(frame, f"Estado: {estado}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Nose_Y: {smoothed_nose_y:.4f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Left_Shoulder_Y: {smoothed_left_shoulder_y:.4f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Right_Shoulder_Y: {smoothed_right_shoulder_y:.4f}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Neck_Angle: {smoothed_neck_angle:.2f}", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Torso_Angle: {smoothed_torso_angle:.2f}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Detección de postura", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
