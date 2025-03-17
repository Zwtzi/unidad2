import cv2
import mediapipe as mp
import pandas as pd
import os
import sys
import fcntl
import numpy as np

# Archivo de bloqueo para evitar múltiples instancias
lock_file_path = "/tmp/posture_detection.lock"

try:
    lock_file = open(lock_file_path, 'w')
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("El script ya se está ejecutando. Abortando...")
        sys.exit(1)

    # Cargar el CSV con los datos previos
    def load_csv(file_path):
        if not os.path.exists(file_path):
            print(f"Error: No se encontró el archivo {file_path}")
            sys.exit(1)
        return pd.read_csv(file_path)

    # Buscar en el CSV la postura más cercana
    def classify_using_csv(nose_y, left_shoulder_y, right_shoulder_y, df):
        df_values = df[['Nose_Y', 'Left_Shoulder_Y', 'Right_Shoulder_Y']].values
        input_values = np.array([nose_y, left_shoulder_y, right_shoulder_y])

        distances = np.linalg.norm(df_values - input_values, axis=1)
        min_index = np.argmin(distances)

        return df.iloc[min_index]['Estado']

    # Inicializar MediaPipe Pose y el dibujador
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    # Abrir la cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        sys.exit(1)

    # Cargar el CSV generado por carga.py
    data = load_csv('resultados_estudiantes.csv')

    print("Presiona 'q' para salir.")
    cv2.namedWindow("Detección de Postura", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar el fotograma.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y
            left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

            # Buscar el estado más cercano en el CSV
            estado = classify_using_csv(nose_y, left_shoulder_y, right_shoulder_y, data)

            # Dibujar la postura en el frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

            # Mostrar el estado en el frame
            cv2.putText(frame, estado, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Detección de Postura", frame)

        # Salir con 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Cerrando el programa...")
            break

    cap.release()
    cv2.destroyAllWindows()

finally:
    try:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()
        os.remove(lock_file_path)
    except:
        pass
