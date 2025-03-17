import cv2
import mediapipe as mp
import os
import sys
import fcntl

# Archivo de bloqueo para evitar múltiples instancias
lock_file_path = "/tmp/posture_detection.lock"

try:
    lock_file = open(lock_file_path, 'w')
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("El script ya se está ejecutando. Abortando...")
        sys.exit(1)

    def classify_posture(nose_y, left_shoulder_y, right_shoulder_y):
        avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
        if nose_y < avg_shoulder_y - 30:
            return "Leyendo"
        elif avg_shoulder_y - 30 <= nose_y <= avg_shoulder_y + 30:
            return "Encorvado"
        elif nose_y > avg_shoulder_y + 30:
            return "Mirando"
        return "Desconocido"

    # Inicializar MediaPipe Pose y el dibujador
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    # Abrir la cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        sys.exit(1)

    print("Presiona 'q' para salir.")
    cv2.namedWindow("Detección de Postura", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar el fotograma.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y * frame.shape[0]
            left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0]
            right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0]

            estado = classify_posture(nose_y, left_shoulder_y, right_shoulder_y)

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
