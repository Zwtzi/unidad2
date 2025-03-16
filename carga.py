import cv2
import mediapipe as mp
import os
import csv
import math

def calculate_angle(p1, p2):
    """Calcula el ángulo entre dos puntos en un plano vertical"""
    return abs(math.degrees(math.atan2(p2.y - p1.y, p2.x - p1.x)))

def detect_posture(landmarks):
    nose_y = landmarks[0].y
    left_shoulder_y = landmarks[11].y
    right_shoulder_y = landmarks[12].y
    avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
    shoulder_diff = abs(left_shoulder_y - right_shoulder_y)

    nose_x = landmarks[0].x
    hip_center_x = (landmarks[23].x + landmarks[24].x) / 2  # Promedio de ambas caderas

    neck_angle = calculate_angle(landmarks[0], landmarks[11])  # Ángulo del cuello
    torso_angle = calculate_angle(landmarks[11], landmarks[23])  # Ángulo del torso

    # Criterios refinados
    if (
        nose_y < avg_shoulder_y - 0.01 
        and shoulder_diff < 0.02 
        and 10 < neck_angle < 25
        and abs(nose_x - hip_center_x) < 0.02
    ):
        return "Leyendo"
    
    elif (
        nose_y > avg_shoulder_y + 0.04 
        or shoulder_diff > 0.04 
        or neck_angle >= 30
        or torso_angle >= 20
    ):
        return "Encorvado"
    
    elif (
        avg_shoulder_y - 0.01 <= nose_y <= avg_shoulder_y + 0.01
        and abs(neck_angle) < 10
        and abs(nose_x - hip_center_x) < 0.03
    ):
        return "Mirando"

    elif abs(nose_x - hip_center_x) >= 0.05:  # Si la nariz está muy desplazada lateralmente
        return "Mirando hacia un lado"

    else:
        return "Desconocido"

def process_videos(input_folder, output_csv):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Video", "Frame", "Nose_Y", "Left_Shoulder_Y", 
            "Right_Shoulder_Y", "Neck_Angle", "Torso_Angle", "Estado"
        ])

        for video_file in os.listdir(input_folder):
            if video_file.endswith(".mp4"):
                cap = cv2.VideoCapture(os.path.join(input_folder, video_file))
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(frame_rgb)

                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        estado = detect_posture(landmarks)
                        neck_angle = calculate_angle(landmarks[0], landmarks[11])
                        torso_angle = calculate_angle(landmarks[11], landmarks[23])

                        writer.writerow([
                            video_file, frame_count,
                            round(landmarks[0].y, 4), 
                            round(landmarks[11].y, 4), 
                            round(landmarks[12].y, 4),
                            round(neck_angle, 2),
                            round(torso_angle, 2),
                            estado
                        ])
                    frame_count += 1

                cap.release()

    print(f"Archivo CSV generado: {output_csv}")

if __name__ == "__main__":
    input_folder = "videos_estudiantes"
    output_csv = "resultados_estudiantes.csv"

    if not os.path.exists(input_folder):
        print(f"La carpeta '{input_folder}' no existe. Asegúrate de colocar los videos en ella.")
    else:
        process_videos(input_folder, output_csv)
