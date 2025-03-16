import cv2
import mediapipe as mp
import os
import csv

def detect_posture(landmarks):
    nose_y = landmarks[0].y
    left_shoulder_y = landmarks[11].y
    right_shoulder_y = landmarks[12].y

    if nose_y < left_shoulder_y and nose_y < right_shoulder_y:
        return "Leyendo"
    elif nose_y > left_shoulder_y and nose_y > right_shoulder_y:
        return "Encorvado"
    else:
        return "Mirando"

def process_videos(input_folder, output_csv):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Video", "Frame", "Nose_Y", "Left_Shoulder_Y", "Right_Shoulder_Y", "Estado"])

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
                        writer.writerow([
                            video_file, frame_count,
                            landmarks[0].y, landmarks[11].y, landmarks[12].y,
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