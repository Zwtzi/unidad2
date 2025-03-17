import cv2
import mediapipe as mp
import os
import csv
import math
import numpy as np

def calculate_angle(p1, p2):
    return abs(math.degrees(math.atan2(p2.y - p1.y, p2.x - p1.x)))

# Umbrales basados en estadísticas
POSTURE_THRESHOLDS = {
    "Encorvado": {
        "nose_y": (0.2673, 0.4699),
        "left_shoulder_y": (0.3272, 0.4317),
        "right_shoulder_y": (0.3459, 0.4379),
        "neck_angle": (4.68, 37.38),
        "torso_angle": (93.04, 96.66)
    },
    "Leyendo": {
        "nose_y": (0.1900, 0.2783),
        "left_shoulder_y": (0.3054, 0.3696),
        "right_shoulder_y": (0.3224, 0.3896),
        "neck_angle": (40.13, 57.60),
        "torso_angle": (91.24, 94.60)
    },
    "Mirando": {
        "nose_y": (0.1394, 0.2119),
        "left_shoulder_y": (0.3202, 0.3704),
        "right_shoulder_y": (0.3336, 0.3841),
        "neck_angle": (58.01, 65.63),
        "torso_angle": (91.54, 95.84)
    }
}

# Outliers detectados
OUTLIERS = {
    "Encorvado": {"nose_y": 395, "left_shoulder_y": 0, "right_shoulder_y": 24, "neck_angle": 368, "torso_angle": 152},
    "Leyendo": {"nose_y": 138, "left_shoulder_y": 39, "right_shoulder_y": 56, "neck_angle": 524, "torso_angle": 360},
    "Mirando": {"nose_y": 346, "left_shoulder_y": 112, "right_shoulder_y": 148, "neck_angle": 205, "torso_angle": 82}
}

def remove_outliers(value, posture, key):
    if OUTLIERS[posture][key] > 100:  # Umbral arbitrario para detección de valores extremos
        return np.nan  # Ignorar el valor si es un outlier severo
    return value

def detect_posture(landmarks):
    features = {
        "nose_y": landmarks[0].y,
        "left_shoulder_y": landmarks[11].y,
        "right_shoulder_y": landmarks[12].y,
        "neck_angle": calculate_angle(landmarks[0], landmarks[11]),
        "torso_angle": calculate_angle(landmarks[11], landmarks[23])
    }
    
    best_match = None
    best_distance = float('inf')
    
    for posture, thresholds in POSTURE_THRESHOLDS.items():
        distance = sum(abs(features[key] - np.mean(thresholds[key])) for key in features)
        if distance < best_distance:
            best_distance = distance
            best_match = posture
    
    return best_match

def process_videos(input_folder, output_csv):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Video", "Frame", "Nose_Y", "Left_Shoulder_Y", "Right_Shoulder_Y", "Neck_Angle", "Torso_Angle", "Estado"])

        for video_file in os.listdir(input_folder):
            if video_file.endswith(".mp4"):
                cap = cv2.VideoCapture(os.path.join(input_folder, video_file))
                frame_count = 0
                print(f"Procesando video: {video_file}")

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(frame_rgb)

                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        estado = detect_posture(landmarks)
                        neck_angle = remove_outliers(calculate_angle(landmarks[0], landmarks[11]), estado, "neck_angle")
                        torso_angle = remove_outliers(calculate_angle(landmarks[11], landmarks[23]), estado, "torso_angle")

                        writer.writerow([
                            video_file, frame_count,
                            round(landmarks[0].y, 4),
                            round(landmarks[11].y, 4),
                            round(landmarks[12].y, 4),
                            round(neck_angle, 2) if not np.isnan(neck_angle) else "NA",
                            round(torso_angle, 2) if not np.isnan(torso_angle) else "NA",
                            estado
                        ])
                    frame_count += 1
                cap.release()
    print(f"Archivo CSV generado: {output_csv}")

if __name__ == "__main__":
    input_folder = "videos2"
    output_csv = "resultado_posturas.csv"
    if not os.path.exists(input_folder):
        print(f"La carpeta '{input_folder}' no existe. Asegúrate de colocar los videos en ella.")
    else:
        process_videos(input_folder, output_csv)