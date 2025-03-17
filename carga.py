import cv2
import mediapipe as mp
import os
import csv
import math

def calculate_angle(p1, p2):
    """Calcula el ángulo entre dos puntos en un plano vertical"""
    return abs(math.degrees(math.atan2(p2.y - p1.y, p2.x - p1.x)))

# Rangos de referencia optimizados
POSTURE_THRESHOLDS = {
    "Encorvado": {
        "nose_y": (0.406, 0.510),
        "left_shoulder_y": (0.525, 0.609),
        "right_shoulder_y": (0.532, 0.609),
        "neck_angle": (36.77, 59.23),
        "torso_angle": (92.30, 99.48)
    },
    "Leyendo": {
        "nose_y": (0.241, 0.459),
        "left_shoulder_y": (0.380, 0.575),
        "right_shoulder_y": (0.363, 0.575),
        "neck_angle": (25.77, 55.49),
        "torso_angle": (89.32, 101.66)
    },
    "Mirando": {
        "nose_y": (0.338, 0.469),
        "left_shoulder_y": (0.482, 0.606),
        "right_shoulder_y": (0.512, 0.606),
        "neck_angle": (56.35, 61.69),
        "torso_angle": (89.36, 93.88)
    }
}

def detect_posture(landmarks):
    nose_y = landmarks[0].y
    left_shoulder_y = landmarks[11].y
    right_shoulder_y = landmarks[12].y
    nose_x = landmarks[0].x
    hip_center_x = (landmarks[23].x + landmarks[24].x) / 2
    
    neck_angle = calculate_angle(landmarks[0], landmarks[11])
    torso_angle = calculate_angle(landmarks[11], landmarks[23])
    
    for posture, thresholds in POSTURE_THRESHOLDS.items():
        if (thresholds["nose_y"][0] <= nose_y <= thresholds["nose_y"][1] and
            thresholds["left_shoulder_y"][0] <= left_shoulder_y <= thresholds["left_shoulder_y"][1] and
            thresholds["right_shoulder_y"][0] <= right_shoulder_y <= thresholds["right_shoulder_y"][1] and
            thresholds["neck_angle"][0] <= neck_angle <= thresholds["neck_angle"][1] and
            thresholds["torso_angle"][0] <= torso_angle <= thresholds["torso_angle"][1]):
            return posture
    
    # Detectar si el estudiante está mirando hacia un lado
    if abs(nose_x - hip_center_x) >= 0.05:
        return "Mirando hacia un lado"
    
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
    output_csv = "resultados_estudiantes2.csv"
    
    if not os.path.exists(input_folder):
        print(f"La carpeta '{input_folder}' no existe. Asegúrate de colocar los videos en ella.")
    else:
        process_videos(input_folder, output_csv)
