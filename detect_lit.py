from ultralytics import YOLO
import cv2
import face_recognition
import numpy as np
import os
import time
import csv
from datetime import datetime

# Load YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")

# Webcam settings
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Load known faces
known_face_encodings = []
known_face_names = []
KNOWN_FACES_DIR = "known_faces"

for name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    for filename in os.listdir(person_dir):
        img_path = os.path.join(person_dir, filename)
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) == 0:
            print(f"[!] No face found in {img_path}, skipping...")
            continue
        encoding = face_recognition.face_encodings(image, face_locations)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(name)

# For saving unknown snapshots
UNKNOWN_DIR = "snapshots"
os.makedirs(UNKNOWN_DIR, exist_ok=True)
last_saved_time = 0
SAVE_INTERVAL = 2  # seconds

# For logging unknowns
log_file = "unknown_log.csv"
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Filename"])

print("[INFO] Starting detection...")

frame_count = 0
process_every_n_frames = 3  # Process every 3rd frame for better performance

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLO object detection (every frame)
    results = model.predict(source=frame, verbose=False, stream=False)[0]

    # Draw YOLO boxes
    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            if conf > 0.4:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Face recognition (only every few frames for better performance)
    if frame_count % process_every_n_frames == 0:
        # Resize for face recognition (faster)
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            # Draw face box and label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Save cropped unknown face image every SAVE_INTERVAL seconds
            if name == "Unknown":
                current_time = time.time()
                if current_time - last_saved_time >= SAVE_INTERVAL:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{timestamp}.jpg"
                    filepath = os.path.join(UNKNOWN_DIR, filename)
                    face_crop = frame[top:bottom, left:right]
                    cv2.imwrite(filepath, face_crop)
                    with open(log_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([timestamp, filename])
                    last_saved_time = current_time

    frame_count += 1

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
