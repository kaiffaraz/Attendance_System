import cv2
import mediapipe as mp
import pickle
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

ENCODINGS_PATH = "encodings.pickle"
INPUT_FOLDER = "input_photos"
ATTENDANCE_CSV = "attendance.csv"

# Load encodings
with open(ENCODINGS_PATH, "rb") as f:
    data = pickle.load(f)

known_encodings = np.array(data["encodings"])
known_names = data["names"]

# Initialize mediapipe face detector
mp_face_detection = mp.solutions.face_detection

# Process each input photo
for image_file in os.listdir(INPUT_FOLDER):
    if not image_file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    image_path = os.path.join(INPUT_FOLDER, image_file)
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    present = []
    unrecognized = []
    absent = known_names.copy()

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as detector:
        results = detector.process(img_rgb)

        if results.detections:
            idx_unrec = 1
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = img.shape
                x1, y1 = int(bboxC.xmin * w), int(bboxC.ymin * h)
                x2, y2 = int((bboxC.xmin + bboxC.width) * w), int((bboxC.ymin + bboxC.height) * h)

                # Extract face
                face = img_rgb[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                face = cv2.resize(face, (160, 160))
                embedding = face.flatten() / 255.0

                # Compare with known encodings
                sims = cosine_similarity([embedding], known_encodings)[0]
                best_idx = np.argmax(sims)
                best_score = sims[best_idx]

                if best_score > 0.75:
                    name = known_names[best_idx]
                    present.append(name)
                    if name in absent:
                        absent.remove(name)
                    color = (0, 255, 0)
                    label = name
                else:
                    name = f"Unrecognized_{idx_unrec}"
                    unrecognized.append(name)
                    idx_unrec += 1
                    color = (0, 0, 255)
                    label = name

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Save attendance results in neat format
    attendance_data = {
        "Date": [datetime.now().strftime("%Y-%m-%d")],
        "Time": [datetime.now().strftime("%H:%M:%S")],
        "Present": [", ".join(present)],
        "Absent": [", ".join(absent)],
        "Unrecognized": [", ".join(unrecognized)],
        "Total Present": [len(present)],
        "Total Absent": [len(absent)],
        "Total Unrecognized": [len(unrecognized)]
    }

    attendance_df = pd.DataFrame(attendance_data)

    if not os.path.exists(ATTENDANCE_CSV):
        attendance_df.to_csv(ATTENDANCE_CSV, index=False)
    else:
        attendance_df.to_csv(ATTENDANCE_CSV, mode="a", header=False, index=False)

    print(f"[INFO] Processed {image_file} → Attendance saved.")
    print(attendance_df.to_string(index=False))  # Optional: print neatly in console

    # Show result
    cv2.imshow(f"Attendance - {image_file}", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
