import cv2
import mediapipe as mp
import os
import numpy as np
import pickle

# Initialize mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection

DATASET_DIR = "dataset"
ENCODINGS_PATH = "encodings.pickle"

def get_face_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARNING] Could not read image: {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as detector:
        results = detector.process(img_rgb)
        if not results.detections:
            print(f"[WARNING] No face detected in {image_path}")
            return None

        detection = results.detections[0]

        # Get bounding box
        bboxC = detection.location_data.relative_bounding_box
        h, w, _ = img.shape
        x1 = max(0, int(bboxC.xmin * w))
        y1 = max(0, int(bboxC.ymin * h))
        x2 = min(w, int((bboxC.xmin + bboxC.width) * w))
        y2 = min(h, int((bboxC.ymin + bboxC.height) * h))

        # Crop face safely
        face = img_rgb[y1:y2, x1:x2]
        if face.size == 0 or y2 <= y1 or x2 <= x1:
            print(f"[WARNING] Skipping invalid face in {image_path}")
            return None

        face = cv2.resize(face, (160, 160))  # fixed size for embedding
        return face.flatten() / 255.0  # normalize

# Process dataset
known_encodings = []
known_names = []

for filename in os.listdir(DATASET_DIR):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        name = os.path.splitext(filename)[0]
        path = os.path.join(DATASET_DIR, filename)
        embedding = get_face_embedding(path)
        if embedding is not None:
            known_encodings.append(embedding)
            known_names.append(name)
            print(f"[INFO] Processed {name}")
        else:
            print(f"[WARNING] Skipped {name}")

# Save encodings
data = {"encodings": known_encodings, "names": known_names}
with open(ENCODINGS_PATH, "wb") as f:
    pickle.dump(data, f)

print("[INFO] Encoding complete. Saved to encodings.pickle")
