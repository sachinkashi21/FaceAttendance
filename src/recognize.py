import cv2
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import os

EMB_FILE = "embeddings.npy"
THRESHOLD = 0.6   # adjust between 0.35–0.45

# ---- Load DB ----
if not os.path.exists(EMB_FILE):
    print("No embeddings found. Run register_user.py first.")
    exit()

data = np.load(EMB_FILE, allow_pickle=True).item()
labels = data["labels"]
embeddings = data["embeddings"]

cap = cv2.VideoCapture(0)

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        reps = DeepFace.represent(
            frame,
            detector_backend="mtcnn",
            model_name="Facenet512",
            enforce_detection=True,
            align=True,
            normalization="Facenet"
        )

        for rep in reps:
            emb = np.array(rep["embedding"])
            box = rep["facial_area"]

            sims = cosine_similarity([emb], embeddings)[0]
            j = np.argmax(sims)

            if sims[j] > THRESHOLD:
                name = labels[j]
            else:
                name = "Unknown"

            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({sims[j]:.2f})",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2)

    except Exception:
        pass

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
