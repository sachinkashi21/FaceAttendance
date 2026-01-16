import cv2
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

EMB_FILE = "embeddings.npy"
THRESHOLD = 0.6

data = np.load(EMB_FILE, allow_pickle=True).item()
labels = data["labels"]
embeddings = data["embeddings"]

def recognize_and_draw(frame):
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
        idx = np.argmax(sims)

        if sims[idx] > THRESHOLD:
            name = labels[idx]
        else:
            name = "Unknown"

        x, y, w, h = box["x"], box["y"], box["w"], box["h"]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{name} ({sims[idx]:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    return frame, name
