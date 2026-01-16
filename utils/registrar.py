import numpy as np
from deepface import DeepFace
import os

EMB_FILE = "embeddings.npy"
TARGET = 100

def register_face(frames, name):
    face_embeds = []

    for frame in frames:
        reps = DeepFace.represent(
            frame,
            detector_backend="mtcnn",
            model_name="Facenet512",
            enforce_detection=True,
            align=True,
            normalization="Facenet"
        )
        face_embeds.append(reps[0]["embedding"])

    avg_emb = np.mean(face_embeds, axis=0)

    if os.path.exists(EMB_FILE):
        data = np.load(EMB_FILE, allow_pickle=True).item()
        labels = data["labels"]
        embeddings = data["embeddings"]
    else:
        labels, embeddings = [], np.empty((0,512))

    labels.append(name)
    embeddings = np.vstack([embeddings, avg_emb])

    np.save(EMB_FILE, {"labels": labels, "embeddings": embeddings})
