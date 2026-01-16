import cv2
import numpy as np
from deepface import DeepFace
import os

EMB_FILE = "embeddings.npy"
TARGET = 100   # number of embeddings to average

def load_db():
    if not os.path.exists(EMB_FILE):
        return [], np.empty((0,512))
    data = np.load(EMB_FILE, allow_pickle=True).item()
    return data["labels"], data["embeddings"]

def save_db(labels, embeddings):
    np.save(EMB_FILE, {"labels": labels, "embeddings": embeddings})
    print("Saved database")

labels, embeddings = load_db()

print("Enter name to register (or blank to exit): ")
name = input().strip()

if name == "":
    print("No name entered. Exiting.")
    exit()

print(f"Capturing embeddings for {name}")

cap = cv2.VideoCapture(0)

face_embeds = []

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

        emb = np.array(reps[0]["embedding"])
        face_embeds.append(emb)

        print(f"{len(face_embeds)}/{TARGET}", end="\r")

    except Exception:
        pass

    cv2.imshow("Register User", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if len(face_embeds) >= TARGET:
        break

cap.release()
cv2.destroyAllWindows()

if len(face_embeds) == 0:
    print("No embeddings collected. Try again.")
    exit()

avg_emb = np.mean(face_embeds, axis=0)

labels.append(name)
embeddings = np.vstack([embeddings, avg_emb])

save_db(labels, embeddings)

print(f"\nStored embedding for {name}")
