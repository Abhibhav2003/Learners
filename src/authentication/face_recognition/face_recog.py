import cv2 as cv
import numpy as np
import os
from cvzone.FaceMeshModule import FaceMeshDetector
from sklearn.metrics.pairwise import cosine_similarity
from preprocess_embedding import preprocess_embedding

data_path = "Learners/src/authentication/face_recognition/Faces/train"
known_embeddings = []
labels = []

for person_name in os.listdir(data_path):
    embedding_dir = os.path.join(data_path, person_name, "embeddings")
    if not os.path.isdir(embedding_dir):
        continue

    for emb_file in os.listdir(embedding_dir):
        if emb_file.endswith(".npy"):
            emb = np.load(os.path.join(embedding_dir, emb_file))
            known_embeddings.append(preprocess_embedding(emb))
            labels.append(person_name)


cap = cv.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
print("Recognition started. Press 'q' to quit.")

def recognize(embedding):
    embedding = preprocess_embedding(embedding)

    best_score = -1
    best_match = "Unknown"

    for ref_emb, label in zip(known_embeddings, labels):
        score = cosine_similarity([embedding], [ref_emb])[0][0]
        if score > best_score:
            best_score = score
            best_match = label

    return best_match if best_score > 0.90 else "Unknown"  

while True:
    success, img = cap.read()
    if not success:
        break

    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        embedding = np.array(face).flatten()
        name = recognize(embedding)

        x_vals = [p[0] for p in face]
        y_vals = [p[1] for p in face]
        x_min, x_max = max(min(x_vals) - 10, 0), min(max(x_vals) + 10, img.shape[1])
        y_min, y_max = max(min(y_vals) - 10, 0), min(max(y_vals) + 10, img.shape[0])

        cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv.putText(img, f"{name}", (x_min, y_min - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv.imshow("Face Recognition", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()