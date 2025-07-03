import cv2 as cv
import os
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector
from blink_detection import is_blinking
from preprocess_embedding import preprocess_embedding

person_name = input("Enter your name: ").strip()
save_dir = os.path.join('Learners/src/authentication/face_recognition/Faces/train', person_name)
embedding_dir = os.path.join(save_dir, "embeddings")
os.makedirs(embedding_dir, exist_ok=True)

cap = cv.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)

ratio_list = []
counter = 0
captured = 0
blink_triggered = False

print("Blink to capture an image. Press 'q' to quit.")

while True:
    success, img = cap.read()
    if not success:
        break

    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        blinked, ratio_list, counter = is_blinking(face, detector, ratio_list, counter)

        if blinked:
            blink_triggered = True
        elif blink_triggered and not blinked:
            x_vals = [p[0] for p in face]
            y_vals = [p[1] for p in face]
            x_min, x_max = max(min(x_vals) - 10, 0), min(max(x_vals) + 10, img.shape[1])
            y_min, y_max = max(min(y_vals) - 10, 0), min(max(y_vals) + 10, img.shape[0])
            face_img = img[y_min:y_max, x_min:x_max]

            filename = os.path.join(save_dir, f"{person_name}_{captured}.jpg")
            cv.imwrite(filename, face_img)
            print(f"[Captured] {filename}")

            embedding = preprocess_embedding(np.array(face).flatten())
            emb_path = os.path.join(embedding_dir, f"{person_name}_{captured}.npy")
            np.save(emb_path, embedding)
            print(f"[Saved embedding] {emb_path}")

            captured += 1
            blink_triggered = False

    cv.putText(img, f"Blink to Capture | Captured: {captured}", (20, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv.imshow("Face Trainer", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()