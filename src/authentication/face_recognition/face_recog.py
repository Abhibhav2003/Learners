import face_recognition
import cv2
import os
import numpy as np

# Load encodings
data_path = os.path.join(os.path.dirname(__file__), "face_data")
known_encodings = []
known_names = []

for name in os.listdir(data_path):
    person_dir = os.path.join(data_path, name)
    for file in os.listdir(person_dir):
        if file.endswith(".npy"):
            enc = np.load(os.path.join(person_dir, file))
            known_encodings.append(enc)
            known_names.append(name)

# Function to convert face distance to match percentage
def face_confidence(face_distance, threshold=0.5):
    range_val = (1.0 - threshold)
    linear_val = (1.0 - face_distance) / (range_val * 2.0)
    return round(linear_val * 100, 2)

# Initialize webcam
cap = cv2.VideoCapture(0)
print("Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb_frame)
    if boxes:
        box = boxes[0]  # Take the first face only
        encoding = face_recognition.face_encodings(rgb_frame, [box])[0]

        distances = face_recognition.face_distance(known_encodings, encoding)
        best_match_index = np.argmin(distances)

        name = "Unknown"
        confidence = 0.0

        if distances[best_match_index] <= 0.5:
            name = known_names[best_match_index]
            confidence = face_confidence(distances[best_match_index])

        # Draw bounding box and label with confidence
        top, right, bottom, left = box
        label = f"{name} ({confidence}%)" if name != "Unknown" else name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()