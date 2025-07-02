import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('Learners/src/demo/authentication/face_recognition/model/face_trained.yml')

people = np.load('Learners/src/demo/authentication/face_recognition/meta/people.npy')


cap = cv.VideoCapture(0)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces_rect:
        face_roi = gray[y:y+h, x:x+w]

        label, confidence = face_recognizer.predict(face_roi)

        match_percent = max(0, min(100, 100 - confidence))

        if confidence < 60:
            name = people[label]
        else:
            name = "Unknown"

        print(f"Detected: {name} | Match: {match_percent:.0f}%")

        cv.putText(frame, f'{name} ({match_percent:.0f}%)', (x, y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow('Live Face Recognition', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()