import cv2 as cv
import os

# Constants
WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480
FACE_BOX = (160, 100, 320, 320)  # x, y, w, h

# Person name input
person_name = input("Enter your name: ").strip()
save_dir = os.path.join('Learners/src/demo/authentication/face_recognition/Faces/train', person_name)
os.makedirs(save_dir, exist_ok=True)

# Webcam setup
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
cv.namedWindow("Face Registration", cv.WINDOW_NORMAL)
cv.moveWindow("Face Registration", 100, 100)

# Haar cascades
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

# Steps to complete
steps = [
    "Align your face in the box",
    "Move closer to the camera",
    "Move farther from the camera",
    "Blink your eyes",
    "Turn head right",
    "Turn head left"
]
step_index = 0
captured = 0

started = False

print("Press SPACE to begin registration.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    key = cv.waitKey(1) & 0xFF

    # Press SPACE to start
    if key == ord(' '):
        started = True

    # Draw guide box
    x_box, y_box, w_box, h_box = FACE_BOX
    cv.rectangle(frame, (x_box, y_box), (x_box + w_box, y_box + h_box), (255, 0, 0), 2)

    instruction = "Press SPACE to start"
    if started and step_index < len(steps):
        instruction = steps[step_index]
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (fx, fy, fw, fh) in faces:
            face_center = (fx + fw//2, fy + fh//2)

            if step_index == 0:
                if x_box < face_center[0] < x_box + w_box and y_box < face_center[1] < y_box + h_box:
                    filename = os.path.join(save_dir, f'{person_name}_{captured}.jpg')
                    cv.imwrite(filename, frame[fy:fy+fh, fx:fx+fw])
                    print(f"[Step 0 Captured] {filename}")
                    step_index += 1
                    captured += 1

            # Step 1: Move closer (face width > 180)
            elif step_index == 1 and fw > 180:
                filename = os.path.join(save_dir, f'{person_name}_{captured}.jpg')
                cv.imwrite(filename, frame[fy:fy+fh, fx:fx+fw])
                print(f"[Step 1 Captured: Moved closer] {filename}")
                step_index += 1
                captured += 1

            # Step 2: Move farther (face width < 120)
            elif step_index == 2 and fw < 120:
                filename = os.path.join(save_dir, f'{person_name}_{captured}.jpg')
                cv.imwrite(filename, frame[fy:fy+fh, fx:fx+fw])
                print(f"[Step 2 Captured: Moved farther] {filename}")
                step_index += 1
                captured += 1

            # Step 3: Blink detection (no eyes = blink)
            elif step_index == 3:
                roi_gray = gray[fy:fy+fh, fx:fx+fw]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                if len(eyes) == 0:
                    filename = os.path.join(save_dir, f'{person_name}_{captured}.jpg')
                    cv.imwrite(filename, frame[fy:fy+fh, fx:fx+fw])
                    print(f"[Step 3 Captured: Blinked] {filename}")
                    step_index += 1
                    captured += 1

            # Step 4: Turn head right (face left of center)
            elif step_index == 4 and face_center[0] < x_box + w_box//3:
                filename = os.path.join(save_dir, f'{person_name}_{captured}.jpg')
                cv.imwrite(filename, frame[fy:fy+fh, fx:fx+fw])
                print(f"[Step 4 Captured: Turned Right] {filename}")
                step_index += 1
                captured += 1

            # Step 5: Turn head left (face right of center)
            elif step_index == 5 and face_center[0] > x_box + 2*w_box//3:
                filename = os.path.join(save_dir, f'{person_name}_{captured}.jpg')
                cv.imwrite(filename, frame[fy:fy+fh, fx:fx+fw])
                print(f"[Step 5 Captured: Turned Left] {filename}")
                step_index += 1
                captured += 1

            break  # Only one face is enough

    elif step_index >= len(steps):
        instruction = "All actions completed. Press Q to quit."

    cv.putText(frame, instruction, (30, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv.imshow("Face Registration", frame)

    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()