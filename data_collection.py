import cv2
import os
import numpy as np
import mediapipe as mp
from Extra.utils import preprocess_image, normalize_landmarks
from Extra.config import DATA_PATH, LANDMARK_PATH, CLASSES

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

label = input("Enter label: ")

img_dir = os.path.join(DATA_PATH, label)
lm_dir = os.path.join(LANDMARK_PATH, label)

os.makedirs(img_dir, exist_ok=True)
os.makedirs(lm_dir, exist_ok=True)

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm_list = []
            for lm in hand_landmarks.landmark:
                lm_list.append([lm.x, lm.y, lm.z])

            landmarks = normalize_landmarks(lm_list)

            img = preprocess_image(frame)

            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                np.save(f"{lm_dir}/{count}.npy", landmarks)
                cv2.imwrite(f"{img_dir}/{count}.jpg", frame)
                count += 1
                print(f"Saved {count}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()