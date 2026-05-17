import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from Extra.utils import preprocess_image, normalize_landmarks
from Extra.config import *

model = load_model(MODEL_PATH)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

# 🔥 Features
sentence = ""
sentence_building = True   # Only controls sentence building
last_label = ""
cooldown = 20
counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    label = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm_list = []
            for lm in hand_landmarks.landmark:
                lm_list.append([lm.x, lm.y, lm.z])

            landmarks = normalize_landmarks(lm_list)
            img = preprocess_image(frame)

            pred = model.predict([np.array([img]), np.array([landmarks])])
            class_id = np.argmax(pred)
            label = CLASSES[class_id]

            # 🔥 Sentence building control
            if sentence_building:
                if label == last_label:
                    counter += 1
                else:
                    counter = 0

                if counter > cooldown:
                    sentence += label
                    counter = 0

            last_label = label

    # ================= UI =================

    # 🔥 Highlighted Prediction Box
    if label != "":
        cv2.rectangle(frame, (40, 20), (250, 90), (0, 0, 0), -1)  # background
        cv2.putText(frame, label, (60, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

    # 🔥 Sentence at Bottom
    cv2.rectangle(frame, (0, h - 80), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, "Sentence: " + sentence, (20, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 🔥 Mode Display
    status = "ON" if sentence_building else "OFF"
    cv2.putText(frame, f"Sentence Mode: {status}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    # 🔥 Controls
    if key == ord('q'):
        break
    elif key == ord('b'):  # toggle sentence building only
        sentence_building = not sentence_building
    elif key == ord('c'):  # clear sentence
        sentence = ""
    elif key == 32:  # space key
        sentence += " "

cap.release()
cv2.destroyAllWindows()