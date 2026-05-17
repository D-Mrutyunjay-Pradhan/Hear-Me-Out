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

            pred = model.predict([np.array([img]), np.array([landmarks])])
            class_id = np.argmax(pred)
            label = CLASSES[class_id]

            cv2.putText(frame, label, (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()