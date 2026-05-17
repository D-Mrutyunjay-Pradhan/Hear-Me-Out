import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model
from Extra.utils import preprocess_image, normalize_landmarks
from Extra.config import *

app = Flask(__name__)

model = load_model(MODEL_PATH)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# STATES
sentence = ""
detected_text = ""
sentence_building = True
camera_on = True
last_label = ""
current_label = ""
counter = 0
cooldown = 10

# highlight control
highlight_frames = 0


def generate_frames():
    global sentence, detected_text, sentence_building
    global last_label, counter, camera_on, current_label
    global highlight_frames

    cap = cv2.VideoCapture(0)  # ✅ FIX: moved inside

    last_frame = None

    while True:
        success, frame = cap.read()
        if not success:
            break

        h, w, _ = frame.shape

        # CAMERA OFF → freeze + dim
        if not camera_on and last_frame is not None:
            frame = last_frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            cv2.putText(frame, "CAMERA OFF", (w//3, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        else:
            last_frame = frame.copy()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            label = ""

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    lm_list = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

                    landmarks = normalize_landmarks(lm_list)
                    img = preprocess_image(frame)

                    pred = model.predict([np.array([img]), np.array([landmarks])])
                    class_id = np.argmax(pred)
                    label = CLASSES[class_id]

                    # detection stability
                    if label == last_label:
                        counter += 1
                    else:
                        counter = 0

                    # ✅ ACCEPT LETTER
                    if counter > cooldown and label != "":
                        detected_text += label + " "
                        highlight_frames = 5   # ✅ green highlight duration

                        if sentence_building:

                            if label == "SPACE":
                                if not sentence.endswith(" "):
                                    sentence += " "

                            elif label == "DELETE":
                                if len(sentence) > 0:
                                    sentence = sentence[:-1]

                            else:
                                sentence += label

                        counter = 0

                    last_label = label
                    current_label = label

            # 🔥 SHOW LABEL ON CAMERA
            if label != "":
                box_w, box_h = 150, 80
                x = 30
                y = 60

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 2

                (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

                while text_w > box_w - 20:
                    font_scale -= 0.1
                    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

                overlay = frame.copy()
                cv2.rectangle(overlay, (x, y), (x+box_w, y+box_h), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

                text_x = x + (box_w - text_w) // 2
                text_y = y + (box_h + text_h) // 2

                # ✅ COLOR LOGIC
                if highlight_frames > 0:
                    color = (0, 255, 0)  # green
                    highlight_frames -= 1
                else:
                    color = (0, 255, 255)  # yellow

                cv2.putText(frame, label, (text_x, text_y),
                            font, font_scale, color,
                            thickness, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()  # ✅ release camera


# ROUTES
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/toggle_camera')
def toggle_camera():
    global camera_on
    camera_on = not camera_on
    return "OK"


@app.route('/toggle_sentence')
def toggle_sentence():
    global sentence_building
    sentence_building = not sentence_building
    return "OK"


@app.route('/clear')
def clear():
    global sentence, detected_text
    sentence = ""
    detected_text = ""
    return "OK"


@app.route('/get_sentence')
def get_sentence():
    return sentence


@app.route('/get_detected')
def get_detected():
    return detected_text


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)  # ✅ FIXED