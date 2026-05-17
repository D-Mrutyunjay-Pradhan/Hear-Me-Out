# utils.py

import cv2
import numpy as np
from Extra.config import IMG_SIZE

def preprocess_image(frame):
    """
    Resize and normalize image
    """
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    return img


def normalize_landmarks(landmarks):
    """
    Normalize landmarks:
    - Centering
    - Scale normalization
    - Flatten to 63 values
    """
    landmarks = np.array(landmarks)

    # Centering (translation invariance)
    center = np.mean(landmarks, axis=0)
    landmarks = landmarks - center

    # Scale normalization (avoid division by zero)
    max_val = np.max(np.abs(landmarks))
    if max_val != 0:
        landmarks = landmarks / max_val

    return landmarks.flatten()


def load_image(path):
    """
    Safe image loading
    """
    img = cv2.imread(path)
    if img is None:
        return None
    return preprocess_image(img)