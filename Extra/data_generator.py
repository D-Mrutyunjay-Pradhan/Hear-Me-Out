# data_generator.py

import os
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
from Extra.utils import load_image
from Extra.config import *

class DataGenerator(Sequence):
    def __init__(self, batch_size=BATCH_SIZE, mode='train'):
        self.batch_size = batch_size
        self.samples = []

        for idx, label in enumerate(CLASSES):
            img_dir = os.path.join(DATA_PATH, label)
            lm_dir = os.path.join(LANDMARK_PATH, label)

            if not os.path.exists(img_dir):
                continue

            for file in os.listdir(img_dir):
                if file.endswith(".jpg"):
                    img_path = os.path.join(img_dir, file)
                    lm_path = os.path.join(lm_dir, file.replace(".jpg", ".npy"))

                    if os.path.exists(lm_path):
                        self.samples.append((img_path, lm_path, idx))

        print(f"✅ Total samples found: {len(self.samples)}")

    def __len__(self):
        return len(self.samples) // self.batch_size

    def __getitem__(self, index):
        batch = self.samples[index * self.batch_size:(index + 1) * self.batch_size]

        X_img = []
        X_lm = []
        y = []

        for img_path, lm_path, label in batch:
            img = load_image(img_path)
            if img is None:
                continue

            lm = np.load(lm_path)

            X_img.append(img)
            X_lm.append(lm)
            y.append(label)

        return [np.array(X_img), np.array(X_lm)], np.array(y)