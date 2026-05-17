# config.py

# Image settings
IMG_SIZE = 160

# Paths
DATA_PATH = "data/images"
LANDMARK_PATH = "data/landmarks"
MODEL_PATH = "models/all_sign_model.h5"

# Classes: 1–9, A–Z + SPACE + DELETE
CLASSES = (
    [str(i) for i in range(1, 10)] +
    [chr(i) for i in range(ord('A'), ord('Z') + 1)] +
    ["SPACE", "DELETE"]
)

NUM_CLASSES = len(CLASSES)

# Training params
BATCH_SIZE = 16
EPOCHS = 10
VALIDATION_SPLIT = 0.3