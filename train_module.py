# train.py

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Concatenate, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

from Extra.config import *
from Extra.data_generator import DataGenerator
from Extra.callbacks import get_callbacks


# =========================
# CPU OPTIMIZATION
# =========================
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)


# =========================
# BUILD MODEL
# =========================
def build_model():
    print("🧠 Building model...")

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    base_model.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)   # 🔥 reduced size
    x = Dropout(0.5)(x)

    lm_input = Input(shape=(63,))
    lm_branch = Dense(64, activation='relu')(lm_input)  # 🔥 lighter
    lm_branch = Dropout(0.3)(lm_branch)

    combined = Concatenate()([x, lm_branch])
    z = Dense(128, activation='relu')(combined)  # 🔥 reduced
    z = Dropout(0.5)(z)

    output = Dense(NUM_CLASSES, activation='softmax')(z)

    model = Model(inputs=[base_model.input, lm_input], outputs=output)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',  # 🔥 faster than one-hot
        metrics=['accuracy']
    )

    print("✅ Model ready!")
    return model


# =========================
# TRAIN
# =========================
def train():
    print("🚀 Starting CPU-optimized training...\n")

    train_gen = DataGenerator(batch_size=BATCH_SIZE)

    model = build_model()
    callbacks = get_callbacks(MODEL_PATH)

    print("\n🔥 Training started...\n")

    model.fit(
        train_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
        workers=4,
        use_multiprocessing=True
    )

    model.save(MODEL_PATH)
    print("\n💾 Model saved!")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    train()