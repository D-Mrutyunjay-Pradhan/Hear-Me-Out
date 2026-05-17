# callbacks.py

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def get_callbacks(model_path):
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    return [early_stop, checkpoint]