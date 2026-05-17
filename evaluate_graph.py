# evaluate.py

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from Extra.data_generator import DataGenerator
from Extra.config import *

print("🔍 Loading model...")
model = load_model(MODEL_PATH)

print("📦 Loading data using generator...")
gen = DataGenerator(batch_size=32)

y_true = []
y_pred = []

print("🚀 Running predictions...\n")

for i in range(len(gen)):
    (X_img, X_lm), y = gen[i]

    preds = model.predict([X_img, X_lm], verbose=0)
    preds = np.argmax(preds, axis=1)

    y_true.extend(y)
    y_pred.extend(preds)

    # 🔥 progress
    if i % 10 == 0:
        print(f"Processed batch {i}/{len(gen)}")

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# =========================
# ACCURACY
# =========================
acc = accuracy_score(y_true, y_pred)
print("\n✅ Accuracy:", acc)

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=False, cmap="Blues",
            xticklabels=CLASSES,
            yticklabels=CLASSES)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()