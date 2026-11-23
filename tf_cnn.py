"""
Train a CNN on MNIST and save it.
Run:  python tf_cnn.py
Creates: tf-cnn-model.h5 (+ .keras copy), result.png,
         assets/training_curves.png, assets/confusion_matrix.png,
         assets/images/digit_0..9.png (sample digits for load_model.py)
"""
import os
import numpy as np
from PIL import Image
import matplotlib
# (Agg backend: saves plots without opening windows)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

EPOCHS = 8
BATCH = 128

# ---------- 1. Load data ----------
os.makedirs("assets", exist_ok=True)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = (x_train.astype("float32") / 255.0)[..., None]   # (60000,28,28,1)
x_test = (x_test.astype("float32") / 255.0)[..., None]
print("Train:", x_train.shape, " Test:", x_test.shape)

# ---------- 2. Build model ----------
# Augmentation layers only act during training; they make the model
# more robust to hand-drawn digits that are slightly shifted/tilted.
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.RandomRotation(0.08),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomZoom(0.1),

    layers.Conv2D(32, 3, activation="relu"),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Conv2D(64, 3, activation="relu"),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# ---------- 3. Train ----------
history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH,
                    validation_split=0.1, verbose=2)

# ---------- 4. Evaluate ----------
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTEST ACCURACY: {acc * 100:.2f}%   TEST LOSS: {loss:.4f}")
model.save("tf-cnn-model.h5")
model.save("digit_model.keras")
print("Saved model -> tf-cnn-model.h5 and digit_model.keras")

# ---------- 5. Plots for the report/slides ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(history.history["accuracy"], label="train")
ax[0].plot(history.history["val_accuracy"], label="validation")
ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend()
ax[1].plot(history.history["loss"], label="train")
ax[1].plot(history.history["val_loss"], label="validation")
ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend()
plt.tight_layout(); plt.savefig("assets/training_curves.png", dpi=150); plt.close()

probs = model.predict(x_test, verbose=0)
preds = probs.argmax(axis=1)

cm = np.zeros((10, 10), dtype=int)
for t, p in zip(y_test, preds):
    cm[t, p] += 1
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues"); plt.colorbar()
plt.title(f"Confusion matrix (test acc {acc * 100:.2f}%)")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.xticks(range(10)); plt.yticks(range(10))
for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout(); plt.savefig("assets/confusion_matrix.png", dpi=150); plt.close()

fig, axes = plt.subplots(2, 8, figsize=(12, 3.6))
for a, i in zip(axes.ravel(), range(16)):
    a.imshow(x_test[i].squeeze(), cmap="gray"); a.axis("off")
    ok = preds[i] == y_test[i]
    a.set_title(f"P:{preds[i]} T:{y_test[i]}", color="green" if ok else "red", fontsize=9)
plt.tight_layout(); plt.savefig("result.png", dpi=150); plt.close()

# ---------- 6. Save one sample image per digit for load_model.py ----------
os.makedirs("assets/images", exist_ok=True)
for d in range(10):
    i = int(np.where(y_test == d)[0][0])
    img = (x_test[i].squeeze() * 255).astype("uint8")
    Image.fromarray(img).resize((140, 140), Image.NEAREST).save(f"assets/images/digit_{d}.png")
print("Saved result.png, assets/training_curves.png, assets/confusion_matrix.png, assets/images/")
