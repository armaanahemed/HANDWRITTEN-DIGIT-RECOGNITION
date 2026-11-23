"""Shared helpers: model loading and MNIST-style preprocessing."""
import numpy as np
from PIL import Image

MODEL_PATHS = ["tf-cnn-model.h5", "digit_model.keras"]


def preprocess(img):
    """
    Convert a PIL image (white digit on black background, any size)
    into a (1, 28, 28, 1) float array exactly the way MNIST digits look:
    cropped, scaled to fit 20x20, then centred by centre-of-mass in 28x28.
    Returns None if the image is empty.
    """
    img = img.convert("L")
    arr = np.array(img, dtype=np.uint8)
    if arr.max() < 30:                      # nothing drawn
        return None

    # 1. crop to the bounding box of the digit
    ys, xs = np.where(arr > 30)
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # 2. scale so the longest side is 20 px (keep aspect ratio)
    h, w = arr.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    small = Image.fromarray(arr).resize((new_w, new_h), Image.LANCZOS)

    # 3. paste into the middle of a 28x28 canvas
    canvas = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = np.array(small, dtype=np.float32)

    # 4. shift so the centre of mass is at the image centre (as in MNIST)
    total = canvas.sum()
    cy = (canvas.sum(axis=1) * np.arange(28)).sum() / total
    cx = (canvas.sum(axis=0) * np.arange(28)).sum() / total
    shift_y, shift_x = int(round(13.5 - cy)), int(round(13.5 - cx))
    canvas = np.roll(canvas, (shift_y, shift_x), axis=(0, 1))

    canvas /= 255.0
    return canvas.reshape(1, 28, 28, 1)


def load_model(path=None):
    import os
    import tensorflow as tf
    candidates = [path] if path else MODEL_PATHS
    for p in candidates:
        if os.path.exists(p):
            return tf.keras.models.load_model(p)
    raise SystemExit("Model file not found. Run:  python tf_cnn.py   first.")
