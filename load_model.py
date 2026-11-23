"""
Load the trained model and predict digits from image files.
Run:  python load_model.py                 (predicts every image in assets/images/)
      python load_model.py my_digit.png    (predicts one image)
Works with white digit on black, or dark digit on light paper (auto-inverts).
"""
import sys, glob
import numpy as np
from PIL import Image, ImageOps
from common import load_model, preprocess

files = sys.argv[1:] or sorted(glob.glob("assets/images/*.png") + glob.glob("assets/images/*.jpg"))
if not files:
    raise SystemExit("No images found. Pass an image path, or run tf_cnn.py first.")

model = load_model()
for f in files:
    img = Image.open(f).convert("L")
    if np.array(img).mean() > 127:
        img = ImageOps.invert(img)
    x = preprocess(img)
    if x is None:
        print(f"{f}: no digit found")
        continue
    probs = model.predict(x, verbose=0)[0]
    best = int(probs.argmax())
    print(f"{f}: predicted {best} (confidence {probs[best] * 100:.1f}%)")
