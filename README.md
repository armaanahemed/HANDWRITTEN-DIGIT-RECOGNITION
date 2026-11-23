# Handwritten Digit Recognition

A Convolutional Neural Network (CNN) trained on the MNIST dataset that recognizes
handwritten digits (0-9), with a **live demo** where you draw a digit with the mouse
and the model predicts it instantly.

Built with Python, TensorFlow/Keras, Tkinter, NumPy and Pillow.

## Features
- CNN trained on 60,000 MNIST images (about 99% test accuracy)
- Live drawing window with the predicted digit and confidence bars for all 10 digits
- MNIST-style preprocessing (crop, scale, center) so hand-drawn digits work well
- Predict from an image file

## Project structure
| File | Purpose |
|---|---|
| `tf_cnn.py` | Builds, trains, evaluates and saves the CNN |
| `tf_cnn.ipynb` | Same training code as a Jupyter notebook |
| `tf-cnn-model.h5` | Trained model (created by `tf_cnn.py`) |
| `load_model.py` | Load the model and predict digits from image files |
| `app.py` | Live demo: draw a digit and get a prediction |
| `common.py` | Preprocessing and model loading helpers |
| `result.png` | Sample predictions on test images |
| `assets/` | Sample digit images, training curves, confusion matrix |
| `requirements.txt` | Python libraries needed |

## Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```
Use Python 3.9 - 3.12.

## Usage
1. Train the model (about 5-10 minutes, needs internet once to download MNIST):
   ```bash
   python tf_cnn.py
   ```
2. Run the live demo:
   ```bash
   python app.py
   ```
3. Predict from images (all of `assets/images/` if no path is given):
   ```bash
   python load_model.py my_digit.png
   ```

## Model architecture
Input (28x28x1) -> data augmentation (rotation, shift, zoom) ->
Conv2D(32) x2 -> MaxPooling -> Dropout ->
Conv2D(64) x2 -> MaxPooling -> Dropout ->
Flatten -> Dense(128) -> Dropout -> Dense(10, softmax)

Optimizer: Adam | Loss: sparse categorical cross-entropy | Epochs: 8

## Results
Test accuracy: **XX.XX%** (replace with your number from tf_cnn.py)

![Sample predictions](result.png)
![Training curves](assets/training_curves.png)
![Confusion matrix](assets/confusion_matrix.png)

## Credits
Inspired by the reference project
[aakashjhawar/handwritten-digit-recognition](https://github.com/aakashjhawar/handwritten-digit-recognition).
This version is rewritten for modern TensorFlow/Python and adds a live drawing demo.

## Team
- Name 1
- Name 2
- Name 3

## License
MIT - see [LICENSE](LICENSE).
