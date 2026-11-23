# Plant Disease Classification

Image classification of plant leaf diseases, comparing a custom CNN against a transfer-learning model (MobileNetV2) in TensorFlow/Keras.

## Dataset

[Plant Disease Recognition Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset) from Kaggle, split into `Train`, `Validation` and `Test` folders. Classes are discovered from the training folder names.

## What the script does

1. Downloads the dataset via the Kaggle API and inspects the splits, class balance and image integrity.
2. Trains a **custom CNN** (4 conv blocks, global average pooling, dropout) with built-in augmentation.
3. Trains **MobileNetV2** (ImageNet weights): first with a frozen base, then fine-tunes the last 20 layers at a low learning rate (BatchNorm layers kept frozen).
4. Evaluates both with accuracy, precision/recall/F1, classification reports and confusion matrices, and plots a side-by-side comparison.

Common settings: 224x224 input, batch size 32, Adam (3e-4; 1e-5 for fine-tuning), early stopping, LR reduction on plateau, seed 42.

## Running

The script is exported from a Google Colab notebook (`!pip` / `!kaggle` shell lines, `google.colab.userdata`), so run it on Colab with a GPU runtime:

1. Add your Kaggle token as a Colab secret named `KAGGLE_API_TOKEN`.
2. Upload `DLA_Proj.ipynb` to Colab and run all cells (`dla_proj.py` is the same code exported as a script).

Outputs (`.keras` checkpoints, plots) are written to `/content/`.

## Results

Dataset: 3 classes (Healthy, Powdery, Rust); 1,322 train / 60 validation / 150 test images, no corrupted files.

| Model | Val acc | Test acc | Test precision | Test recall | Test F1 |
|---|---|---|---|---|---|
| Custom CNN (best epoch 6) | 0.5833 | 0.6667 | 0.7021 | 0.6667 | 0.6654 |
| MobileNetV2, frozen base | 0.9500 | 0.9733 | 0.9753 | 0.9733 | 0.9733 |
| **MobileNetV2, fine-tuned** | **0.9833** | **0.9800** | 0.9811 | 0.9800 | 0.9800 |

Fine-tuned MobileNetV2 beats the custom CNN by about 31 points of test accuracy. The CNN struggles most with Rust (validation recall 0.30); the fine-tuned MobileNetV2 classifies every Rust test image correctly, and its only test errors are a few Powdery leaves predicted as Healthy.

The validation set is small (60 images, about 1.7 points per image), so the gain from fine-tuning (0.950 to 0.983) is 2 images and should not be over-read.

The full run with outputs is in `DLA_Proj.ipynb`. The CNN figures in the final comparison cell are hard-coded from its epoch-6 evaluation; update them if you retrain it.

## Requirements

`tensorflow`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `opencv-python-headless`, `Pillow`, `imagehash`, `kaggle`
