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
2. Paste `dla_proj.py` into a notebook (cells are marked with `# CELL n` comments) or upload it as a notebook and run all cells.

Outputs (`.keras` checkpoints, plots) are written to `/content/`.

## Note on the comparison

The final comparison cell uses hard-coded results for the custom CNN (epoch 6 checkpoint: val acc 0.5833, test acc 0.6667) and measures MobileNetV2 live. Rerun the CNN evaluation and update those constants if you retrain it.

## Requirements

`tensorflow`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `opencv-python-headless`, `Pillow`, `imagehash`, `kaggle`
