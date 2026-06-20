# Pneumonia Detection from Chest X-Rays using CNN

A deep learning project that classifies chest X-ray images as **NORMAL** or **PNEUMONIA** using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

## Overview

Pneumonia is a common and serious lung infection that is typically diagnosed by a radiologist reading a chest X-ray. This project trains a CNN to automatically classify chest X-rays, serving as a proof-of-concept for AI-assisted diagnostic support.

## Dataset

This project uses the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset, which contains labeled chest X-ray images split into training and test sets.

Expected directory structure:

```
chest_xray/
├── train/
│   ├── PNEUMONIA/
│   └── NORMAL/
└── test/
    ├── PNEUMONIA/
    └── NORMAL/
```

## Features

- End-to-end image classification pipeline using `ImageDataGenerator` for efficient batch loading
- Train/validation split with on-the-fly data augmentation (rotation, shifts, zoom, horizontal flip)
- Class weighting to handle dataset imbalance between NORMAL and PNEUMONIA classes
- Custom CNN architecture trained from scratch
- Training callbacks for early stopping, checkpointing, and adaptive learning rate
- Evaluation via classification report and confusion matrix on a held-out test set
- Trained model persisted to disk for reuse

## Requirements

```
numpy
pandas
tensorflow
scikit-learn
pillow
```

Install dependencies:

```bash
pip install numpy pandas tensorflow scikit-learn pillow
```

## Model Architecture

```
Input(150, 150, 3)
Conv2D(32, 3x3, relu)      -> MaxPooling2D(2, 2)
Conv2D(64, 3x3, relu)      -> MaxPooling2D(3, 3)
Conv2D(128, 3x3, relu)     -> MaxPooling2D(3, 3)
Flatten
Dropout(0.5)
Dense(128, relu)
Dense(1, sigmoid)
```

Optimizer: `Adam`
Loss: `binary_crossentropy`
Metric: `accuracy`

## Project Structure

| Step | Description |
|---|---|
| 1. Data loading | Builds a DataFrame of image filenames and labels from the training directory |
| 2. Data generators | Creates augmented training and non-augmented validation generators with an 80/20 split |
| 3. Class weights | Computes balanced class weights to address class imbalance |
| 4. Model definition | Defines and compiles the CNN |
| 5. Callbacks | EarlyStopping, ModelCheckpoint, ReduceLROnPlateau |
| 6. Training | Fits the model on the training data, validating each epoch |
| 7. Test data loading | Builds a DataFrame of image filenames and labels from the test directory |
| 8. Test generator | Loads test images for evaluation |
| 9. Evaluation | Generates a classification report and confusion matrix on the test set |
| 10. Model saving | Saves the trained model to disk |

## Usage

1. Download the dataset and update `train_dir` and `test_dir` to point to your local copy.
2. Open the notebook in Jupyter and run all cells top to bottom.
3. Review training curves and the final classification report / confusion matrix.
4. The trained model is saved as `pneumonia_cnn_model.keras`, with the best checkpoint saved as `best_pneumonia_model.keras` during training.

## Results

Evaluation metrics (precision, recall, F1-score, accuracy) are printed at the end of the notebook, along with a confusion matrix summarizing performance on the held-out test set.

## Future Improvements

- Incorporate transfer learning (e.g. EfficientNetB0, MobileNetV2) for higher accuracy with less training data
- Add BatchNormalization layers for faster, more stable convergence
- Experiment with grayscale input, since X-ray images carry no color information
- Apply test-time augmentation or model ensembling for more robust predictions
- Deploy the trained model behind a simple inference API or web interface

## License

This project is intended for educational and research purposes. It is not a certified medical diagnostic tool and should not be used for clinical decision-making.
