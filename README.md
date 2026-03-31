# Bone-Fracture-Classification

An automated medical diagnostic pipeline utilizing deep learning to classify X-ray imagery into "Fractured" and "Non-Fractured" categories. This project leverages Convolutional Neural Networks (CNNs) to assist in medical imaging analysis.

## Project Overview
* **Accomplished:** A highly accurate diagnostic tool for automated bone fracture detection.
* **Measured by:** Training on a structured dataset of X-ray images, evaluating performance via precision, recall, and loss/accuracy curves.
* **By doing:** Designing a custom 6-layer Convolutional Neural Network (CNN) using TensorFlow and Keras, implementing image augmentation with `ImageDataGenerator`.

## Tech Stack
* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Image Processing:** PIL (Python Imaging Library), TensorFlow ImageDataGenerator
* **Data Visualization:** Matplotlib
* **Environment:** Google Colab / Jupyter Notebook

## Model Architecture
The project utilizes a Sequential CNN architecture optimized for binary image classification:
1. **Input Layer:** Rescaling and standardizing image inputs.
2. **Convolutional Layers (`Conv2D`):** Extracting spatial features from the X-ray images.
3. **Pooling Layers (`MaxPooling2D`):** Reducing dimensionality and computational load.
4. **Flatten Layer:** Converting 2D feature maps to a 1D vector.
5. **Dense Layers:** Fully connected layers for classification with Dropout to prevent overfitting.
6. **Output Layer:** A single neuron with a Sigmoid activation function for binary classification (Fractured vs. Non-Fractured).

## Dataset Structure
The model expects data organized in the following directory structure:
```text
dataset/
│
├── train/
│   ├── fractured/
│   └── nonfractured/
│
└── val/
    ├── fractured/
    └── nonfractured/
