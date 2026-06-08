# 🐕 CNN Dog Breed Classification (60 Breeds)

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-FF6F00.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Powered-D00000.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)


> An end-to-end Deep Learning pipeline leveraging **Transfer Learning (ResNet50V2)** to classify 60 distinct dog breeds from raw images, achieving robust real-world generalization.

## 📖 Project Overview

Classifying 60 visually similar dog breeds is a highly complex computer vision task due to immense intra-class variation (different colors/poses of the same breed) and minimal inter-class variation (similar-looking breeds like Malamutes and Huskies). 

To solve this, I engineered a robust **Convolutional Neural Network (CNN)** pipeline utilizing the **ResNet50V2** architecture pre-trained on ImageNet. The project relies heavily on aggressive data augmentation via `ImageDataGenerator` to prevent overfitting and ensure the model performs accurately on unseen, real-world images.

### 🌟 Key Technical Highlights
* **Two-Phase Transfer Learning:** Initially froze the base ResNet50V2 model to train a custom dense classification head, followed by unfreezing the top 30 layers for fine-tuned feature extraction.
* **Aggressive Augmentation:** Implemented dynamic `ImageDataGenerator` pipelines (rotation, shift, shear, zoom, flip) to mathematically expand the training dataset.
* **Optimized Callbacks:** Utilized `EarlyStopping` to prevent overfitting, `ModelCheckpoint` to save the best weights, and `ReduceLROnPlateau` to dynamically decay the RMSprop learning rate during convergence.
* **Final Performance:** Achieved a highly robust **80.5% validation accuracy** across 60 dense classes.

---

## 🏗️ Repository Architecture

```text
dog_breed_classification/
├── data/                    # Dataset directory (train/val splits)
├── model/                   # Serialized model artifacts (.h5)
├── outputs/                 # Automatically generated training history & confusion matrix plots
├── src/                     # Core source code
│   ├── train.py             # Advanced CNN training script with two-phase Transfer Learning
│   └── predict.py           # Inference script leveraging OpenCV to validate new images
├── requirements.txt         # Project dependencies
└── README.md                # Documentation
```

---

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<YOUR_USERNAME>/dog_breed_classification.git
cd dog_breed_classification
```

### 2. Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Dataset Acquisition
This pipeline was engineered to train on a 60-breed subset of the Kaggle **[Dog Breed Identification Dataset](https://www.kaggle.com/c/dog-breed-identification/data)**. 
To train from scratch, download the dataset and organize the images into `data/train/` and `data/val/` using the standard Keras format (one sub-folder per breed).

---

## 💻 Usage

### Training the Model
To initiate the two-phase training pipeline:
```bash
python -m src.train
```

### Inference / Predicting New Images
To validate the model's performance on a completely new, downloaded image of a dog:
```bash
python -m src.predict --image path/to/your/dog_image.jpg --model model/resnet50v2_dog_breed.h5
```
*The script utilizes OpenCV to dynamically resize and normalize the image, feeds it through the ResNet50V2 pipeline, and outputs the top predicted breed alongside a confidence percentage.*

---

## 📈 Performance & Results

### Training History
By utilizing `ReduceLROnPlateau` and aggressive augmentation, the model successfully generalized without severe overfitting, converging at an **80.5% Validation Accuracy**.

*(View the generated plots in the `outputs/` directory)*

- **Training & Validation Accuracy:** Shows the impact of Phase 2 (unfreezing the base model) on breaking the accuracy plateau.
- **Confusion Matrix:** Demonstrates the model's strong diagonal predictive power across all 60 breeds.

---


