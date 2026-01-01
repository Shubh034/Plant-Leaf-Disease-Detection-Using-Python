# Plant Leaf Disease Detection using Python

## Overview
This project is a **Plant Leaf Disease Detection system** implemented in **Python**.  
It uses **image processing**, **feature extraction**, **machine learning**, and **deep learning** techniques to detect and classify diseases in rice leaves.  

The system can classify leaves into three categories:  
1. **Bacterial Leaf Blight**  
2. **Brown Spot**  
3. **Leaf Smut**

---

## Features

### 1. Image Preprocessing & Feature Extraction
- Reads images from directories corresponding to each disease class
- Resizes images to a fixed size
- Extracts **HOG (Histogram of Oriented Gradients)** features for classical ML models
- Normalizes features using standardization
- Reduces feature dimensionality using **PCA** for large datasets

---

### 2. Machine Learning Models
- **Logistic Regression**
- **Random Forest Classifier**
- **Naive Bayes Classifier**
- **Decision Tree Classifier**
- **Support Vector Machine (SVM)**

Each model is trained and tested to evaluate accuracy, with training and testing scores displayed, along with:
- Confusion Matrix  
- Classification Report

---

### 3. Deep Learning Model (CNN)
- **Convolutional Neural Network** implemented using **TensorFlow/Keras**
- Architecture:
  - 3 Convolutional layers with increasing filters (16, 32, 64)
  - MaxPooling layers after each Conv layer
  - Flatten layer followed by Dense layers
  - Output layer with 3 neurons (one for each class)
- Preprocesses images by scaling pixel values to `[0,1]`
- Trains the model for **30 epochs** on the dataset
- Predicts class of new images

---

### 4. Prediction
- Load a test image
- Resize and scale it
- Predict disease class using the trained CNN
- Displays the image and predicted label
- Labels mapping: [0] Bacterial Leaf Blight
                  [1] Brown Spot
                  [2] Leaf Smut

---

## Directory Structure

rice_leaf_diseases/
│
├── 1/ # Bacterial Leaf Blight images
├── 2/ # Brown Spot images
├── 3/ # Leaf Smut images
├── features.csv # Extracted HOG features (optional)
└── final.ipynb # Main code (if separated)

---

## Installation of Required Libraries

### Prerequisites
- Python 3.x installed
- Optionally, **Anaconda** for environment management

### Required Libraries
Install using pip:

```bash
pip install opencv-python
pip install numpy
pip install pandas
pip install scikit-image
pip install matplotlib
pip install scikit-learn
pip install tensorflow
pip install keras

To run the project, execute the script: final.ipynb


- Predict disease class using the trained CNN
- Displays the image and predicted label
- Labels mapping:
