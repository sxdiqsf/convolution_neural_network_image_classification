# CNN Image Classifier for Cats and Dogs
This repository contains a Convolutional Neural Network (CNN) model built with Keras to classify images of cats and dogs. The model includes several convolutional, pooling, and dense layers and is trained on a small dataset. The classifier can be used to predict whether a new image is a cat or a dog.

# Table of Contents
## Overview
Requirements
Model Architecture
Data Preparation
Training the Model
Making Predictions
Example Output

## Overview
This CNN model was developed using Keras, a deep learning library. It consists of convolutional, pooling, dropout, and dense layers to achieve high accuracy in classifying images of cats and dogs. The model trains on an example dataset and can then make predictions on new images.

## Requirements
Python 3.x
TensorFlow 2.x
Keras
NumPy
Install the requirements using:

## Model Architecture
Convolutional Layers: Three convolutional layers with ReLU activation.
Pooling Layers: MaxPooling layers after each convolution to reduce dimensionality.
Dropout Layers: Dropout to prevent overfitting.
Flattening Layer: Flattens the pooled feature maps.
Dense Layers: Fully connected layers, including a sigmoid output layer for binary classification.
Data Preparation
Organize your dataset in two directories: train and test. Each directory should contain two subfolders representing each class (e.g., cat and dog).

## The model expects images to be located in:

train/
  ├── cat/
  └── dog/
test/
  ├── cat/
  └── dog/
  
## Training the Model
Instantiate the CNN architecture and compile it using the Adam optimizer and binary cross-entropy loss.
Use Keras's ImageDataGenerator to preprocess the images and perform data augmentation.
Fit the model to the training set, specifying epochs and batch size.


## License
This project is licensed under the MIT License.
