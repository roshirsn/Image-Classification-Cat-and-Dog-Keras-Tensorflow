# Cat and Dog Classification Using Keras and TensorFlow

This project is an image classification task that distinguishes between cats and dogs using a Convolutional Neural Network (CNN) built with Keras and TensorFlow. The model is trained on a dataset sourced from Kaggle's Cat and Dog Classification dataset.

## Overview

The objective of this project is to build a CNN that can accurately classify images of cats and dogs. The model processes images and predicts whether the input image is of a cat or a dog.

## Dataset

check out kaggle 

### Dataset Structure

The dataset is organized as follows: 
    base_dir: input_csv, input_test.csv, labels.csv, labels_test.csv

## Model Architecture

The model architecture is a Sequential CNN consisting of the following layers:

1. Convolutional layers for feature extraction
2. MaxPooling layers to down-sample the feature maps
3. Fully connected layers for classification
4. Output layer with a softmax activation function to produce probabilities for the two classes (cat or dog)

## Installation

To run this project, you need to have Python installed along with the following libraries:

- TensorFlow
- Keras
- NumPy
- Matplotlib
- Pandas

You can install the required libraries using pip:

```bash
pip install tensorflow keras numpy matplotlib pandas
