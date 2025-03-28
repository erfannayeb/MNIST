# MNIST Classification with Multiple Models

This repository contains implementations of various machine learning and deep learning models for classifying the MNIST handwritten digits dataset. The models include traditional machine learning algorithms as well as deep neural networks.

### Dataset

The models are trained and tested on the MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits (0-9).

### Implemented Models

The following models are implemented in the Jupyter Notebook:
 
**1. Dense Neural Network (Fully Connected)**

A simple feedforward neural network with multiple dense layers.

Uses ReLU activation and softmax for classification.

Includes dropout for regularization.

**2. Convolutional Neural Network (CNN)**

A standard CNN architecture with Conv2D and MaxPooling layers.

Uses ReLU activation and softmax for classification.

Trained with Adam optimizer and categorical cross-entropy loss.

**3. Modern CNN with Batch Normalization and Dropout**

An improved CNN model incorporating BatchNormalization and Dropout layers.

Helps improve model generalization and stability.

**4.Support Vector Machine (SVM)**

A classical machine learning algorithm for classification.

Uses pixel values as input features.

Implements kernel-based decision boundaries.


**5. Random Forest (RF)**

A traditional machine learning model based on decision trees.

Trained on pixel values as features.

Provides baseline performance for comparison with deep learning models.


**6. VGG16**

A deep convolutional network (VGG16) adapted for MNIST.

Uses pre-trained weights and fine-tunes the model for digit classification.


**7. Vision Transformer (ViT)**

A transformer-based model for image classification.

Uses self-attention mechanisms to capture global dependencies in images.

Pre-trained ViT models are fine-tuned on MNIST.

**8. ResNet18**

A deep residual network (ResNet18) adapted for MNIST.

Uses transfer learning from ImageNet weights.

Includes fully connected layers for classification.

###Installation and Requirements

To run the notebook, install the required dependencies:

pip install tensorflow torch transformers scikit-learn matplotlib numpy

### Usage

Clone the repository:

git clone https://github.com/erfannayeb/MNIST.git
cd MNIST

Open the Jupyter Notebook:

jupyter notebook MNIST.ipynb

Run the notebook and train the models.

## Results

Each model's accuracy and loss curves are plotted for comparison. The deep learning models achieve higher accuracy compared to traditional machine learning models.
### Accuracy Comparison of Models

| Model                                         | Accuracy (%) |
|-----------------------------------------------|--------------|
| Dense Neural Network (Fully Connected)        | 99.2         |
| Convolutional Neural Network (CNN)            | 99.7         |
| Modern CNN with Batch Normalization & Dropout | 99.4         |
| Support Vector Machine (SVM)                  | 96.7         |
| Random Forest (RF)                            | 97.3         |
| VGG16 (Transfer Learning)                     | 97.0         |
| VGG16 (Fine-Tuning)                           | 99.6         |
| Vision Transformer (ViT)                      | 88.0         |
| ResNet18 (Transfer Learning)                  | 94.1         |
| ResNet18 (Fine_Tuning_)                       | 94.1         |
