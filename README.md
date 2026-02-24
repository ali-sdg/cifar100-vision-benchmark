# Deep Learning Architectures on CIFAR-100

## Overview
This repository is a project-based educational endeavor focused on implementing, training, and comparing classic Deep Learning architectures. The primary goal of this project is to gain a hands-on understanding of the evolution of Convolutional Neural Networks (CNNs) by applying them to the CIFAR-100 dataset. 

A key technical aspect of this project involves an image preprocessing pipeline where the original 32x32 images are resized to 224x224 to properly feed into these standard architectures, simulating real-world data transformation challenges.

## Models Implemented
The following foundational architectures have been implemented from scratch using TensorFlow and Keras:
- **AlexNet**: The pioneering deep CNN architecture that sparked the modern deep learning era.
- **VGG-13**: A deeper network demonstrating the effectiveness of stacked small (3x3) convolutional filters.
- **ResNet-18**: A deep residual network utilizing skip connections to effectively train deeper models and solve the vanishing gradient problem.

## Dataset and Preprocessing
- **Dataset**: [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) (100 classes, 600 images per class)
- **Preprocessing**: 
  - Images are upscaled from their native `32x32` resolution to `224x224`.
  - Normalization and standard scaling applied for stable training.

## Results and Comparison
The models were trained and evaluated based on their Validation Accuracy. The residual connections in ResNet-18 allowed it to capture more complex patterns and outperform the earlier architectures.

| Model | Best Validation Accuracy |
| :--- | :---: |
| **ResNet-18** | **63.16%** |
| **VGG-13** | 57.03% |
| **AlexNet** | 56.97% |

---

## Performance Analysis & Training Plots

Below are the training history plots (Accuracy and Loss) for each architecture. A common observation across all models is the presence of **Overfitting**. 

Because these are large capacity models trained on CIFAR-100 (which has only 500 training images per class), the models eventually memorize the training data (training accuracy approaches high values, and training loss approaches zero), while the validation accuracy plateaus and validation loss begins to increase. This is a classic Deep Learning challenge.

### 1. ResNet-18
ResNet-18 achieved the highest validation accuracy. The skip connections helped the model learn features faster and generalize slightly better than the non-residual networks, though the gap between training and validation metrics still indicates overfitting in later epochs.
![ResNet-18 Performance](results/resnet18_plot.png)

### 2. VGG-13
VGG-13 shows a steady increase in training accuracy, but the validation accuracy flattens out around 57%. The capacity of VGG-13 is very high, making it highly susceptible to overfitting without aggressive regularization techniques.
![VGG-13 Performance](results/vgg13_plot.png)

### 3. AlexNet
Similar to VGG-13, AlexNet quickly learns the training set. The validation loss curve explicitly shows the point where the model stops generalizing and starts memorizing the noise in the training data.
![AlexNet Performance](results/alexnet_plot.png)

*Future Improvements to mitigate overfitting could include implementing advanced Data Augmentation (e.g., MixUp, Cutout), increasing Dropout rates, or using pre-trained weights (Transfer Learning).*

## Technologies Used
- **Python**
- **TensorFlow / Keras** (including Mixed Precision training)
- **Jupyter Notebook**
- **Matplotlib** (for performance visualization)
