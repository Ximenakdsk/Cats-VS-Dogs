# Kaggle Challenge: Dogs vs. Cats with CNN 🐾

The primary goal is to solve a binary image classification problem using Deep Learning, comparing a custom-built architecture against a pre-trained model.

## 🚀 Project Objective
Achieve a classification accuracy higher than **95%** using **Transfer Learning**, evaluating its performance against a Convolutional Neural Network (CNN) built from scratch.

## 📂 Project Structure
Based on the implementation plan, the repository is organized as follows:
- `/train`: Labeled training dataset.
- `/validation`: Dataset for hyperparameter tuning and model validation.
- `main.py`: Main script for the training and evaluation workflow.

## 🛠️ Implementation Strategy

### 1. Data Preparation and Augmentation
To prevent *overfitting* and reach the accuracy target, `ImageDataGenerator` is implemented to apply real-time transformations (rotations, zoom, and shifts) to the training images.

### 2. Modeling and Architectures
Two competitive models are developed:
* **CNN from Scratch:** A sequential architecture featuring 3 or 4 `Conv2D` + `MaxPooling2D` blocks, regularized with `Dropout` layers.
* **Transfer Learning (VGG16):** Utilizing the VGG16 convolutional base (ImageNet weights) with custom dense layers and *Fine-tuning* techniques if necessary to exceed 95% accuracy.

### 3. Evaluation and Comparison
Performance is analyzed using `matplotlib` to generate comparative charts of:
- **Accuracy (Training vs. Validation)** for both models.
- **Loss (Training vs. Validation)** for both models.

## 📺 References and Foundations
- **Theory:** Neural Networks series by *DotCSV* (Neuron, Backpropagation, and CNN).
- **Ethical Analysis:** Discussion on algorithmic bias based on the documentary *"Coded Bias"*.

---
**Developed by:** Stephanie Ximena Pérez Hernández
