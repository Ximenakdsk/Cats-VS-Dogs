# Kaggle Challenge: Dogs vs Cats with CNN

This repository contains the solution to the classic image classification challenge (Dogs vs Cats). The main goal of the project is to compare the performance of a Convolutional Neural Network (CNN) built from scratch against an advanced model that uses **Transfer Learning** (based on VGG16) to exceed 95% accuracy.

## 📂 Project Structure

The source code is organized modularly to ensure readability, simplify debugging, and support scalability:

- `data_prep.py`: Handles image preprocessing. It uses `ImageDataGenerator` to load images from local folders and applies **Data Augmentation** techniques (rotation, zoom, flips) to the training set to help prevent overfitting.
- `cnn_basica.py`: Defines the architecture of our CNN from scratch (Baseline). It consists of multiple convolutional blocks (`Conv2D`) and pooling layers (`MaxPooling2D`), ending with a `Dropout` layer and a Dense classifier with Sigmoid output.
- `transfer_learning.py`: Implements the high-performance model. It imports the pretrained **VGG16** architecture (with *ImageNet* weights), freezes its base layers, and adds a custom classifier for our binary classification task.
- `main.py`: Acts as the **project orchestrator**. It calls the previous modules, trains both models sequentially, and uses `matplotlib` to generate and save comparison plots.

## 🛠️ Prerequisites

Make sure you have the following Python libraries installed before running the project:

```bash
pip install tensorflow matplotlib
```

Also, the program expects the image data to be organized in the same directory as the script, with the following folder structure:

```text
/
├── train/
│   ├── cats/
│   └── dogs/
├── validation/
│   ├── cats/
│   └── dogs/
├── data_prep.py
├── cnn_basica.py
├── transfer_learning.py
└── main.py
```

## 🚀 How to Run the Program

To start training the neural networks and generate the result plots, simply run the main file from your terminal:

```bash
python main.py
```

*Note: The process may take some time depending on your computer's compute capacity (CPU vs GPU).* 

## 📊 Results and Output

Once the `main.py` script finishes, the program will generate the following files in the same folder:

1. **`modelo_cnn_basica.h5`**: The trained basic CNN model file.
2. **`modelo_vgg16.h5`**: The trained Transfer Learning model file.
3. **`comparacion_modelos.png`**: An image containing two plots (Accuracy and Loss) comparing the training and validation performance of both models. **This is the primary file to submit for the challenge.**

---
**Developed by:** Stephanie Ximena Pérez Hernández