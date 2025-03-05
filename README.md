# Food-Category-Image-Classification

This repository contains the code for an image classification project aimed at predicting food categories from images. The task is to build a neural network from scratch (without using any pretrained models) to classify images into 11 major food categories (represented by integer labels 0 to 10). For example, an image of a cat (or any food item) is assigned a label corresponding to its food category.

---

## Overview

- **Objective:**  
  Predict the food category of an image. The model is trained to classify images into one of 11 categories (bread, dessert, rice, noodles, meat, seafood, dairy products, egg, soup, fruit, and fried food).

- **Method:**  
  A neural network is built and trained from scratch for image classification. Key components include:
  - **Custom CNN Architecture:** Designed without pretrained weights.
  - **Data Augmentation:** Optional usage of torchvision.transforms to increase dataset diversity.
  - **Hyperparameter Tuning & Optimization:** Adjustments to learning rates, optimizers, and other hyperparameters to boost performance.
  - **Cross Validation / Train-Validation Split:** For robust model evaluation.
  - **Kaggle Submission Format:** Generates a CSV file with predicted labels for test images.

---

## Data ; (https://drive.google.com/file/d/1TBSv0tkTGLnqPUP9nPHUAWAExzQsisQV/view)

- **Training Set:**  
  9,867 JPEG images.
  
- **Validation Set:**  
  3,431 JPEG images.
  
- **Test Set:**  
  1,501 JPEG images.
  
- **Label Format:**  
  The image file names are in the format `[label]_[id].jpg` (e.g., `0_5.jpg` indicates an image with id 5 and label 0).

---

## Model Architecture

- **Convolutional Neural Network (CNN):**  
  - Built from scratch (no use of pretrained models).
  - The architecture includes several convolutional layers followed by activation functions and pooling layers.
  - Fully connected layers map the final feature maps to 11 output classes.
  
- **Data Augmentation (Optional):**  
  - Utilizes torchvision.transforms to generate additional diverse images.
  - Custom Dataset modifications to create linear combinations of images and adjust labels for further robustness (optional experiments).

---

## Training Details

- **Loss Function:**  
  Cross-entropy loss is used for multiclass classification.
  
- **Optimization:**  
  The model is trained with an optimizer (e.g., Adam) with tuned hyperparameters.
  
- **Validation:**  
  A separate validation set is used to monitor performance and fine-tune hyperparameters.
  
- **Kaggle Submission:**  
  The final model is used to generate predictions on the test set. The submission file is a CSV file containing:
  - **Id:** Sequential test image IDs.
  - **Label:** Predicted category labels.
  
- **Evaluation Metric:**  
  Categorization Accuracy.

---

## Code Structure

- **Notebook:**  
  - `DL_HW3_final (1).ipynb` – Contains the full implementation, including:
    - Data loading and preprocessing.
    - Definition of the custom Dataset and data augmentation methods.
    - Design and training of the CNN model.
    - Hyperparameter tuning and optimization.
    - Generation of the prediction CSV file for submission.
  
- **Dependencies:**  
  - `torch` and `torch.nn` for building and training the model.
  - `torchvision` for image transformations and data augmentation.
  - `numpy`, `pandas`, and other utilities for data processing and evaluation.

---

## Kaggle and Canvas Submission

- **Kaggle:**  
  - The generated CSV file follows the required format with two columns: `Id` and `Label`.
  
- **Canvas:**  
  - Submit your Python code along with the CSV submission file.
  - Ensure that the first line of your code includes your Kaggle account ID, e.g.:
    ```python
    # My Kaggle ID: your_kaggle_id
    ```

---

## Conclusion

This project demonstrates the process of building a neural network from scratch for the task of food category image classification. By leveraging custom CNN architectures, data augmentation, and careful hyperparameter tuning, the model aims to achieve high categorization accuracy on a challenging dataset. All code, experiments, and final predictions are provided in this repository.
