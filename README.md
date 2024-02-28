# Image Classification using Machine Learning

This repository contains the source files and documentation for a machine learning project focused on image classification. The project utilizes the CIFAR-10 dataset and explores various techniques to achieve efficient and accurate classification.

## Feature Extraction

### Histogram of Oriented Gradients (HOG)

This technique extracts features that capture the local object shape and appearance by calculating the distribution of oriented gradients within small image regions.

### Reference

https://towardsdatascience.com/hog-histogram-of-oriented-gradients-67ecd887675f

## Dimensionality Reduction

### Principal Component Analysis (PCA)

This method reduces the number of features while preserving the maximum amount of information. It uses linear transformations to project data onto a lower-dimensional subspace while capturing significant variations.

### Reference

https://builtin.com/data-science/step-step-explanation-principal-component-analysis

## Exploratory Data Analysis (EDA)

The project performs various analyses to understand the dataset better:

### Mean Images

Visualizing the average pixel values across all images in each class.

### HOG Feature Distribution

Exploring the distribution of HOG features within different classes.

### PCA Component Visualizations

Analyzing how the PCA components capture variation in the data.

## Machine Learning Model

### Support Vector Machine (SVM)

This supervised learning algorithm is used to classify images into one of the 10 categories in the CIFAR-10 dataset. The model is trained using the provided training data (50,000 images) and evaluated on the test set (10,000 images), achieving an accuracy of 71%.

## Model Training

Initial training using 50,000 images.
Evaluation using 10,000 images (71% accuracy). 
Final training using the entire 60,000-image dataset.

## Testing

The model is subsequently tested on unseen images for further evaluation and potential performance improvement.

## Getting Started

1. Run `HOG_PCA_SVM.ipynb` file to create a PCA model and a SVM model trained on CIFAR-10 dataset.

    The models will be saved to pickle files named `pca_model.pkl` and `svm_model.pkl` respectively.

2. Create a virtual environment 

    `python -m venv venv`

    `venv\Scripts\activate`

3. Install required dependencies like - numpy, flask, opencv, scikit-learn etc.

4. Execute the `app.py` file

    `python app.py`

5. Upload images and let the model predict its class using Image Classification

6. `EDA.ipynb` contains all the EDA done on the CIFAR-10 dataset

