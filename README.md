# Brain Tumor Classification and Explanation Using Deep Learning

## Deployed Application

You can view and interact with the deployed application [here](https://brain-tumour-detection-vitascan-r.onrender.com/).

## Overview

This project demonstrates a deep learning approach to classifying brain tumor images and provides explanations for the model's predictions using image perturbation techniques. The project includes:

- **Data Download and Preparation**: Downloading and preparing the brain tumor dataset.
- **Model Training**: Training a Convolutional Neural Network (CNN) for binary classification.
- **Model Evaluation**: Evaluating the model's performance.
- **Explanation**: Using image perturbation methods to explain model predictions.
- **Frontend Application**: A Streamlit and Flask-based web application for making predictions and visualizing explanations.

## Requirements

To run this code, you need to have the following libraries installed:

- `numpy`
- `pandas`
- `opencv-python` (OpenCV)
- `matplotlib`
- `seaborn`
- `tensorflow` (Keras)
- `scikit-image`
- `scikit-learn`
- `streamlit`
- `flask`

You can install these libraries using pip:

```bash
pip install numpy pandas opencv-python matplotlib seaborn tensorflow scikit-image scikit-learn streamlit flask bash```

## Getting Started
1. Data Download and Preparation
The data is downloaded from Kaggle and unzipped.

2. Model Training
A Convolutional Neural Network (CNN) is created and trained on the dataset.

3. Model Evaluation
Evaluate the model and visualize training history.

4. Explanation Using Perturbation
The model predictions are explained using image perturbations.

5. Frontend Application
A Streamlit application is available to make predictions and visualize the explanations. You can access it here.

Usage
Run the Streamlit Application
To run the Streamlit application locally, navigate to the project directory and execute:

bash
Copy code
streamlit run app.py
The app will provide options to upload an image, get predictions, and view explanations.

Streamlit Frontend
The frontend is deployed and available online. You can use the web application to upload images, receive predictions, and view the explanation of the model's decision.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
Dataset: Brain Tumor Dataset
Tools and Libraries: TensorFlow, Keras, OpenCV, Scikit-Image, Streamlit, Flask.


