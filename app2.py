import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import skimage.transform
import cv2
import streamlit as st
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    model = tf.keras.models.load_model('brain_final.hdf5')
    return model

def predict_class(image, model):
    lime_img = skimage.transform.resize(image, (150, 150))
    resized_image = cv2.resize(lime_img, (150, 150))
    reshaped_image = np.reshape(resized_image, (1, 150, 150, 3))
    predictions = model.predict(reshaped_image)
    pred = predictions[0][0]
    if pred > 0.5:
        pred = "HEALTHY"
    else:
        pred = "BRAIN_TUMOR"
    return pred

st.title("Brain Tumor Detection")

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg', 'gif'])

if uploaded_file is not None:
    if allowed_file(uploaded_file.name):
        file_path = os.path.join(UPLOAD_FOLDER, secure_filename(uploaded_file.name))
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        img = image.load_img(file_path, target_size=(224, 224))
        pred = predict_class(np.asarray(img), model)
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        st.write(f'Prediction: {pred}')
        os.remove(file_path)  # Clean up the uploaded file
    else:
        st.write("File not allowed")

