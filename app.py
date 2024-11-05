import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

IMAGE_SIZE = 250
CHANNELS = 3
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

st.set_page_config(page_title='Cats vs. Dogs')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_predict(img, model):
    # Prepare the image for prediction
    data = np.ndarray(shape=(1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), dtype=np.float32)
    image = ImageOps.fit(img, (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 255.0)
    data[0] = normalized_image_array
    prediction = model.predict(data)
    return prediction

@st.cache(allow_output_mutation=True)
def load_model():
    model_path = 'model.hdf5'  # Updated model file name
    if not os.path.exists(model_path):
        return None  # Indicate that the model wasn't found
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        return None  # Indicate that an error occurred

st.write("### Cats vs. Dogs Classification")
with st.spinner('Loading Model'):
    model = load_model()

if model is None:
    st.error("Model file not found or failed to load.")
else:
    uploaded_file = st.file_uploader("Choose an Image")

    if st.button("Predict"):
        if uploaded_file is not None and allowed_file(uploaded_file.name):
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', width=300)
            with st.spinner('Predicting'):
                label = upload_predict(image, model)
            if label[0] >= 0.5:
                st.success("### The Image is a Dog")
            else:
                st.success("### The Image is a Cat")
        else:
            st.error("Please upload a valid image file.")
