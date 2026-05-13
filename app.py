import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout
)

# Rebuild model architecture
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Load saved weights
model.load_weights("mobilenet_weights.weights.h5")

# Class names
class_names = [
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# Clean display names
display_names = {
    'Tomato_Bacterial_spot': 'Tomato Bacterial Spot',
    'Tomato_Early_blight': 'Tomato Early Blight',
    'Tomato_Late_blight': 'Tomato Late Blight',
    'Tomato_Leaf_Mold': 'Tomato Leaf Mold',
    'Tomato_Septoria_leaf_spot': 'Tomato Septoria Leaf Spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Tomato Spider Mites',
    'Tomato__Target_Spot': 'Tomato Target Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Tomato Yellow Leaf Curl Virus',
    'Tomato__Tomato_mosaic_virus': 'Tomato Mosaic Virus',
    'Tomato_healthy': 'Healthy Tomato Leaf'
}

# Treatment dictionary
treatment_dict = {

    "Tomato_Bacterial_spot":
    "Remove infected leaves and apply copper-based bactericides.",

    "Tomato_Early_blight":
    "Use fungicides and avoid overhead irrigation.",

    "Tomato_Late_blight":
    "Remove infected plants immediately and apply protective fungicides.",

    "Tomato_Leaf_Mold":
    "Improve air circulation and reduce humidity levels.",

    "Tomato_Septoria_leaf_spot":
    "Prune infected leaves and apply appropriate fungicide spray.",

    "Tomato_Spider_mites_Two_spotted_spider_mite":
    "Use insecticidal soap or neem oil to control mites.",

    "Tomato__Target_Spot":
    "Remove affected leaves and apply fungicide treatment.",

    "Tomato__Tomato_YellowLeaf__Curl_Virus":
    "Control whiteflies and remove infected plants.",

    "Tomato__Tomato_mosaic_virus":
    "Disinfect tools and remove infected plants to prevent spread.",

    "Tomato_healthy":
    "The plant appears healthy. Continue proper care and monitoring."
}

# Streamlit UI
st.set_page_config(page_title="Tomato Leaf Disease Classification")

st.title("🍅 Tomato Leaf Disease Classification")

st.write(
    "Upload a tomato leaf image to predict the disease "
    "and receive treatment advice."
)

uploaded_file = st.file_uploader(
    "Upload Tomato Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((224,224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)

    predicted_index = np.argmax(predictions)

    predicted_class = class_names[predicted_index]

    confidence = float(np.max(predictions) * 100)

    st.subheader("Prediction Result")

    st.success(display_names[predicted_class])

    st.info(f"Confidence Score: {confidence:.2f}%")

    st.subheader("Treatment Recommendation")

    st.write(treatment_dict[predicted_class])

    st.subheader("Prediction Probabilities")

    prob_dict = {
        display_names[class_names[i]]: float(predictions[0][i])
        for i in range(len(class_names))
    }

    st.bar_chart(prob_dict)

# Disclaimer
st.warning(
    "Disclaimer: This system is intended for educational "
    "and research purposes only. Please consult agricultural "
    "experts before making large-scale farming decisions."
)
