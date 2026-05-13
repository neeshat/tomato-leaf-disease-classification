
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("tomato_mobilenetv2_model.h5")

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

# Treatment recommendations
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

# Streamlit page config
st.set_page_config(page_title="Tomato Leaf Disease Classifier")

st.title("🍅 Tomato Leaf Disease Classification")

st.write(
    "Upload a tomato leaf image to predict the disease "
    "and receive treatment advice."
)

# File uploader
uploaded_file = st.file_uploader(
    "Upload a Tomato Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Display image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.resize((224,224))
    img_array = np.array(img) / 255.0

    # Ensure RGB
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions)]

    confidence = np.max(predictions) * 100

    # Output
    st.subheader("Prediction Result")

    st.success(f"Disease: {predicted_class}")

    st.info(f"Confidence Score: {confidence:.2f}%")

    # Treatment
    st.subheader("Treatment Recommendation")

    st.write(treatment_dict[predicted_class])

    # Probability chart
    st.subheader("Prediction Probabilities")

    prob_dict = {
        class_names[i]: float(predictions[0][i])
        for i in range(len(class_names))
    }

    st.bar_chart(prob_dict)

# Disclaimer
st.warning(
    "Disclaimer: This prediction system is for educational "
    "and research purposes only. Always consult agricultural "
    "experts before taking large-scale farming decisions."
)
