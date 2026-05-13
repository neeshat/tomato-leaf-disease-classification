# Tomato Leaf Disease Classification

## Live Demo

Streamlit App:
https://tomato-disease-detector-nishat-305.streamlit.app/

## Project Overview

This project is a deep learning-based tomato leaf disease classification system developed using TensorFlow and Keras.

The system classifies 10 different tomato leaf conditions, including healthy leaves and multiple diseases, from uploaded leaf images.

A Streamlit web application is included for real-time disease prediction and treatment recommendation.

---

## Features

- Tomato leaf disease classification
- Custom CNN model
- Transfer Learning using MobileNetV2
- Image upload interface
- Confidence score prediction
- Treatment recommendation system
- Visualization of prediction probabilities

---

## Dataset

PlantVillage Dataset (Tomato Classes)

Dataset Source:
https://www.kaggle.com/datasets/emmarex/plantdisease

---

## Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Matplotlib
- Seaborn

---

## Model Performance

### Custom CNN

- Validation Accuracy: ~85%

### MobileNetV2

- Validation Accuracy: ~88%

---

## Project Structure

- app.py
- requirements.txt
- tomato_mobilenetv2_model.keras
- Tomato_Leaf_Disease_Classification.ipynb

---

## Run Locally

pip install -r requirements.txt

streamlit run app.py

---

## Disclaimer

This system is developed for educational and research purposes only.

Always consult agricultural experts before making large-scale farming decisions.
