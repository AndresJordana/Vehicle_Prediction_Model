import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Load the trained model from GitHub
import requests
import os

MODEL_URL = 'https://raw.githubusercontent.com/AndresJordana/Vehicle_Prediction_Model/main/model_realistic.pkl'
MODEL_LOCAL_PATH = 'model_realistic.pkl'

# Download the model file if not already present locally
if not os.path.exists(MODEL_LOCAL_PATH):
    response = requests.get(MODEL_URL)
    with open(MODEL_LOCAL_PATH, 'wb') as file:
        file.write(response.content)

# Load the trained model
with open(MODEL_LOCAL_PATH, 'rb') as file:
    model_realistic = pickle.load(file)

# Encoder used during training
encoder = OneHotEncoder()

# Sample categories for encoding (used during training)
categories = {
    "Gender": ["Male", "Female", "Non-binary"],
    "Income": ["20000-39999", "40000-59999", "60000-79999", "80000-99999", "100000+"],
    "Age Range": ["18-25", "26-35", "36-50", "51+"],
    "Household Size": list(range(1, 9)),
    "Location": ["Urban", "Suburban", "Rural"],
    "Education Level": ["High School", "Associate", "Bachelor's", "Master's", "PhD"]
}

def encode_inputs(input_data):
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # Apply one-hot encoding
    encoded_input = encoder.fit(categories.values())  # Use categories for fitting
    input_encoded = encoder.transform(input_df)

    return input_encoded

# Streamlit app setup
st.title("Toyota Vehicle Prediction App")
st.write("Predict the most likely Toyota vehicle a customer might purchase based on their profile.")

# Input fields
gender = st.selectbox("Gender", categories["Gender"])
income = st.selectbox("Income Range", categories["Income"])
age_range = st.selectbox("Age Range", categories["Age Range"])
household_size = st.slider("Household Size", min_value=1, max_value=8, step=1)
location = st.selectbox("Location", categories["Location"])
education_level = st.selectbox("Education Level", categories["Education Level"])

# Prediction
if st.button("Predict Vehicle"):
    # Prepare input data
    input_data = {
        "Gender": gender,
        "Income": income,
        "Age Range": age_range,
        "Household Size": household_size,
        "Location": location,
        "Education Level": education_level
    }

    # Encode inputs
    encoded_input = encode_inputs(input_data)

    # Predict
    prediction = model_realistic.predict(encoded_input)

    # Display prediction
    st.subheader("Predicted Vehicle")
    st.write(prediction[0])
