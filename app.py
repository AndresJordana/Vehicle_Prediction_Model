import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from sklearn.preprocessing import OneHotEncoder

# Define constants
MODEL_URL = 'https://raw.githubusercontent.com/AndresJordana/Vehicle_Prediction_Model/main/model_realistic.pkl'
MODEL_LOCAL_PATH = 'model_realistic.pkl'

# Download the model file if not already present locally
@st.cache_data
def download_model():
    try:
        response = requests.get(MODEL_URL)
        response.raise_for_status()
        with open(MODEL_LOCAL_PATH, 'wb') as file:
            file.write(response.content)
        return MODEL_LOCAL_PATH
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading model: {e}")
        return None

# Load the model
model_path = download_model()
if model_path:
    with open(model_path, 'rb') as file:
        model_realistic = pickle.load(file)
else:
    st.stop()

# Define categories used during training
categories = {
    "Gender": ["Male", "Female", "Non-binary"],
    "Income": ["20000-39999", "40000-59999", "60000-79999", "80000-99999", "100000+"],
    "Age Range": ["18-25", "26-35", "36-50", "51+"],
    "Household Size": [1, 2, 3, 4, 5, 6, 7, 8],
    "Location": ["Urban", "Suburban", "Rural"],
    "Education Level": ["High School", "Associate", "Bachelor's", "Master's", "PhD"]
}

# Initialize the encoder
encoder = OneHotEncoder(categories=[categories[cat] for cat in categories], handle_unknown='ignore')
encoder.fit(pd.DataFrame({key: [val[0]] for key, val in categories.items()}))

# Function to encode user inputs
def encode_inputs(input_data):
    input_df = pd.DataFrame([input_data])
    encoded_input = encoder.transform(input_df).toarray()
    return encoded_input

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

# Predict the vehicle
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

    # Make prediction
    try:
        prediction = model_realistic.predict(encoded_input)
        st.subheader("Predicted Vehicle")
        st.write(prediction[0])
    except Exception as e:
        st.error(f"Error making prediction: {e}")
