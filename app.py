import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Load the trained model (assuming it's saved locally)
# Replace 'model_realistic.pkl' with your actual file path
# with open('model_realistic.pkl', 'rb') as file:
#     model_realistic = pickle.load(file)

# For demonstration, using a placeholder model
model_realistic = RandomForestClassifier(random_state=42)

# Sample categories for encoding (used during training)
categories = {
    "Gender": ["Male", "Female", "Non-binary"],
    "Income": ["20000-39999", "40000-59999", "60000-79999", "80000-99999", "100000+"],
    "Age Range": ["18-25", "26-35", "36-50", "51+"],
    "Household Size": [1, 2, 3, 4, 5, 6, 7, 8],
    "Location": ["Urban", "Suburban", "Rural"],
    "Education Level": ["High School", "Associate", "Bachelor's", "Master's", "PhD"]
}

# Initialize the OneHotEncoder with predefined categories
encoder = OneHotEncoder(categories=list(categories.values()))
encoder.fit(np.array([list(cat) for cat in categories.values()]).T)

def encode_inputs(input_data):
    # Convert input to a 2D list (required by the encoder)
    input_list = [[
        input_data["Gender"],
        input_data["Income"],
        input_data["Age Range"],
        input_data["Household Size"],
        input_data["Location"],
        input_data["Education Level"]
    ]]

    # Apply one-hot encoding
    input_encoded = encoder.transform(input_list)

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
