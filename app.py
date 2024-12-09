import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load or Train the Model
model_file = 'model_realistic.pkl'
try:
    # Load the trained model if available
    with open(model_file, 'rb') as file:
        model_realistic = pickle.load(file)
except FileNotFoundError:
    # If model not found, train it using sample data
    st.warning("Model not found, training a new model.")

    # Example training data
    df = pd.DataFrame({
        "Gender": ["Male", "Female", "Female", "Male"],
        "Income": ["20000-39999", "40000-59999", "100000+", "60000-79999"],
        "Age Range": ["18-25", "26-35", "36-50", "51+"],
        "Household Size": [1, 2, 5, 4],
        "Location": ["Urban", "Suburban", "Rural", "Urban"],
        "Education Level": ["High School", "Bachelor's", "PhD", "Master's"],
        "Toyota Model": ["Corolla", "RAV4", "Highlander", "Camry"]
    })

    # Define features and target
    X = df.drop("Toyota Model", axis=1)
    y = df["Toyota Model"]

    # Initialize the encoder and encode features
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Train the RandomForestClassifier
    model_realistic = RandomForestClassifier(random_state=42)
    model_realistic.fit(X_train, y_train)

    # Save the trained model
    with open(model_file, 'wb') as file:
        pickle.dump(model_realistic, file)

# Function to encode user input
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
gender = st.selectbox("Gender", ["Male", "Female", "Non-binary"])
income = st.selectbox("Income Range", ["20000-39999", "40000-59999", "60000-79999", "80000-99999", "100000+"])
age_range = st.selectbox("Age Range", ["18-25", "26-35", "36-50", "51+"])
household_size = st.slider("Household Size", min_value=1, max_value=8, step=1)
location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
education_level = st.selectbox("Education Level", ["High School", "Associate", "Bachelor's", "Master's", "PhD"])

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
