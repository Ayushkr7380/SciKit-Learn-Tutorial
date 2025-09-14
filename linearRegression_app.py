import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open("linear_regression_model.pkl", "rb"))

# Web App Title
st.title("TV/Radio/Newspaper Sales Predictor")

# Input fields with explanation
st.write("Enter your advertising budget:")

tv = st.number_input("TV Advertising Budget", min_value=0.0)
radio = st.number_input("Radio Advertising Budget", min_value=0.0)
newspaper = st.number_input("Newspaper Advertising Budget", min_value=0.0)

# Predict button
if st.button("Predict Sales"):
    # Prepare features as 2D array
    features = np.array([[tv, radio, newspaper]], dtype=np.float64)

    # Predict using model
    prediction = model.predict(features)[0]  # directly scalar

    # Display results with explanation
    st.success(f"Predicted Sales: {prediction:.2f} units")
    st.info(f"This means, based on your advertising budget, the model estimates approximately {prediction:.2f} sales units.")

