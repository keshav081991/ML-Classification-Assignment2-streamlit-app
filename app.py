import streamlit as st
import pickle
import pandas as pd

st.title("Bank Deposit Prediction")

# Load model, scaler, and feature names
model = pickle.load(open("models/model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
feature_cols = pickle.load(open("models/feature_columns.pkl", "rb"))

st.write("Enter input values for prediction")

# Create input fields dynamically for all features
input_data = {}
for col in feature_cols:
    input_data[col] = st.number_input(col, value=0.0)

input_df = pd.DataFrame([input_data])

# Predict button
if st.button("Predict"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("Customer likely to subscribe")
    else:
        st.error("Customer not likely to subscribe")
