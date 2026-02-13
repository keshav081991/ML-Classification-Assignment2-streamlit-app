import streamlit as st
import pickle
import pandas as pd

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/ML-Classification-Assignment2-streamlit-app/main/bank_bg.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    .title-box {
        background-color: rgba(0,0,0,0.6);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 32px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)


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

