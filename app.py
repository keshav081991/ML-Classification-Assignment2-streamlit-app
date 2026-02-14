import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Bank Deposit Prediction", layout="wide")

st.title("Bank Deposit Prediction System")

# Load scaler and feature names
scaler = pickle.load(open("models/scaler.pkl","rb"))
feature_cols = pickle.load(open("models/feature_columns.pkl","rb"))

# Load trained models
models = {
    "Logistic Regression": pickle.load(open("models/logistic.pkl","rb")),
    "Decision Tree": pickle.load(open("models/dt.pkl","rb")),
    "KNN": pickle.load(open("models/knn.pkl","rb")),
    "Naive Bayes": pickle.load(open("models/nb.pkl","rb")),
    "Random Forest": pickle.load(open("models/rf.pkl","rb")),
    "XGBoost": pickle.load(open("models/xgb.pkl","rb"))
}

# Sidebar model selection
st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox("Choose Model", list(models.keys()))
model = models[model_name]

# Create tabs
tab1, tab2 = st.tabs(["Prediction", "Model Evaluation"])

# ---------- TAB 1 : Prediction ----------
with tab1:
    st.subheader("Enter Input Values")

    input_data = {}
    for col in feature_cols:
        input_data[col] = st.number_input(col, value=0.0)

    input_df = pd.DataFrame([input_data])

    if st.button("Predict"):
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)

        if pred[0] == 1:
            st.success("Customer likely to subscribe")
        else:
            st.error("Customer not likely to subscribe")

# ---------- TAB 2 : Evaluation ----------
with tab2:
    st.subheader("Model Evaluation on Test Data")

    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv")

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", round(accuracy_score(y_test, y_pred),3))
    col2.metric("Precision", round(precision_score(y_test, y_pred),3))
    col3.metric("Recall", round(recall_score(y_test, y_pred),3))
    col4.metric("F1 Score", round(f1_score(y_test, y_pred),3))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)
