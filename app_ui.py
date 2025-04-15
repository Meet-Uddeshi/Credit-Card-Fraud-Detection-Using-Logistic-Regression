# Step 1 => Importing required libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from app import (
    load_dataset,
    check_missing_values,
    count_classes,
    separate_transactions,
    show_statistics,
    get_class_mean,
    balance_dataset,
    separate_features_target,
    normalize_features,
    split_dataset,
    train_model,
    evaluate_model,
    visualize_results
)

# Step 2 => Set page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💰",
    layout="wide"
)

# Step 3 => Add title and description
st.title("Credit Card Fraud Detection System")
st.markdown("""
This application uses Logistic Regression to detect fraudulent credit card transactions.
Upload your dataset or use the sample dataset to analyze and predict fraud.
""")

# Step 4 => Sidebar for file upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Step 5 => Main content
if uploaded_file is not None:
    data = load_dataset(uploaded_file)
    st.subheader("Dataset Overview")
    st.write("Shape of the dataset:", data.shape)
    st.write("Missing values:", check_missing_values(data))
    st.subheader("Class Distribution")
    class_counts = count_classes(data)
    st.write(class_counts)
    fig, ax = plt.subplots()
    ax.pie(class_counts, labels=['Legitimate', 'Fraudulent'], autopct='%1.1f%%')
    st.pyplot(fig)
    legit_transactions, fraud_transactions = separate_transactions(data)
    st.subheader("Transaction Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Legitimate Transactions:")
        st.write(legit_transactions.Amount.describe())
    with col2:
        st.write("Fraudulent Transactions:")
        st.write(fraud_transactions.Amount.describe())
    st.subheader("Mean Values by Class")
    st.write(get_class_mean(data))
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            balanced_data = balance_dataset(legit_transactions, fraud_transactions)
            features, target = separate_features_target(balanced_data)
            norm_features = normalize_features(features)
            X_train, X_test, Y_train, Y_test = split_dataset(norm_features, target)
            model = train_model(X_train, Y_train)
            training_accuracy, test_accuracy = evaluate_model(model, X_train, Y_train, X_test, Y_test)
            st.subheader("Model Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Accuracy", f"{training_accuracy:.2%}")
            with col2:
                st.metric("Test Accuracy", f"{test_accuracy:.2%}")
            st.subheader("Accuracy Comparison")
            visualize_results(training_accuracy, test_accuracy)
            st.pyplot(plt.gcf())
else:
    st.info("Please upload a CSV file to begin analysis.")