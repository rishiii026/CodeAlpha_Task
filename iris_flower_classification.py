# Iris Flower Classification Project — Enhanced UI Version

# -----------------------------------------------------------
# File: iris_flower_classification.py
# Description: A creative and professional Streamlit ML project
# Author: Rishi (Intern @ CodeAlpha)
# -----------------------------------------------------------

# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st

# Set Streamlit page config
st.set_page_config(page_title="Iris Flower Classifier | CodeAlpha", layout="wide")

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2909/2909764.png", width=100)
st.sidebar.title("About This App")
st.sidebar.markdown("""
This professional-grade web application demonstrates the use of supervised machine learning to classify iris flowers based on their measurements.

- Built with Python & Streamlit  
- Model Used: Random Forest Classifier  
- Internship Project @ CodeAlpha
""")
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by **Rishi**, Intern @ CodeAlpha")

# Main Title
st.markdown("""
<h1 style='text-align: center; color: #4B8BBE;'>Iris Flower Classification</h1>
<p style='text-align: center;'>An interactive machine learning application that predicts the species of an iris flower based on four morphological features.</p>
<hr>
""", unsafe_allow_html=True)

# Load Dataset
df = pd.read_csv("Iris.csv")
df.drop(columns=["Id"], inplace=True)
encoder = LabelEncoder()
df['Species_encoded'] = encoder.fit_transform(df['Species'])

# Tabs
tabs = st.tabs(["Dataset", "Exploratory Data Analysis", "Model Training", "Live Prediction"])

with tabs[0]:
    st.subheader("Dataset Overview")
    st.dataframe(df.style.set_properties(**{'background-color': '#f9f9f9', 'color': 'black'}))

with tabs[1]:
    st.subheader("Exploratory Data Analysis")
    st.write("Pairplot grouped by Species")
    fig1 = sns.pairplot(df, hue="Species")
    st.pyplot(fig1)

    st.write("Correlation Heatmap")
    fig2, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.drop(columns=['Species']).corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig2)

with tabs[2]:
    st.subheader("Model Training & Evaluation")
    X = df.drop(columns=["Species", "Species_encoded"])
    y = df["Species_encoded"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.metric(label="Model Accuracy", value=f"{accuracy * 100:.2f}%")

    with st.expander("View Classification Report"):
        st.text(classification_report(y_test, y_pred, target_names=encoder.classes_))

    st.write("Confusion Matrix")
    fig3, ax3 = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    st.pyplot(fig3)

with tabs[3]:
    st.subheader("Live Species Prediction")

    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.slider("Sepal Length", float(df.SepalLengthCm.min()), float(df.SepalLengthCm.max()), float(df.SepalLengthCm.mean()))
        sepal_width = st.slider("Sepal Width", float(df.SepalWidthCm.min()), float(df.SepalWidthCm.max()), float(df.SepalWidthCm.mean()))
    with col2:
        petal_length = st.slider("Petal Length", float(df.PetalLengthCm.min()), float(df.PetalLengthCm.max()), float(df.PetalLengthCm.mean()))
        petal_width = st.slider("Petal Width", float(df.PetalWidthCm.min()), float(df.PetalWidthCm.max()), float(df.PetalWidthCm.mean()))

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = clf.predict(input_data)
    predicted_class = encoder.inverse_transform(prediction)[0]

    st.success(f"Predicted Iris Species: **{predicted_class}**")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center;'>Developed by <b>Rishi</b> | Internship @ CodeAlpha</div>", unsafe_allow_html=True)

