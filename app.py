import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load('model.pkl')

# Title of the app
st.title('Simple Linear Classifier App')

# Sidebar input
st.sidebar.header('Input Features')
feature1 = st.sidebar.slider('Feature 1', -5.0, 5.0, 0.0)
feature2 = st.sidebar.slider('Feature 2', -5.0, 5.0, 0.0)

input_features = np.array([[feature1, feature2]])

# Prediction
if st.sidebar.button('Predict'):
    prediction = model.predict(input_features)
    probability = model.predict_proba(input_features)
    st.write(f'### Prediction: {"Class 1" if prediction[0]==1 else "Class 0"}')
    st.write(f'Probability of Class 0: {probability[0][0]:.2f}')
    st.write(f'Probability of Class 1: {probability[0][1]:.2f}')

# Visualization
if st.checkbox('Show Data Visualization'):
    df = pd.read_csv('data.csv')
    fig, ax = plt.subplots()
    sns.scatterplot(x='feature1', y='feature2', hue='target', data=df, ax=ax)
    st.pyplot(fig)
