#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# Load model
with open('rf_model_heart.pkl', 'rb') as file:
    model = pickle.load(file)

# Page setup
st.set_page_config(page_title="Heart Failure Predictor", page_icon="❤️", layout="wide")

# Background & CSS styling
st.markdown("""
    <style>
        .main {
            background-color: #f9f9f9;
        }
        .block-container {
            padding: 2rem 3rem;
            background: linear-gradient(to bottom, #ffffff, #ffe6e6);
            border-radius: 10px;
        }
        .stButton>button {
            color: white;
            background: #d72638;
            border-radius: 8px;
            height: 3em;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #a21b2d;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center; color: #d72638;'>💓 Heart Failure Risk Predictor 💓</h1>", unsafe_allow_html=True)
st.image("https://cdn.pixabay.com/photo/2016/10/18/21/22/heart-1756240_1280.png", width=250)

# Sidebar - Patient Inputs
st.sidebar.header("🧾 Enter Patient Data")

def user_input_features():
    age = st.sidebar.slider('Age', 20, 100, 60)
    anaemia = st.sidebar.slider('Anaemia (1 = Yes, 0 = No)', 0, 1, 0)
    creatinine_phosphokinase = st.sidebar.slider('Creatinine Phosphokinase', 20, 8000, 250)
    diabetes = st.sidebar.slider('Diabetes (1 = Yes, 0 = No)', 0, 1, 0)
    ejection_fraction = st.sidebar.slider('Ejection Fraction (%)', 10, 80, 38)
    high_blood_pressure = st.sidebar.slider('High Blood Pressure (1 = Yes, 0 = No)', 0, 1, 0)
    platelets = st.sidebar.slider('Platelets count', 10000.0, 900000.0, 250000.0)
    serum_creatinine = st.sidebar.slider('Serum Creatinine', 0.1, 10.0, 1.1)
    serum_sodium = st.sidebar.slider('Serum Sodium', 100, 150, 137)
    sex = st.sidebar.slider('Sex (1 = Male, 0 = Female)', 0, 1, 1)
    smoking = st.sidebar.slider('Smoking (1 = Yes, 0 = No)', 0, 1, 0)
    time = st.sidebar.slider('Follow-up Period (days)', 0, 300, 120)

    data = {
        'age': age,
        'anaemia': anaemia,
        'creatinine_phosphokinase': creatinine_phosphokinase,
        'diabetes': diabetes,
        'ejection_fraction': ejection_fraction,
        'high_blood_pressure': high_blood_pressure,
        'platelets': platelets,
        'serum_creatinine': serum_creatinine,
        'serum_sodium': serum_sodium,
        'sex': sex,
        'smoking': smoking,
        'time': time
    }
    return pd.DataFrame(data, index=[0])

df = user_input_features()

# Patient data display
st.subheader("📋 Patient Summary")
st.dataframe(df.style.set_properties(**{'text-align': 'center'}))

# Prediction & Graphs
if st.button("💡 Predict Heart Risk"):
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)[0]

    result = 'Yes' if prediction[0] == 1 else 'No'

    # Show result
    st.subheader("🔍 Prediction Result")
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Failure")
    else:
        st.success("✅ Low Risk of Heart Failure")

    # Bar chart
    st.subheader("📊 Probability Chart")
    bar_fig = go.Figure(go.Bar(
        x=['No Heart Failure', 'Heart Failure'],
        y=prediction_proba,
        marker=dict(color=['green', 'red']),
        text=[f'{prediction_proba[0]*100:.2f}%', f'{prediction_proba[1]*100:.2f}%'],
        textposition='auto'
    ))
    bar_fig.update_layout(
        xaxis_title="Condition",
        yaxis_title="Probability",
        template="plotly_dark",
        width=700,
        height=400
    )
    st.plotly_chart(bar_fig)

    # Pie chart
    st.subheader("🧠 Risk Distribution")
    pie_fig = go.Figure(go.Pie(
        labels=['No Heart Failure', 'Heart Failure'],
        values=prediction_proba,
        hole=0.3,
        marker=dict(colors=['#28a745', '#dc3545'])
    ))
    pie_fig.update_layout(width=700, height=400)
    st.plotly_chart(pie_fig)



# In[ ]:




