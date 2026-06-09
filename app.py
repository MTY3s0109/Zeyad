import streamlit as st 

import joblib

le_1 = joblib.load("le_1.pkl")
le_4 = joblib.load("le_4.pkl")
le_5 = joblib.load("le_5.pkl")
sc = joblib.load("sc.pkl")
model = joblib.load("insurance.pkl")

st.title("Insurance Data")

age = st.number_input("Age:", min_value=0, max_value= 100)

sex = st.selectbox("Sex:",le_1.classes_)

bmi = st.number_input("BMI:" )

children = st.number_input("Children" )

smoker = st.selectbox("Smoke",le_4.classes_)

region = st.selectbox("Region",le_5.classes_)

input_data = [[age,sex,bmi	,children	,smoker	,region]]

sex = le_1.transform([sex])[0]
smoker = le_4.transform([smoker])[0]
region = le_5.transform([region])[0]

input_data = [[age,sex,bmi	,children	,smoker	,region]]

input_scale = sc.transform(input_data) 

prediction = model.predict(input_scale)
st.success(prediction)




