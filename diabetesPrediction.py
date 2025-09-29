import pickle
import streamlit as st
import pandas as pd




model = pickle.load(open("diabetesModelRandomForestClassifier.pkl","rb"))
transformer = pickle.load(open("diabetesTransformer.pkl","rb"))

st.title("Diabetes Prediction")

gender  = st.selectbox("Select Gender",options=["Male","Female","Other"])
age = st.number_input("Enter age")
hypertension_input = st.selectbox("Do you have Hypertension",options=["Yes","No"])
heart_disease_input = st.selectbox("Do you have Heart disease",options=["Yes","No"])
smoking_history = st.selectbox("Smoking History",options=["No Info","never","former","current", "not current","ever"])
BMI = st.number_input("Enter BMI")
hba1c_level = st.number_input("Enter HbA1c level")
st.info("HbA1c is a blood test that tells your average blood sugar level over the last 2–3 months. Higher values may indicate a higher risk of diabetes.")
glucose_level = st.number_input("Enter Blood glucose level")

#convert hypertension and heart_disease into 0/1

hypertension = 1 if hypertension_input == "Yes" else 0
heart_disease = 1 if heart_disease_input == "Yes" else 0

input_dict = {
    "gender":gender,
    "age":age,
    "hypertension":hypertension,
    "heart_disease":heart_disease,
    "smoking_history":smoking_history,
    "bmi":BMI,
    "HbA1c_level":hba1c_level,
    "blood_glucose_level":glucose_level,
}

def predict_diabetes(input_dict,transformer):

    #Convert the dictionary into a single frame

    input_df = pd.DataFrame([input_dict])

    #Transform using the fitted transformer

    input_tf = transformer.transform(input_df)

    predict = model.predict(input_tf)

    return predict[0]

if st.button("Check Now"):
    prediction = predict_diabetes(input_dict,transformer)

    if prediction == 0:
        st.markdown("<h4 style='color:green'>You are not currently at risk of diabetes.</h4>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<h4 style='color:red'>Based on the input data, there is a risk of diabetes. Please consult a healthcare professional.</h4>",
            unsafe_allow_html=True)


