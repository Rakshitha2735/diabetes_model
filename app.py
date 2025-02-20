import pickle
import streamlit as st
import pandas as pd
import os
from sklearn.metrics import accuracy_score

# Streamlit Page Configuration
st.set_page_config(page_title="Diabetes Prediction", layout="wide", page_icon="🧑‍⚕")

# Apply background gradient with three colors
page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom, #B3E5FC, #FFFFFF, #C8E6C9); /* Light blue to white to light green */
    background-size: cover;
    background-position: center;
}
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.8);
    padding: 10px;
    border-radius: 10px;
}
.stNumberInput > div > div > input {
    background-color: rgba(255, 255, 255, 0.8);
    border-radius: 8px;
    padding: 10px;
}
.stButton > button {
    border-radius: 10px;
    background: linear-gradient(to right, #4CAF50, #45a049);
    color: white;
    padding: 10px 20px;
    border: none;
    transition: all 0.3s ease-in-out;
}
.stButton > button:hover {
    background: linear-gradient(to right, #45a049, #4CAF50);
    transform: scale(1.05);
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load the trained model
diabetes_model_path = r"diabetes_model.sav"

if os.path.exists(diabetes_model_path):
    with open(diabetes_model_path, 'rb') as model_file:
        diabetes_model = pickle.load(model_file)
else:
    st.error("Model file not found. Please check the file path.")
    st.stop()

st.title('🩺 Diabetes Prediction using Machine Learning')

# Input Fields in a 3-column layout
col1, col2, col3 = st.columns(3)

with col1:
    Pregnancies = st.number_input('Number of Pregnancies', min_value=0, step=1)

with col2:
    Glucose = st.number_input('Glucose Level', min_value=0.0, format="%.2f")

with col3:
    BloodPressure = st.number_input('Blood Pressure Value', min_value=0.0, format="%.2f")

with col1:
    SkinThickness = st.number_input('Skin Thickness Value', min_value=0.0, format="%.2f")

with col2:
    Insulin = st.number_input('Insulin Level', min_value=0.0, format="%.2f")

with col3:
    BMI = st.number_input('BMI Value', min_value=0.0, format="%.2f")

with col1:
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function Value', min_value=0.0, format="%.4f")

with col2:
    Age = st.number_input('Age of the Person', min_value=0, step=1)

# Prediction Section
diab_diagnosis = ''
if st.button('Diabetes Test Result'):
    try:
        # Prepare user input
        user_input = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
        
        # Make Prediction
        diab_prediction = diabetes_model.predict(user_input)

        # Display Result
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
            st.error(diab_diagnosis)
        else:
            diab_diagnosis = 'The person is NOT diabetic'
            st.success(diab_diagnosis)

    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")

# Show Model Accuracy
if st.button('Show Model Accuracy'):
    test_data_path = r"C:\aiml_project2\diabetes.csv"

    if os.path.exists(test_data_path):
        try:
            test_data = pd.read_csv(test_data_path)

            # Ensure 'Outcome' column exists
            if "Outcome" not in test_data.columns:
                st.error("The dataset must contain an 'Outcome' column.")
            else:
                x_test = test_data.drop(columns=["Outcome"])
                y_test = test_data["Outcome"]

                y_pred = diabetes_model.predict(x_test)
                accuracy = accuracy_score(y_test, y_pred)

                st.sidebar.header("📊 Model Performance")
                st.sidebar.info(f"Model Accuracy: **{accuracy * 100:.2f}%**")

        except Exception as e:
            st.error(f"Error reading test data: {str(e)}")
    else:
        st.error("Test dataset file not found. Please check the file path.")
