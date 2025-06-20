import pickle
import streamlit as st
import pandas as pd
from groq import Groq  # Ensure you have the 'groq' package installed

# Load the saved models and pre-processing objects
with open('C:/Users/agraw/Desktop/Untitled Folder/rf_diag_model.sav', 'rb') as f:
    rf_diag = pickle.load(f)

with open('C:/Users/agraw/Desktop/Untitled Folder/rf_stage_model.sav', 'rb') as f:
    rf_stage = pickle.load(f)

with open('C:/Users/agraw/Desktop/Untitled Folder/scaler.sav', 'rb') as f:
    scaler = pickle.load(f)

with open('C:/Users/agraw/Desktop/Untitled Folder/imputer.sav', 'rb') as f:
    imputer = pickle.load(f)

# Function to predict Stage and Diagnosis for a new user
def predict_stage_and_diagnosis(user_input):
    # Convert user input to DataFrame
    user_df = pd.DataFrame([user_input])

    # Preprocess the input
    user_df_imputed = imputer.transform(user_df)
    user_df_scaled = scaler.transform(user_df_imputed)

    # Predict Diagnosis
    diagnosis_prediction = rf_diag.predict(user_df_scaled)
    diagnosis_prediction_prob = rf_diag.predict_proba(user_df_scaled)

    # Predict Stage only if Diagnosis = 1
    if diagnosis_prediction[0] == 1:
        stage_prediction = rf_stage.predict(user_df_scaled)
        stage_prediction_prob = rf_stage.predict_proba(user_df_scaled)
        return {
            'Diagnosis': int(diagnosis_prediction[0]),
            'Diagnosis_Probability': diagnosis_prediction_prob[0].tolist(),
            'Stage': int(stage_prediction[0]),
            'Stage_Probability': stage_prediction_prob[0].tolist()
        }
    else:
        return {
            'Diagnosis': int(diagnosis_prediction[0]),
            'Diagnosis_Probability': diagnosis_prediction_prob[0].tolist(),
            'Stage': 0,  # Normal stage for undiagnosed patients
            'Stage_Probability': [1.0, 0.0, 0.0, 0.0]  # 100% probability of Normal
        }

# Example Streamlit app layout
st.title("Alzheimer's Diagnosis Prediction")

# Create user input fields (Streamlit will automatically handle the input form)
user_input = {
    'Age': st.number_input('Age', min_value=1, max_value=120, value=75),
    'Gender': st.selectbox('Gender(0 = Male, 1 = Female)', options=[0, 1], index=1),
    'Ethnicity': st.selectbox('Ethnicity(0:Caucasian,1:African American,2:Asian,3:Other)', options=[0, 1, 2, 3], index=0),
    'EducationLevel': st.selectbox('Education Level(0:None, 1:High School, 2:Bachelors, 3:Higher)', options=[0, 1, 2, 3], index=2),
    'BMI': st.number_input('BMI(15 to 40.)', min_value=0.0, value=25.5),
    'Smoking': st.selectbox('Smoking(0 = No, 1 = Yes)', options=[0, 1], index=0),
    'AlcoholConsumption': st.number_input('Alcohol Consumption (g/day)(Range:0 to 20)', min_value=0.0, value=10.5),
    'PhysicalActivity': st.number_input('Physical Activity (hours/week)(Range:0 to 10)', min_value=0.0, value=4.5),
    'DietQuality': st.number_input('Diet Quality(Range: 0 to 10)', min_value=0.0, value=5.2),
    'SleepQuality': st.number_input('Sleep Quality(Range:4 to 10)', min_value=0.0, value=7.0),
    'FamilyHistoryAlzheimers': st.selectbox('Family History of Alzheimer\'s(0 = No, 1 = Yes)', options=[0, 1], index=1),
    'CardiovascularDisease': st.selectbox('Cardiovascular Disease(0 = No, 1 = Yes)', options=[0, 1], index=0),
    'Diabetes': st.selectbox('Diabetes(0 = No, 1 = Yes)', options=[0, 1], index=0),
    'Depression': st.selectbox('Depression(0 = No, 1 = Yes)', options=[0, 1], index=1),
    'HeadInjury': st.selectbox('Head Injury(0 = No, 1 = Yes)', options=[0, 1], index=0),
    'Hypertension': st.selectbox('Hypertension(0 = No, 1 = Yes)', options=[0, 1], index=1),
    'SystolicBP': st.number_input('Systolic BP(Range: 90 to 180)', min_value=0, value=120),
    'DiastolicBP': st.number_input('Diastolic BP(Range: 60 to 120)', min_value=0, value=80),
    'CholesterolTotal': st.number_input('Total Cholesterol(Range: 150 to 300)', min_value=0.0, value=200.5),
    'CholesterolLDL': st.number_input('LDL Cholesterol(Range: 50 to 200)', min_value=0.0, value=130.5),
    'CholesterolHDL': st.number_input('HDL Cholesterol(Range: 20 to 100)', min_value=0.0, value=50.5),
    'CholesterolTriglycerides': st.number_input('Triglycerides(Range: 50 to 400)', min_value=0.0, value=150.5),
    'MMSE': st.number_input('Mini-Mental State Exam (MMSE)(Range: 0 to 30)', min_value=0, value=15),
    'FunctionalAssessment': st.number_input('Functional Assessment(Range: 0 to 10)', min_value=0.0, value=15.5),
    'MemoryComplaints': st.selectbox('Memory Complaints(0 = No, 1 = Yes)', options=[0, 1], index=1),
    'BehavioralProblems': st.selectbox('Behavioral Problems(0 = No, 1 = Yes)', options=[0, 1], index=0),
    'ADL': st.number_input('Activities of Daily Living (ADL)(Range:0 to 10)', min_value=0.0, value=4.5),
    'Confusion': st.selectbox('Confusion(0 = No, 1 = Yes)', options=[0, 1], index=1),
    'Disorientation': st.selectbox('Disorientation(0 = No, 1 = Yes)', options=[0, 1], index=1),
    'PersonalityChanges': st.selectbox('Personality Changes(0 = No, 1 = Yes)', options=[0, 1], index=1),
    'DifficultyCompletingTasks': st.selectbox('Difficulty Completing Tasks(0 = No, 1 = Yes)', options=[0, 1], index=1),
    'Forgetfulness': st.selectbox('Forgetfulness(0 = No, 1 = Yes)', options=[0, 1], index=1)
}

# Get prediction
if st.button("Generate timetable"):
    prediction = predict_stage_and_diagnosis(user_input)
    
    diagnosis_map = {0: 'No Alzheimer\'s', 1: 'Alzheimer\'s'}
    stage_map = {0: 'Stage 0 (No impairment)', 1: 'Stage 1 (Very mild cognitive decline)',
                 2: 'Stage 2 (Mild cognitive decline)', 3: 'Stage 3 (Moderate cognitive decline)'}
 
    diagnosis = diagnosis_map[prediction['Diagnosis']]
    stage = stage_map[prediction['Stage']]
    
    # Generate the chat input for Groq API
    chat_input = f"The patient is diagnosed with {diagnosis} and is at {stage}. Can you generate a daily timetable for this patient?"

    # Set up Groq API to generate the timetable based on this input
    try:
        client = Groq(api_key="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")  # Replace with your actual API key

        # Call the Groq chat model with the formatted input
        chat_completion = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": chat_input,  # The formatted user input (diagnosis and stage)
            }],
            model="Llama-3.3-70B-Versatile",  # Example model for generating timetable
        )

        # Output the result (the timetable generated by Groq)
        timetable = chat_completion.choices[0].message.content

        # Display the results in Streamlit
        st.subheader("Generated Timetable:")
        st.write(timetable)
    except Exception as e:
        st.error(f"Error generating timetable: {e}")
