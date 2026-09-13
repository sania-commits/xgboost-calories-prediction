import requests
import streamlit as st
import os

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/predict"
)


st.set_page_config(
    page_title="Calories Burned Predictor",
    page_icon="🔥",
    layout="centered"
)


st.title("🔥 Calories Burned Predictor")

st.write(
    "Enter your workout and body metrics to estimate "
    "how many calories you burned."
)


age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

weight = st.number_input(
    "Weight (kg)",
    min_value=30.0,
    max_value=200.0,
    value=65.0
)

height = st.number_input(
    "Height (m)",
    min_value=1.2,
    max_value=2.3,
    value=1.65
)

max_bpm = st.number_input(
    "Maximum BPM",
    min_value=80,
    max_value=220,
    value=185
)

avg_bpm = st.number_input(
    "Average BPM",
    min_value=50,
    max_value=200,
    value=150
)

resting_bpm = st.number_input(
    "Resting BPM",
    min_value=30,
    max_value=120,
    value=65
)

session_duration = st.number_input(
    "Session Duration (hours)",
    min_value=0.1,
    max_value=5.0,
    value=1.2,
    step=0.1
)

workout_type = st.selectbox(
    "Workout Type",
    ["Yoga", "HIIT", "Cardio", "Strength"]
)

fat_percentage = st.number_input(
    "Fat Percentage",
    min_value=5.0,
    max_value=60.0,
    value=25.0
)

water_intake = st.number_input(
    "Water Intake (liters)",
    min_value=0.1,
    max_value=10.0,
    value=2.5,
    step=0.1
)

workout_frequency = st.number_input(
    "Workout Frequency (days/week)",
    min_value=1,
    max_value=7,
    value=4
)

experience_level = st.selectbox(
    "Experience Level",
    [1, 2, 3]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=23.88
)


if st.button("Predict Calories"):

    payload = {
        "Age": age,
        "Gender": gender,
        "Weight_kg": weight,
        "Height_m": height,
        "Max_BPM": max_bpm,
        "Avg_BPM": avg_bpm,
        "Resting_BPM": resting_bpm,
        "Session_Duration_hours": session_duration,
        "Workout_Type": workout_type,
        "Fat_Percentage": fat_percentage,
        "Water_Intake_liters": water_intake,
        "Workout_Frequency_days_per_week":
            workout_frequency,
        "Experience_Level": experience_level,
        "BMI": bmi
    }

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=10
        )

        if response.status_code == 200:

            result = response.json()

            prediction = result[
                "predicted_calories_burned"
            ]

            st.success(
                f"Estimated Calories Burned: "
                f"{prediction:.2f} kcal"
            )

        else:

            st.error(
                f"API Error: {response.status_code}"
            )

            st.write(response.json())

    except requests.exceptions.RequestException:

        st.error(
            "Could not connect to the FastAPI server. "
            "Make sure the backend is running."
        )
