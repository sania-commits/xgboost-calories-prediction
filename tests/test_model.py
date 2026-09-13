import joblib
import pandas as pd


MODEL_PATH = "models/calories_xgboost_pipeline.pkl"


def test_model_prediction():
    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame(
        [
            {
                "Age": 30,
                "Gender": "Female",
                "Weight (kg)": 65.0,
                "Height (m)": 1.65,
                "Max_BPM": 185,
                "Avg_BPM": 150,
                "Resting_BPM": 65,
                "Session_Duration (hours)": 1.2,
                "Workout_Type": "Cardio",
                "Fat_Percentage": 25.0,
                "Water_Intake (liters)": 2.5,
                "Workout_Frequency (days/week)": 4,
                "Experience_Level": 2,
                "BMI": 23.88,
            }
        ]
    )

    prediction = model.predict(sample)

    assert len(prediction) == 1
    assert prediction[0] > 0
