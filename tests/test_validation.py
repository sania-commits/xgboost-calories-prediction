from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_invalid_age():
    payload = {
        "Age": -10,
        "Gender": "Female",
        "Weight_kg": 65.0,
        "Height_m": 1.65,
        "Max_BPM": 185,
        "Avg_BPM": 150,
        "Resting_BPM": 65,
        "Session_Duration_hours": 1.2,
        "Workout_Type": "Cardio",
        "Fat_Percentage": 25.0,
        "Water_Intake_liters": 2.5,
        "Workout_Frequency_days_per_week": 4,
        "Experience_Level": 2,
        "BMI": 23.88
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
