from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Calories Burned Prediction API is running"
    }


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy"
    }


def test_prediction():
    payload = {
        "Age": 30,
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

    assert response.status_code == 200

    data = response.json()

    assert "predicted_calories_burned" in data
    assert isinstance(
        data["predicted_calories_burned"],
        (int, float)
    )
    assert data["predicted_calories_burned"] > 0
