from typing import Literal
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib


# --------------------------------
# Logging configuration
# --------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------
# Load trained ML pipeline
# --------------------------------

MODEL_PATH = "models/calories_xgboost_pipeline.pkl"

try:
    model = joblib.load(MODEL_PATH)
    logger.info("ML model loaded successfully from %s", MODEL_PATH)

except Exception:
    logger.exception("Failed to load ML model.")
    raise


# --------------------------------
# Create FastAPI application
# --------------------------------

app = FastAPI(
    title="Calories Burned Prediction API",
    description="XGBoost API for predicting calories burned during a workout.",
    version="1.0.0"
)


# --------------------------------
# Input schema
# --------------------------------

class WorkoutData(BaseModel):

    Age: int = Field(gt=0, le=100)

    Gender: Literal["Male", "Female"]

    Weight_kg: float = Field(gt=0)

    Height_m: float = Field(gt=0)

    Max_BPM: int = Field(gt=0)

    Avg_BPM: int = Field(gt=0)

    Resting_BPM: int = Field(gt=0)

    Session_Duration_hours: float = Field(gt=0)

    Workout_Type: Literal[
        "Yoga",
        "HIIT",
        "Cardio",
        "Strength"
    ]

    Fat_Percentage: float = Field(gt=0)

    Water_Intake_liters: float = Field(gt=0)

    Workout_Frequency_days_per_week: int = Field(
        ge=1,
        le=7
    )

    Experience_Level: int = Field(
        ge=1,
        le=3
    )

    BMI: float = Field(gt=0)


# --------------------------------
# Home endpoint
# --------------------------------

@app.get("/")
def home():

    return {
        "message": "Calories Burned Prediction API is running"
    }


# --------------------------------
# Health endpoint
# --------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# --------------------------------
# Prediction endpoint
# --------------------------------

@app.post("/predict")
def predict(data: WorkoutData):

    logger.info(
        "Prediction request received | workout_type=%s | age=%s",
        data.Workout_Type,
        data.Age
    )

    try:

        input_df = pd.DataFrame([{

            "Age": data.Age,

            "Gender": data.Gender,

            "Weight (kg)": data.Weight_kg,

            "Height (m)": data.Height_m,

            "Max_BPM": data.Max_BPM,

            "Avg_BPM": data.Avg_BPM,

            "Resting_BPM": data.Resting_BPM,

            "Session_Duration (hours)":
                data.Session_Duration_hours,

            "Workout_Type": data.Workout_Type,

            "Fat_Percentage":
                data.Fat_Percentage,

            "Water_Intake (liters)":
                data.Water_Intake_liters,

            "Workout_Frequency (days/week)":
                data.Workout_Frequency_days_per_week,

            "Experience_Level":
                data.Experience_Level,

            "BMI": data.BMI

        }])

        prediction = model.predict(input_df)[0]

        prediction_value = round(
            float(prediction),
            2
        )

        logger.info(
            "Prediction completed successfully | calories=%s",
            prediction_value
        )

        return {
            "predicted_calories_burned":
                prediction_value
        }

    except Exception:

        logger.exception(
            "Error occurred during calorie prediction."
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to generate prediction."
        )