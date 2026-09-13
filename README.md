# 🔥 XGBoost Calories Burned Prediction System

An end-to-end Machine Learning application that predicts calories burned during a workout using **XGBoost**.

The project demonstrates a complete ML engineering workflow including data exploration, preprocessing, model training, evaluation, REST API development, frontend integration, containerization, automated testing, logging, and CI using GitHub Actions.

---

## 🚀 Project Overview

The system takes workout and physiological information such as age, gender, weight, heart rate, workout duration, workout type, body fat percentage, BMI, and experience level and predicts the estimated number of calories burned.

The trained preprocessing and XGBoost model are packaged into a single reusable Scikit-learn pipeline and served through a FastAPI REST API.

A Streamlit frontend communicates with the API and provides an interactive interface for generating predictions.

---

## 🏗️ System Architecture

```text
User
 │
 ▼
Streamlit Frontend
 │
 │ HTTP POST /predict
 ▼
FastAPI REST API
 │
 ▼
Preprocessing Pipeline
 │
 ├── Numerical Features
 │
 └── One-Hot Encoded Categorical Features
 │
 ▼
XGBoost Regressor
 │
 ▼
Calories Burned Prediction
```

The FastAPI backend and Streamlit frontend run as separate Docker containers and communicate through the Docker Compose network.

---

## 📊 Model Performance

The XGBoost regression model achieved the following performance on the held-out test set:

| Metric | Score |
|---|---:|
| R² | 0.9887 |
| MAE | 19.78 kcal |
| RMSE | 30.67 kcal |

### Cross-Validation

5-fold cross-validation produced:

```text
Mean R²: 0.99145
Standard Deviation: 0.00171
```

The consistently high cross-validation scores indicate stable performance across the evaluated folds.

---

## 🧠 Machine Learning Pipeline

The project uses a Scikit-learn `Pipeline` and `ColumnTransformer` to keep preprocessing and prediction together.

### Numerical Features

- Age
- Weight
- Height
- Maximum BPM
- Average BPM
- Resting BPM
- Session Duration
- Fat Percentage
- Water Intake
- Workout Frequency
- Experience Level
- BMI

### Categorical Features

- Gender
- Workout Type

Categorical variables are transformed using `OneHotEncoder(handle_unknown="ignore")`.

The complete preprocessing pipeline and trained XGBoost model are serialized using Joblib.

---

## 🛠️ Tech Stack

**Machine Learning**

- Python
- Pandas
- Scikit-learn
- XGBoost
- Joblib

**Backend**

- FastAPI
- Pydantic
- Uvicorn

**Frontend**

- Streamlit
- Requests

**DevOps & Testing**

- Docker
- Docker Compose
- Pytest
- Git
- GitHub
- GitHub Actions

**Data Analysis**

- Jupyter Notebook
- Matplotlib

---

## 📁 Project Structure

```text
xgboost-calories-prediction/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── api/
│   └── main.py
│
├── data/
│   └── raw/
│       └── gym_members_exercise_tracking.csv
│
├── frontend/
│   ├── Dockerfile
│   ├── app.py
│   └── requirements.txt
│
├── models/
│   └── calories_xgboost_pipeline.pkl
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── config.py
│   ├── correlation.py
│   ├── data_check.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── inspect_data.py
│   ├── predict.py
│   └── train.py
│
├── tests/
│   ├── test_api.py
│   ├── test_model.py
│   └── test_validation.py
│
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 📈 Exploratory Data Analysis

EDA was performed to understand feature distributions, data quality, and relationships with the target variable.

The dataset was checked for:

- Missing values
- Duplicate records
- Feature distributions
- Categorical variables
- Correlations
- Target relationships

Session duration showed the strongest linear relationship with calories burned in the correlation analysis.

---

## ⚙️ Model Training

Run the training pipeline with:

```bash
python src/train.py
```

The training process:

1. Loads the workout dataset.
2. Separates features and target.
3. Creates preprocessing transformations.
4. Splits the dataset into training and testing sets.
5. Trains an XGBoost regressor.
6. Evaluates model performance.
7. Saves the complete trained pipeline.

---

## 🌐 FastAPI REST API

Start the API locally:

```bash
uvicorn api.main:app --reload
```

The API is available at:

```text
http://127.0.0.1:8000
```

Interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API status message |
| GET | `/health` | Health check |
| POST | `/predict` | Generate calorie prediction |

### Example Prediction Request

```json
{
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
```

Example response:

```json
{
  "predicted_calories_burned": 914.85
}
```

---

## 🖥️ Streamlit Frontend

Run the frontend locally after starting the API:

```bash
streamlit run frontend/app.py
```

Then open:

```text
http://localhost:8501
```

The frontend collects workout information and sends it to the FastAPI `/predict` endpoint.

---

## 🐳 Docker

The backend and frontend are containerized separately.

Build and start the complete application:

```bash
docker compose up --build
```

Services:

```text
FastAPI API      → localhost:8000
Streamlit UI     → localhost:8501
```

Stop the containers:

```bash
docker compose down
```

---

## 🧪 Automated Testing

The project includes tests for:

- API endpoints
- Input validation
- Model inference

Run the test suite:

```bash
pytest -v
```

---

## ⚙️ Continuous Integration

GitHub Actions automatically runs the test suite when code is pushed to the `main` branch or when a pull request targets `main`.

The CI workflow:

```text
Checkout Repository
        ↓
Set Up Python 3.12
        ↓
Install Dependencies
        ↓
Run Pytest
        ↓
Pass / Fail
```

This helps detect regressions before new code is integrated.

---

## 📝 Application Logging

The FastAPI backend includes application logging for model loading and prediction requests.

Example:

```text
ML model loaded successfully
Prediction request received
Prediction completed successfully
```

Prediction failures are handled by the API and logged for debugging.

---

## 💻 Local Installation

Clone the repository:

```bash
git clone https://github.com/sania-commits/xgboost-calories-prediction.git
cd xgboost-calories-prediction
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest -v
```

---

## 🔮 Future Improvements

Planned improvements include:

- Cloud deployment
- Live application URL
- Model monitoring
- Additional model experimentation
- Enhanced frontend visualization
- Production deployment configuration

---

## 👩‍💻 Author

**Sania Anjum**

Data Science | Machine Learning | Generative AI

GitHub: [sania-commits](https://github.com/sania-commits)

---

## ⭐ Project Purpose

This project was developed as an end-to-end ML engineering portfolio project demonstrating the transition from raw data and model experimentation to a tested, containerized, API-driven machine learning application.
