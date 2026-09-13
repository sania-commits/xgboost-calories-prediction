import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


# -----------------------------
# 1. Load dataset
# -----------------------------

df = pd.read_csv(
    "data/raw/gym_members_exercise_tracking.csv"
)


# -----------------------------
# 2. Separate features and target
# -----------------------------

X = df.drop("Calories_Burned", axis=1)

y = df["Calories_Burned"]


# -----------------------------
# 3. Identify column types
# -----------------------------

categorical_features = [
    "Gender",
    "Workout_Type"
]

numerical_features = [
    column for column in X.columns
    if column not in categorical_features
]


# -----------------------------
# 4. Preprocessing
# -----------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# -----------------------------
# 5. XGBoost model
# -----------------------------

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)


# -----------------------------
# 6. Create complete pipeline
# -----------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# -----------------------------
# 7. Train-test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# -----------------------------
# 8. Train model
# -----------------------------

pipeline.fit(X_train, y_train)

# -----------------------------
# Save trained pipeline
# -----------------------------

os.makedirs("models", exist_ok=True)

model_path = "models/calories_xgboost_pipeline.pkl"

joblib.dump(
    pipeline,
    model_path
)

print(f"\nModel saved to: {model_path}")


# -----------------------------
# 9. Make predictions
# -----------------------------

y_pred = pipeline.predict(X_test)


# -----------------------------
# 10. Evaluate model
# -----------------------------

mae = mean_absolute_error(y_test, y_pred)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(y_test, y_pred)


print("\n--- MODEL PERFORMANCE ---")

print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

import matplotlib.pyplot as plt

# -----------------------------
# Cross Validation
# -----------------------------

cv_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
    scoring="r2"
)

print("\n--- 5-FOLD CROSS VALIDATION ---")
print("R2 scores:", cv_scores)
print("Mean R2 :", cv_scores.mean())
print("Std R2  :", cv_scores.std())

# -----------------------------
# 11. Actual vs Predicted
# -----------------------------

plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Calories")
plt.ylabel("Predicted Calories")
plt.title("Actual vs Predicted Calories")

plt.tight_layout()
plt.show()


# -----------------------------
# 12. Feature Importance
# -----------------------------

feature_names = pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

importances = pipeline.named_steps[
    "model"
].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n--- FEATURE IMPORTANCE ---")
print(importance_df)

top_features = importance_df.head(15)

plt.figure(figsize=(9, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance")

plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()