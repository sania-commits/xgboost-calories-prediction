import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/raw/gym_members_exercise_tracking.csv")

# Basic information
print("\n--- DATASET INFO ---")
print(df.info())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

print("\n--- TARGET STATISTICS ---")
print(df["Calories_Burned"].describe())

print("\n--- CATEGORICAL VALUES ---")
print("\nGender:")
print(df["Gender"].value_counts())

print("\nWorkout Type:")
print(df["Workout_Type"].value_counts())

# Histograms
df.hist(figsize=(15, 10))
plt.tight_layout()
plt.show()

# Calories distribution
plt.figure(figsize=(8, 5))
plt.hist(df["Calories_Burned"], bins=30)
plt.xlabel("Calories Burned")
plt.ylabel("Frequency")
plt.title("Distribution of Calories Burned")
plt.show()

# Calories vs Session Duration
plt.figure(figsize=(8, 5))
plt.scatter(
    df["Session_Duration (hours)"],
    df["Calories_Burned"]
)
plt.xlabel("Session Duration (hours)")
plt.ylabel("Calories Burned")
plt.title("Session Duration vs Calories Burned")
plt.show()
