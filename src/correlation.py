import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/raw/gym_members_exercise_tracking.csv")

# Select numerical columns
numeric_df = df.select_dtypes(include="number")

# Calculate correlation
correlation = numeric_df.corr()

print("\n--- CORRELATION WITH CALORIES BURNED ---")
print(
    correlation["Calories_Burned"]
    .sort_values(ascending=False)
)

# Plot correlation with target
target_corr = correlation["Calories_Burned"].sort_values()

plt.figure(figsize=(8, 6))
target_corr.plot(kind="barh")
plt.xlabel("Correlation")
plt.ylabel("Feature")
plt.title("Feature Correlation with Calories Burned")
plt.tight_layout()
plt.show()