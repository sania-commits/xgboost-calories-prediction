import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/gym_members_exercise_tracking.csv")

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

print("\n--- UNIQUE VALUES ---")

for column in df.columns:
    print(f"\n{column}:")
    print(df[column].unique())

print("\n--- CATEGORICAL COLUMNS ---")
print(df.select_dtypes(include="object").columns.tolist())

print("\n--- NUMERICAL COLUMNS ---")
print(df.select_dtypes(include="number").columns.tolist())