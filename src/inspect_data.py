import pandas as pd

df = pd.read_csv("data/raw/gym_members_exercise_tracking.csv")

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- SHAPE ---")
print(df.shape)

print("\n--- COLUMNS ---")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DUPLICATES ---")
print(df.duplicated().sum())

print("\n--- STATISTICAL SUMMARY ---")
print(df.describe(include="all"))
