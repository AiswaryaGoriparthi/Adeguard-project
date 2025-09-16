import pandas as pd

# Load 2020 VAERS data
data = pd.read_csv('data/raw/2020VAERSDATA.csv', encoding='ISO-8859-1')

# Basic exploration
print("Total rows:", len(data))
print("Columns:", data.columns.tolist())
print("Sample SYMPTOM_TEXT:", data['SYMPTOM_TEXT'].head(5).tolist())
print("Missing SYMPTOM_TEXT:", data['SYMPTOM_TEXT'].isna().sum())
print("Missing AGE_YRS:", data['AGE_YRS'].isna().sum())

# Save a small sample for inspection
sample = data[['VAERS_ID', 'SYMPTOM_TEXT', 'AGE_YRS']].head(10)
sample.to_csv('data/processed/sample_2020.csv', index=False)
print("Saved sample to data/processed/sample_2020.csv")





