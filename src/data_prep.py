
import pandas as pd

# Define dtypes to avoid DtypeWarning
dtypes = {
    'VAERS_ID': 'int32',
    'AGE_YRS': 'float32',
    'SYMPTOM_TEXT': 'str',
    'L_THREAT': 'str',  # Column 12
    'X_STAY': 'str',    # Column 15
    # Add other columns as needed
}

# Load data with dtypes
data = pd.read_csv('data/raw/2020VAERSDATA.csv', encoding='ISO-8859-1', dtype=dtypes, low_memory=False)

# Clean data
data = data.drop_duplicates(subset=['VAERS_ID'])  # Remove duplicate reports
data = data.dropna(subset=['SYMPTOM_TEXT'])       # Drop rows with missing SYMPTOM_TEXT

# Bucket ages
def bucket_age(age):
    if pd.isna(age):
        return 'Unknown'
    elif age < 18:
        return 'Kids'
    elif 18 <= age <= 65:
        return 'Adults'
    else:
        return 'Seniors'

data['AGE_GROUP'] = data['AGE_YRS'].apply(bucket_age)

# Basic exploration
print("Total rows after cleaning:", len(data))
print("Columns:", data.columns.tolist())
print("Sample SYMPTOM_TEXT:", data['SYMPTOM_TEXT'].head(5).tolist())
print("Missing SYMPTOM_TEXT:", data['SYMPTOM_TEXT'].isna().sum())
print("Missing AGE_YRS:", data['AGE_YRS'].isna().sum())
print("Age group counts:", data['AGE_GROUP'].value_counts().to_dict())

# Save a sample for labeling (2,000 rows)
sample = data[['VAERS_ID', 'SYMPTOM_TEXT', 'AGE_YRS', 'AGE_GROUP']].sample(n=2000, random_state=42)
sample.to_csv('data/processed/sample_2020_labeled.csv', index=False)
print("Saved sample to data/processed/sample_2020_labeled.csv")






