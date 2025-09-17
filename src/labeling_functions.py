import pandas as pd
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.model import LabelModel

# Load data
try:
    df = pd.read_csv('data/processed/sample_2020_labeled.csv')
except FileNotFoundError:
    print("Error: 'data/processed/sample_2020_labeled.csv' not found. Please run src/data_prep.py first.")
    exit(1)

# Define labels
ABSTAIN = -1
ADE = 1
NO_ADE = 0
DRUG = 2
NO_DRUG = 0

# Labeling functions
@labeling_function()
def lf_ade_keywords(x):
    keywords = ["fever", "chills", "pain", "headache", "shivering", "rash", "fatigue", "vertigo", "nausea", "dizziness"]
    return ADE if any(keyword in x.SYMPTOM_TEXT.lower() for keyword in keywords) else NO_ADE

@labeling_function()
def lf_drug_keywords(x):
    keywords = ["flu shot", "vaccine", "pfizer", "moderna", "astrazeneca", "heparin", "tylenol"]
    return DRUG if any(keyword in x.SYMPTOM_TEXT.lower() for keyword in keywords) else NO_DRUG

@labeling_function()
def lf_severe_symptoms(x):
    severe_keywords = ["stroke", "seizure", "anaphylaxis", "paralysis", "heart attack"]
    return ADE if any(keyword in x.SYMPTOM_TEXT.lower() for keyword in severe_keywords) else NO_ADE

# Apply labeling functions
lfs = [lf_ade_keywords, lf_drug_keywords, lf_severe_symptoms]
applier = PandasLFApplier(lfs=lfs)
try:
    L_train = applier.apply(df)
except Exception as e:
    print(f"Error applying labeling functions: {e}")
    exit(1)

# Combine labels
label_model = LabelModel(cardinality=3, verbose=True)
label_model.fit(L_train, n_epochs=500, seed=42)
labels = label_model.predict(L_train)

# Add labels to dataframe
df['ADE_LABEL'] = [1 if x == 1 else 0 for x in labels]
df['DRUG_LABEL'] = [1 if x == 2 else 0 for x in labels]

# Save weak labels
df.to_csv('data/processed/sample_2020_weak_labeled.csv', index=False)
print("Saved weak labels to data/processed/sample_2020_weak_labeled.csv")

# Save a 100-row subset for manual review
subset = df.sample(n=100, random_state=42)[['VAERS_ID', 'SYMPTOM_TEXT', 'ADE_LABEL', 'DRUG_LABEL']]
subset.to_csv('data/processed/sample_2020_gold_subset.csv', index=False)
print("Saved 100-row subset for manual review to data/processed/sample_2020_gold_subset.csv")
