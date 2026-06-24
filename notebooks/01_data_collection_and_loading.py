# %% [markdown]
"""
# 📋 Notebook 01: Data Collection & Loading
## Healthcare Recommendation System — Phase 1 & 2

This notebook loads all raw open-source datasets (Kaggle/GitHub), performs initial inspection,
cleans column names (dropping unnecessary index/unnamed columns), merges disease features,
and exports the unified dataset for downstream models.
"""

# %%
import pandas as pd
import numpy as np
import os

# Create directories
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

print("=" * 70)
# %%
# Load all datasets and log basic dimensions
raw_dir = 'data/raw'
datasets = {}

file_list = [
    ('Training.csv', ','),
    ('Testing.csv', ','),
    ('Symptom-severity.csv', ','),
    ('description.csv', ','),
    ('medications.csv', ','),
    ('diets.csv', ','),
    ('workout_df.csv', ','),
    ('precautions_df.csv', ','),
    ('symptoms_df.csv', ','),
    ('symptom_Description.csv', ','),
    ('symptom_precaution.csv', ','),
    ('drug_reviews.tsv', '\t')
]

for filename, sep in file_list:
    filepath = os.path.join(raw_dir, filename)
    name = filename.split('.')[0]
    
    if name == 'symptom_Description':
        # Handled as no-header CSV
        df = pd.read_csv(filepath, header=None, names=['Disease', 'Description'], sep=sep)
    elif name == 'symptom_precaution':
        # Handled as no-header CSV
        df = pd.read_csv(filepath, header=None, names=['Disease', 'Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4'], sep=sep)
    else:
        df = pd.read_csv(filepath, sep=sep)
        
    datasets[name] = df
    print(f"Loaded {filename}: shape {df.shape}")

# %% [markdown]
"""
## Clean and Standardize Data
"""

# %%
# Drop Unnamed / Index columns from datasets if they exist
for name, df in datasets.items():
    unnamed_cols = [c for c in df.columns if 'Unnamed' in c]
    if unnamed_cols:
        df.drop(columns=unnamed_cols, inplace=True)
        print(f"Dropped index columns {unnamed_cols} from {name}")

# Standardize column headers and values
# We will clean the 'Disease' / 'disease' / 'prognosis' column names to be standardized
if 'Training' in datasets:
    datasets['Training'].rename(columns={'prognosis': 'Disease'}, inplace=True)
if 'Testing' in datasets:
    datasets['Testing'].rename(columns={'prognosis': 'Disease'}, inplace=True)
if 'workout_df' in datasets:
    datasets['workout_df'].rename(columns={'disease': 'Disease'}, inplace=True)

# %%
# Strip whitespace and standardize disease names to title case across key dataframes
for name in ['Training', 'Testing', 'description', 'medications', 'diets', 'workout_df', 'precautions_df', 'symptoms_df', 'symptom_Description', 'symptom_precaution']:
    if name in datasets and 'Disease' in datasets[name].columns:
        datasets[name]['Disease'] = datasets[name]['Disease'].astype(str).str.strip().str.title()

# %% [markdown]
"""
## Merge Disease Datasets
"""

# %%
# We merge description, precautions, medications, diets into a single unified dataframe
desc = datasets['description']
prec = datasets['precautions_df']
meds = datasets['medications']
diets = datasets['diets']

# Merge on 'Disease'
unified_df = desc.merge(prec, on='Disease', how='left')
unified_df = unified_df.merge(meds, on='Disease', how='left')
unified_df = unified_df.merge(diets, on='Disease', how='left')

# Drop any duplicate rows
unified_df.drop_duplicates(subset=['Disease'], inplace=True)

print(f"\nUnified Healthcare Data Shape: {unified_df.shape}")
print(unified_df.head(5))

# Save processed unified dataset
unified_df.to_csv('data/processed/unified_healthcare_data.csv', index=False)
print("Saved unified healthcare data to data/processed/unified_healthcare_data.csv")

# Save processed user_item matrix placeholder if we need it
# %%
print("\n" + "="*50)
print("DATA COLLECTION AND LOADING SUMMARY")
print("="*50)
for name, df in datasets.items():
    print(f"Dataset '{name}': {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print("-" * 30)

print("\nFinished Phase 1 successfully!")
