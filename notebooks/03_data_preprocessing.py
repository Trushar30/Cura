# %% [markdown]
"""
# ⚙️ Notebook 03: Data Preprocessing & Feature Engineering
## Healthcare Recommendation System — Phase 3

This notebook implements the data cleaning, feature engineering, and matrix building steps:
1. Standardizes symptom names and aligns severity weights.
2. Computes per-row symptom counts and total severity scores.
3. Performs TF-IDF vectorization on disease descriptions and saves resources.
4. Pivots drug reviews to build a condition-drug user-item matrix.
5. Saves clean splits to `data/processed/`.
"""

# %%
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

# %% [markdown]
"""
## 1. Load Data & Data Cleaning
"""

# %%
# Load
training = pd.read_csv('data/raw/Training.csv')
testing = pd.read_csv('data/raw/Testing.csv')
severity = pd.read_csv('data/raw/Symptom-severity.csv')
description = pd.read_csv('data/raw/description.csv')
reviews = pd.read_csv('data/raw/drug_reviews.tsv', sep='\t')

# Drop Unnamed
for df in [training, testing, severity, description, reviews]:
    unnamed = [c for c in df.columns if 'Unnamed' in c]
    if unnamed:
        df.drop(columns=unnamed, inplace=True)

# Standardize prognosis / disease columns
training.rename(columns={'prognosis': 'Disease'}, inplace=True)
testing.rename(columns={'prognosis': 'Disease'}, inplace=True)

# Standardize name string values (strip and title-case)
training['Disease'] = training['Disease'].astype(str).str.strip().str.title()
testing['Disease'] = testing['Disease'].astype(str).str.strip().str.title()
description['Disease'] = description['Disease'].astype(str).str.strip().str.title()

# Clean severity symptoms
severity['Symptom'] = severity['Symptom'].astype(str).str.replace('_', ' ').str.strip().str.lower()

# %% [markdown]
"""
## 2. Feature Engineering
"""

# %%
# Build a dictionary for symptom-severity lookup
# Standardize keys by replacing underscores with spaces, stripping and lowercasing
severity_dict = dict(zip(severity['Symptom'], severity['weight']))

# Map symptom columns in training to their weights
symptom_cols = [c for c in training.columns if c != 'Disease']
cleaned_symptom_cols = [c.replace('_', ' ').strip().lower() for c in symptom_cols]

# Create mapping dictionary from original col name to weight
symptom_weight_map = {}
for orig_col, clean_col in zip(symptom_cols, cleaned_symptom_cols):
    symptom_weight_map[orig_col] = severity_dict.get(clean_col, 0)

print(f"Mapped {len([k for k, v in symptom_weight_map.items() if v > 0])} symptoms to non-zero severity weights.")

# %%
# Feature engineering on training.csv
# A. symptom_count: number of symptoms per row
training['symptom_count'] = training[symptom_cols].sum(axis=1)
testing['symptom_count'] = testing[symptom_cols].sum(axis=1)

# B. total_severity_score: sum of weights for active symptoms
def calculate_severity(row):
    return sum(row[col] * symptom_weight_map[col] for col in symptom_cols)

training['total_severity_score'] = training.apply(calculate_severity, axis=1)
testing['total_severity_score'] = testing.apply(calculate_severity, axis=1)

# C. disease_risk_level: Low/Medium/High based on severity score
# We determine bins based on severity distribution
sev_mean = training['total_severity_score'].mean()
sev_std = training['total_severity_score'].std()

def risk_level(score):
    if score < (sev_mean - 0.5 * sev_std):
        return 'Low'
    elif score > (sev_mean + 0.5 * sev_std):
        return 'High'
    else:
        return 'Medium'

training['disease_risk_level'] = training['total_severity_score'].apply(risk_level)
testing['disease_risk_level'] = testing['total_severity_score'].apply(risk_level)

# %%
# Drug review features
reviews['review_length'] = reviews['review'].fillna('').apply(len)
reviews['word_count'] = reviews['review'].fillna('').apply(lambda x: len(x.split()))

print("\nEngineered Training Data Sample:")
print(training[['Disease', 'symptom_count', 'total_severity_score', 'disease_risk_level']].head(5))

# Save engineered datasets
training.to_csv('data/processed/training_engineered.csv', index=False)
testing.to_csv('data/processed/testing_engineered.csv', index=False)
print("Saved engineered training & testing datasets.")

# %% [markdown]
"""
## 3. TF-IDF on Disease Descriptions
"""

# %%
# Create Combined TF-IDF Feature representations for Content-Based filtering
description['Description_Clean'] = description['Description'].fillna('')
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = vectorizer.fit_transform(description['Description_Clean'])

# Save vectorizer & matrix
joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
joblib.dump(tfidf_matrix, 'models/tfidf_matrix.joblib')
print("Saved TF-IDF Vectorizer and TF-IDF Matrix to models/")

# %% [markdown]
"""
## 4. Build User-Item Interaction Matrix
"""

# %%
# Create a user-item matrix of ratings (or utility matrix)
# uniqueID maps to user, drugName maps to item. Since condition links them to disease,
# we group or build the pivoted utility matrix condition-drug rating
user_item_matrix = reviews.pivot_table(
    index='condition', 
    columns='drugName', 
    values='rating', 
    aggfunc='mean'
).fillna(0)

print(f"\nUser-Item Matrix (Condition x DrugName) shape: {user_item_matrix.shape}")
user_item_matrix.to_csv('data/processed/user_item_matrix.csv')
print("Saved user_item_matrix.csv to data/processed/")

# %% [markdown]
"""
## 5. Train/Test Split on Drug Reviews
"""

# %%
train_reviews, test_reviews = train_test_split(reviews, test_size=0.2, random_state=42)
train_reviews.to_csv('data/processed/reviews_train.csv', index=False)
test_reviews.to_csv('data/processed/reviews_test.csv', index=False)
print(f"Saved Train/Test splits. Train: {train_reviews.shape[0]} rows, Test: {test_reviews.shape[0]} rows")

print("\n✅ Preprocessing phase completed successfully!")
