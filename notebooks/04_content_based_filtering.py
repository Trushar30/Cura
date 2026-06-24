# %% [markdown]
"""
# 🧠 Notebook 04: Content-Based Filtering Recommendation Engine
## Healthcare Recommendation System — Phase 4

This notebook builds a Content-Based Filtering recommendation engine.
It leverages disease descriptions, medications, and diets, combining them into TF-IDF representations.
We then use cosine similarity to recommend similar diseases and map user-entered symptoms to potential conditions.
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.makedirs('reports', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Set styling
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')

# %% [markdown]
"""
## 1. Load Data & Create Features
"""

# %%
# Load cleaned/merged data
unified_df = pd.read_csv('data/processed/unified_healthcare_data.csv')
training_df = pd.read_csv('data/raw/Training.csv')
training_df.rename(columns={'prognosis': 'Disease'}, inplace=True)
training_df['Disease'] = training_df['Disease'].astype(str).str.strip().str.title()

# Concat Description, Medication, Diet into a single text block for TF-IDF
unified_df['combined_features'] = (
    unified_df['Description'].fillna('') + " " +
    unified_df['Medication'].fillna('') + " " +
    unified_df['Diet'].fillna('')
)

print("Unified Disease dataframe columns:", unified_df.columns.tolist())
print(f"Number of diseases available: {len(unified_df)}")

# %% [markdown]
"""
## 2. Fit TF-IDF and Compute Similarity
"""

# %%
# Vectorize
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(unified_df['combined_features'])

# Save vectorizer and matrix for production use
joblib.dump(vectorizer, 'models/cb_vectorizer.joblib')
joblib.dump(tfidf_matrix, 'models/cb_tfidf_matrix.joblib')

# Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
joblib.dump(cosine_sim, 'models/cb_cosine_similarity.joblib')

print("TF-IDF Matrix shape:", tfidf_matrix.shape)
print("Cosine Similarity matrix shape:", cosine_sim.shape)

# %% [markdown]
"""
## 3. Recommendation Function (Disease to Disease)
"""

# %%
# Mapping from disease name to index
disease_to_idx = pd.Series(unified_df.index, index=unified_df['Disease']).to_dict()
idx_to_disease = unified_df['Disease'].to_dict()

def recommend_similar_diseases(disease_name, num_recommendations=5):
    disease_name = disease_name.strip().title()
    if disease_name not in disease_to_idx:
        return f"Disease '{disease_name}' not found in the dataset."
    
    idx = disease_to_idx[disease_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort by similarity score in descending order (exclude itself)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    
    recoms = []
    for i, score in sim_scores:
        recoms.append({
            'Recommended Disease': idx_to_disease[i],
            'Similarity Score': round(score, 4),
            'Description': unified_df.iloc[i]['Description'],
            'Medications': unified_df.iloc[i]['Medication'],
            'Diet': unified_df.iloc[i]['Diet']
        })
    return pd.DataFrame(recoms)

# Test recommendations for 5 sample diseases
test_diseases = ['Fungal Infection', 'Allergy', 'Gerd', 'Malaria', 'Diabetes ']
for d in test_diseases:
    print("\n" + "="*80)
    print(f"Recommendations for: {d}")
    print("="*80)
    res = recommend_similar_diseases(d, 3)
    if isinstance(res, pd.DataFrame):
        print(res.to_string(index=False))
    else:
        print(res)

# %% [markdown]
"""
## 4. Symptom-Based Recommendation
"""

# %%
# Extract symptom names
symptom_cols = [c for c in training_df.columns if c != 'Disease']

# Implement recommend_for_symptoms
def recommend_for_symptoms(symptom_list, num_recommendations=3):
    # Standardize symptom inputs
    standardized_symptoms = [s.replace('_', ' ').strip().lower() for s in symptom_list]
    
    # Map symptoms to original column format
    matched_cols = []
    for s in standardized_symptoms:
        for col in symptom_cols:
            if col.replace('_', ' ').strip().lower() == s:
                matched_cols.append(col)
                break
    
    if not matched_cols:
        return "No matching symptoms found in the training dataset."
        
    print(f"Matched input symptoms to columns: {matched_cols}")
    
    # Create input binary symptom vector
    input_vector = np.zeros(len(symptom_cols))
    for col in matched_cols:
        idx = symptom_cols.index(col)
        input_vector[idx] = 1
        
    # Find matching disease in training.csv using cosine similarity of symptom vectors
    X_train = training_df[symptom_cols].values
    similarities = cosine_similarity(input_vector.reshape(1, -1), X_train).flatten()
    
    # Get index of best matching training row
    best_row_idx = np.argmax(similarities)
    best_match_score = similarities[best_row_idx]
    matched_disease = training_df.iloc[best_row_idx]['Disease']
    
    print(f"Closest matched disease: '{matched_disease}' (Similarity score: {best_match_score:.4f})")
    
    # Return recommendations for this disease
    return recommend_similar_diseases(matched_disease, num_recommendations)

# Test symptom-based recommendations
print("\n" + "="*80)
print("Testing recommendations for symptoms: ['itching', 'skin_rash', 'nodal_skin_eruptions']")
print("="*80)
symptom_recs = recommend_for_symptoms(['itching', 'skin_rash', 'nodal_skin_eruptions'])
if isinstance(symptom_recs, pd.DataFrame):
    print(symptom_recs.to_string(index=False))
else:
    print(symptom_recs)

# %% [markdown]
"""
## 5. Disease Similarity Heatmap
"""

# %%
# Similarity Heatmap for top 15 diseases
top_n = 15
sample_diseases = unified_df['Disease'].head(top_n).tolist()
sample_indices = [disease_to_idx[d] for d in sample_diseases]

sample_sim = cosine_sim[np.ix_(sample_indices, sample_indices)]

plt.figure(figsize=(12, 10))
sns.heatmap(sample_sim, annot=True, xticklabels=sample_diseases, yticklabels=sample_diseases, cmap='YlGnBu', fmt='.2f')
plt.title(f'Cosine Similarity Heatmap (Top {top_n} Diseases)', fontsize=16, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('reports/cb_disease_similarity_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n📊 Saved similarity heatmap to reports/cb_disease_similarity_heatmap.png")
