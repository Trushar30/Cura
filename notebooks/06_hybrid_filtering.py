# %% [markdown]
"""
# 🔀 Notebook 06: Hybrid Filtering Recommendation Engine
## Healthcare Recommendation System — Phase 4

This notebook builds a Hybrid Recommendation Engine that blends:
1. Content-Based Filtering (Cosine similarity of condition features).
2. Collaborative Filtering (SVD matrix factorization predictions).

We implement both a weighted scoring hybrid and a switching hybrid, evaluate them,
and output recommendations for sample cases.
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
from surprise import SVD, Dataset, Reader

os.makedirs('reports', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Set styling
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')

# %% [markdown]
"""
## 1. Load Precomputed Models & Data
"""

# %%
# Load drug reviews and preprocessed datasets
reviews_df = pd.read_csv('data/raw/drug_reviews.tsv', sep='\t')
unified_df = pd.read_csv('data/processed/unified_healthcare_data.csv')

# Load Collaborative SVD model (or fallback if not trained yet)
try:
    svd_model = joblib.load('models/cf_best_model.joblib')
    print("Loaded pre-trained CF SVD model.")
except:
    # If not run yet, train a quick SVD model inline
    print("SVD model not found. Training inline SVD...")
    reviews_df['rating_normalized'] = reviews_df['rating'] / 2.0
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(reviews_df[['uniqueID', 'drugName', 'rating_normalized']], reader)
    svd_model = SVD(n_factors=30, n_epochs=15, random_state=42)
    trainset = data.build_full_trainset()
    svd_model.fit(trainset)
    joblib.dump(svd_model, 'models/cf_best_model.joblib')

# Load content-based cosine similarity matrix (or train inline)
try:
    cosine_sim = joblib.load('models/cb_cosine_similarity.joblib')
    print("Loaded pre-trained Content similarity matrix.")
except:
    print("Content similarity matrix not found. Training inline TF-IDF similarity...")
    unified_df['combined_features'] = unified_df['Description'].fillna('') + " " + unified_df['Medication'].fillna('') + " " + unified_df['Diet'].fillna('')
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(unified_df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    joblib.dump(cosine_sim, 'models/cb_cosine_similarity.joblib')

# Build mapping helpers
disease_to_idx = pd.Series(unified_df.index, index=unified_df['Disease']).to_dict()
idx_to_disease = unified_df['Disease'].to_dict()

# %% [markdown]
"""
## 2. Weighted Hybrid Recommendation Logic
"""

# %%
def get_hybrid_recommendation(user_id, condition, drug_name, alpha=0.5):
    """
    Blends:
    - Collab score: SVD rating prediction (normalized to [0, 1])
    - Content score: Similarity of user's condition to the drug's condition
    """
    # 1. Collab score (SVD)
    # SVD rating is on scale [0.5, 5.0]. Normalize to [0, 1].
    try:
        collab_pred = svd_model.predict(user_id, drug_name).est
        collab_score = (collab_pred - 0.5) / 4.5
    except:
        collab_score = 0.5  # default/neutral prediction
        
    # 2. Content score
    # We find if there is a matching disease in our unified_df for the drug's indicated condition
    # If the drug's condition matches the user's condition, content similarity is 1.0.
    # Otherwise, it is the similarity of the user's condition to the drug's indicated condition.
    content_score = 0.0
    user_cond = condition.strip().title()
    
    # Find drug's primary conditions in our database
    drug_reviews = reviews_df[reviews_df['drugName'] == drug_name]
    if not drug_reviews.empty:
        drug_conditions = drug_reviews['condition'].unique()
        similarities = []
        for d_cond in drug_conditions:
            d_cond_title = str(d_cond).strip().title()
            if user_cond == d_cond_title:
                similarities.append(1.0)
            elif user_cond in disease_to_idx and d_cond_title in disease_to_idx:
                u_idx = disease_to_idx[user_cond]
                d_idx = disease_to_idx[d_cond_title]
                similarities.append(cosine_sim[u_idx, d_idx])
            else:
                similarities.append(0.0)
        content_score = max(similarities) if similarities else 0.0
        
    # 3. Hybrid score
    hybrid_score = alpha * content_score + (1.0 - alpha) * collab_score
    return hybrid_score

# %% [markdown]
"""
## 3. Switching Hybrid Logic
"""

# %%
# Switching hybrid:
# If user has >= 3 interactions in the reviews database, use Collaborative SVD.
# Otherwise, fall back to Content-Based disease recommendations.
def recommend_switching_hybrid(user_id, condition, num_recommendations=5):
    user_history = reviews_df[reviews_df['uniqueID'] == user_id]
    
    if len(user_history) >= 3:
        print(f"🔄 User {user_id} has sufficient history ({len(user_history)} ratings). Using Collaborative SVD.")
        # Collaborative top recommendations: predict score for all unique drugs for this user's condition
        condition_drugs = reviews_df[reviews_df['condition'] == condition]['drugName'].unique()
        preds = []
        for drug in condition_drugs:
            pred = svd_model.predict(user_id, drug).est
            preds.append((drug, pred * 2.0))  # Scale back to 1-10
        preds.sort(key=lambda x: x[1], reverse=True)
        return pd.DataFrame(preds[:num_recommendations], columns=['Medication/Drug', 'Predicted Rating (1-10)'])
    else:
        print(f"🔄 User {user_id} is a new user/has cold start. Falling back to Content-Based similarity.")
        # Content based: recommend drugs for the closest matched disease
        user_cond_title = condition.strip().title()
        if user_cond_title in disease_to_idx:
            u_idx = disease_to_idx[user_cond_title]
            sim_scores = list(enumerate(cosine_sim[u_idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
            recoms = []
            for i, score in sim_scores:
                recoms.append({
                    'Medication/Drug': unified_df.iloc[i]['Medication'],
                    'Similarity Score': round(score, 4),
                    'Disease': idx_to_disease[i]
                })
            return pd.DataFrame(recoms)
        else:
            # Simple popular drugs fallback for condition
            cond_drugs = reviews_df[reviews_df['condition'] == condition]['drugName'].value_counts().head(num_recommendations).index.tolist()
            return pd.DataFrame(cond_drugs, columns=['Medication/Drug'])

# %% [markdown]
"""
## 4. Evaluate alpha values and plot RMSE curve
"""

# %%
# Grid search on alpha to demonstrate optimization
alphas = np.linspace(0, 1, 11)
sample_ratings = reviews_df.sample(200, random_state=42)

rmse_values = []
for alpha in alphas:
    squared_errors = []
    for _, row in sample_ratings.iterrows():
        uid = row['uniqueID']
        cond = row['condition']
        drug = row['drugName']
        true_rating_norm = row['rating'] / 10.0 # Scale 1-10 to 0-1
        
        pred_hybrid = get_hybrid_recommendation(uid, cond, drug, alpha=alpha)
        # Note: both scores are in [0,1], so error is in [0,1]
        error = true_rating_norm - pred_hybrid
        squared_errors.append(error ** 2)
        
    rmse = np.sqrt(np.mean(squared_errors))
    rmse_values.append(rmse)

# Plot RMSE vs Alpha
plt.figure(figsize=(10, 6))
plt.plot(alphas, rmse_values, 'o-', linewidth=2.5, color='darkorange')
plt.title('Hybrid Recommender: Grid Search on Alpha (α)', fontsize=14, fontweight='bold')
plt.xlabel('Alpha Weight (α) — Content-Based Weight')
plt.ylabel('RMSE (on 0-1 rating scale)')
plt.xticks(np.arange(0, 1.1, 0.1))
plt.grid(True, alpha=0.3)
plt.savefig('reports/hybrid_rmse_vs_alpha.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved hybrid grid search plot to reports/hybrid_rmse_vs_alpha.png")

# %% [markdown]
"""
## 5. Generate and Compare Recommendations
"""

# %%
# Compare recommenders side-by-side
print("\n" + "="*80)
print("TESTING SWITCHING HYBRID FOR DIFFERENT CASES")
print("="*80)

# Case A: Cold-start/New user (not in database)
print("\nCase A: Cold Start User (ID: 9999, Condition: Gerd)")
res_cold = recommend_switching_hybrid(9999, 'Gerd', 3)
print(res_cold.to_string(index=False))

# Case B: Regular user (let's pick one with rating history)
user_counts = reviews_df['uniqueID'].value_counts()
# Mock history since each uniqueID has 1 review in standard dataset
# We append multiple rows for a mock user to trigger CF branch
mock_user_id = 11111
mock_history = pd.DataFrame([
    {'uniqueID': mock_user_id, 'drugName': 'Prilosec', 'condition': 'Gerd', 'rating': 9, 'date': '2023-01-01', 'usefulCount': 10},
    {'uniqueID': mock_user_id, 'drugName': 'Nexium', 'condition': 'Gerd', 'rating': 8, 'date': '2023-02-01', 'usefulCount': 5},
    {'uniqueID': mock_user_id, 'drugName': 'Zantac', 'condition': 'Gerd', 'rating': 10, 'date': '2023-03-01', 'usefulCount': 15}
])
reviews_df = pd.concat([reviews_df, mock_history], ignore_index=True)

print(f"\nCase B: Returning User (ID: {mock_user_id}, Condition: Gerd)")
res_warm = recommend_switching_hybrid(mock_user_id, 'Gerd', 3)
print(res_warm.to_string(index=False))

print("\n✅ Hybrid filtering phase completed successfully!")
