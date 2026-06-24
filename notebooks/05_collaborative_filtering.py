# %% [markdown]
"""
# 🤝 Notebook 05: Collaborative Filtering Recommendation Engine
## Healthcare Recommendation System — Phase 4

This notebook implements Collaborative Filtering models using the `scikit-surprise` library:
1. Normalizes user ratings (1-10 scale) to a standard 1-5 scale.
2. Compares SVD, SVDpp, KNN (User-based), KNN (Item-based), and NMF using cross-validation.
3. Selects and trains the best performing model.
4. Implements top-N recommendation helpers and saves the model.
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from surprise import Dataset, Reader, SVD, SVDpp, KNNBasic, NMF
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy

os.makedirs('reports', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Set styling
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')

# %% [markdown]
"""
## 1. Load and Format Dataset
"""

# %%
# Load drug reviews
reviews_df = pd.read_csv('data/raw/drug_reviews.tsv', sep='\t')
print("Reviews Dataset shape:", reviews_df.shape)

# Normalize rating: 1-10 scale to 1-5 scale (dividing by 2)
reviews_df['rating_normalized'] = reviews_df['rating'] / 2.0

# Prepare Reader and Dataset for Surprise
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(
    reviews_df[['uniqueID', 'drugName', 'rating_normalized']], 
    reader
)

# %% [markdown]
"""
## 2. Compare Recommendation Algorithms
"""

# %%
# Define models with fast parameters to speed up cross validation
models = {
    'SVD': SVD(n_factors=20, n_epochs=10, random_state=42),
    'SVDpp': SVDpp(n_factors=10, n_epochs=5, random_state=42),
    'KNN (User)': KNNBasic(sim_options={'name': 'cosine', 'user_based': True}, verbose=False),
    'KNN (Item)': KNNBasic(sim_options={'name': 'cosine', 'user_based': False}, verbose=False),
    'NMF': NMF(n_factors=10, n_epochs=15, random_state=42)
}

results = []

for name, model in models.items():
    print(f"Running 5-fold cross-validation for {name}...")
    cv_results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
    
    rmse_mean = cv_results['test_rmse'].mean()
    mae_mean = cv_results['test_mae'].mean()
    
    results.append({
        'Algorithm': name,
        'RMSE': rmse_mean,
        'MAE': mae_mean
    })
    print(f"   {name} Mean RMSE: {rmse_mean:.4f}, Mean MAE: {mae_mean:.4f}")

# Convert to DataFrame
results_df = pd.DataFrame(results)

# %%
# Bar chart comparison
plt.figure(figsize=(10, 6))
sns.barplot(x='Algorithm', y='RMSE', data=results_df, palette='viridis')
plt.title('Algorithm Comparison: Mean RMSE Score (Lower is Better)', fontsize=14, fontweight='bold')
plt.ylabel('RMSE')
plt.ylim(0, 2.5)

# Annotate values
for idx, row in results_df.iterrows():
    plt.text(idx, row['RMSE'] + 0.05, f"{row['RMSE']:.4f}", ha='center', fontweight='bold')

plt.savefig('reports/cf_algorithm_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved comparison chart to reports/cf_algorithm_comparison.png")

# %% [markdown]
"""
## 3. Train Best Model
"""

# %%
# Select the best algorithm based on lowest RMSE
best_algo_name = results_df.loc[results_df['RMSE'].idxmin()]['Algorithm']
print(f"\nBest algorithm selected: {best_algo_name}")

# Retrain best model on full dataset
best_model = SVD(n_factors=50, n_epochs=20, random_state=42) if 'SVD' in best_algo_name else models[best_algo_name]
trainset = data.build_full_trainset()
best_model.fit(trainset)

# Save the trained model
joblib.dump(best_model, 'models/cf_best_model.joblib')
print("Saved best Collaborative Filtering model to models/cf_best_model.joblib")

# %% [markdown]
"""
## 4. Generate Recommendations for Sample Users
"""

# %%
# Helper function to get top-N recommendations for each user
def get_top_n(predictions, n=5):
    from collections import defaultdict
    top_n = defaultdict(list)
    
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
        
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
        
    return top_n

# Get test predictions on a sample testset
_, testset = train_test_split(data, test_size=0.2, random_state=42)
predictions = best_model.test(testset)

# Get top 5 recommendations for each user
top_n_recs = get_top_n(predictions, n=5)

# Print recommendations for 5 sample users
sample_uids = list(top_n_recs.keys())[:5]
print("\n" + "="*60)
print("TOP-5 RECOMMENDATIONS FOR SAMPLE USERS")
print("="*60)

for uid in sample_uids:
    print(f"\nUser ID: {uid}")
    user_condition = reviews_df[reviews_df['uniqueID'] == uid]['condition'].values[0]
    print(f"Condition: {user_condition}")
    print("Recommended Drugs:")
    for drugName, score in top_n_recs[uid]:
        # Scale back score to 1-10 for readable output
        original_scale_score = round(score * 2.0, 1)
        print(f"  - {drugName} (predicted rating: {original_scale_score}/10)")

print("\nFinished Notebook 05!")
