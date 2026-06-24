# src/recommenders/hybrid.py
"""
Hybrid Filtering Recommendation module.
"""

import pandas as pd
import numpy as np

class HybridRecommender:
    def __init__(self, cb_model, cf_model, reviews_df):
        self.cb_model = cb_model
        self.cf_model = cf_model
        self.reviews_df = reviews_df

    def get_hybrid_score(self, user_id, condition, drug_name, alpha=0.5):
        """Computes hybrid score blending TF-IDF disease similarity and SVD predictions."""
        # 1. Collab score
        try:
            collab_pred = self.cf_model.predict_rating(user_id, drug_name)
            # Normalize [1, 10] to [0, 1]
            collab_score = (collab_pred - 1.0) / 9.0
        except:
            collab_score = 0.5
            
        # 2. Content score
        content_score = 0.0
        user_cond = condition.strip().title()
        
        # Get conditions treated by drug in historical records
        drug_conds = self.reviews_df[self.reviews_df['drugName'] == drug_name]['condition'].unique()
        similarities = []
        for d_cond in drug_conds:
            d_cond_title = str(d_cond).strip().title()
            if user_cond == d_cond_title:
                similarities.append(1.0)
            elif user_cond in self.cb_model.disease_to_idx and d_cond_title in self.cb_model.disease_to_idx:
                u_idx = self.cb_model.disease_to_idx[user_cond]
                d_idx = self.cb_model.disease_to_idx[d_cond_title]
                similarities.append(self.cb_model.cosine_sim[u_idx, d_idx])
            else:
                similarities.append(0.0)
        content_score = max(similarities) if similarities else 0.0
        
        # 3. Blend
        return alpha * content_score + (1.0 - alpha) * collab_score

    def recommend(self, user_id, condition, num_recommendations=5, alpha=0.5):
        """Generates hybrid top-N drug recommendations for a user."""
        # Retrieve all unique drugs that treat this condition in historical reviews
        condition_drugs = self.reviews_df[self.reviews_df['condition'] == condition]['drugName'].unique()
        
        scores = []
        for drug in condition_drugs:
            score = self.get_hybrid_score(user_id, condition, drug, alpha=alpha)
            scores.append((drug, score))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        return pd.DataFrame(scores[:num_recommendations], columns=['DrugName', 'HybridScore'])
