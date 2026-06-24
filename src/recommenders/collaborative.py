# src/recommenders/collaborative.py
"""
Collaborative Filtering Recommendation module.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from surprise import Dataset, Reader, SVD

class CollaborativeRecommender:
    def __init__(self, n_factors=50, n_epochs=20, random_state=42):
        self.model = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=random_state)
        self.reader = Reader(rating_scale=(0.5, 5))
        self.trainset = None

    def fit(self, reviews_df, user_col='uniqueID', item_col='drugName', rating_col='rating'):
        """Fits SVD on normalized ratings (1-10 rating scaled to 1-5)."""
        reviews = reviews_df.copy()
        reviews['rating_norm'] = reviews[rating_col] / 2.0
        
        data = Dataset.load_from_df(reviews[[user_col, item_col, 'rating_norm']], self.reader)
        self.trainset = data.build_full_trainset()
        self.model.fit(self.trainset)

    def predict_rating(self, user_id, drug_name):
        """Predicts rating for a specific user and drug (returns in 1-10 scale)."""
        pred = self.model.predict(user_id, drug_name).est
        return pred * 2.0  # Scale back to 1-10 range

    def get_top_n_recommendations(self, reviews_df, user_id, condition, n=5):
        """Generates top-N drug recommendations for a user under a specific condition."""
        # Find all unique drugs that treat this condition in the reviews dataset
        condition_drugs = reviews_df[reviews_df['condition'] == condition]['drugName'].unique()
        
        predictions = []
        for drug in condition_drugs:
            pred_score = self.predict_rating(user_id, drug)
            predictions.append((drug, pred_score))
            
        # Sort in descending order
        predictions.sort(key=lambda x: x[1], reverse=True)
        return pd.DataFrame(predictions[:n], columns=['DrugName', 'PredictedRating'])
