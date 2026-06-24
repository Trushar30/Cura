# src/recommenders/content_based.py
"""
Content-Based Filtering Recommendation module.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.disease_to_idx = {}
        self.idx_to_disease = {}
        self.unified_df = None

    def fit(self, unified_df):
        """Fits TF-IDF and computes cosine similarity of merged features."""
        self.unified_df = unified_df.copy()
        
        # Concat Description, Medications, Diets for feature vector
        self.unified_df['combined_text'] = (
            self.unified_df['Description'].fillna('') + ' ' +
            self.unified_df['Medication'].fillna('') + ' ' +
            self.unified_df['Diet'].fillna('')
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(self.unified_df['combined_text'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        self.disease_to_idx = pd.Series(self.unified_df.index, index=self.unified_df['Disease']).to_dict()
        self.idx_to_disease = self.unified_df['Disease'].to_dict()

    def recommend_similar_diseases(self, disease_name, num_recommendations=5):
        """Returns similar diseases based on description/treatment TF-IDF features."""
        name = disease_name.strip().title()
        if name not in self.disease_to_idx:
            return f"Disease '{name}' not found in training corpus."
            
        idx = self.disease_to_idx[name]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
        
        recoms = []
        for i, score in sim_scores:
            recoms.append({
                'Disease': self.idx_to_disease[i],
                'Similarity': round(score, 4),
                'Description': self.unified_df.iloc[i]['Description'],
                'Medications': self.unified_df.iloc[i]['Medication'],
                'Diet': self.unified_df.iloc[i]['Diet']
            })
        return pd.DataFrame(recoms)

    def recommend_for_symptoms(self, training_df, symptom_list, symptom_cols, num_recommendations=3):
        """Maps input symptoms to the closest matching disease, then recommends similar diseases."""
        standardized_inputs = [s.replace('_', ' ').strip().lower() for s in symptom_list]
        
        matched_cols = []
        for s in standardized_inputs:
            for col in symptom_cols:
                if col.replace('_', ' ').strip().lower() == s:
                    matched_cols.append(col)
                    break
                    
        if not matched_cols:
            return "No matching symptoms found."
            
        # Create input binary symptom vector
        input_vector = np.zeros(len(symptom_cols))
        for col in matched_cols:
            input_vector[symptom_cols.index(col)] = 1.0
            
        # Match against Training.csv binary rows
        X_train = training_df[symptom_cols].values
        sims = cosine_similarity(input_vector.reshape(1, -1), X_train).flatten()
        
        best_match_idx = np.argmax(sims)
        matched_disease = training_df.iloc[best_match_idx]['Disease']
        
        print(f"Closest matched disease: '{matched_disease}' (Score: {sims[best_match_idx]:.4f})")
        return self.recommend_similar_diseases(matched_disease, num_recommendations)
