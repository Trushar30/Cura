# src/recommenders/deep_learning.py
"""
Deep Learning Recommendation module (NCF Model).
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder

class DeepLearningRecommender:
    def __init__(self, latent_dim=32):
        self.latent_dim = latent_dim
        self.cond_encoder = LabelEncoder()
        self.drug_encoder = drug_encoder = LabelEncoder()
        self.model = None

    def build_model(self, n_conditions, n_drugs):
        """Constructs NCF architecture."""
        cond_input = Input(shape=(1,), name='cond_input')
        drug_input = Input(shape=(1,), name='drug_input')
        
        cond_emb = Flatten()(Embedding(n_conditions, self.latent_dim, name='cond_emb')(cond_input))
        drug_emb = Flatten()(Embedding(n_drugs, self.latent_dim, name='drug_emb')(drug_input))
        
        concat = Concatenate()([cond_emb, drug_emb])
        x = Dense(128, activation='relu')(concat)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(1, activation='sigmoid')(x)
        
        self.model = Model(inputs=[cond_input, drug_input], outputs=output)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return self.model

    def fit(self, reviews_df, epochs=15, batch_size=64, validation_split=0.2):
        """Preprocesses data, builds and trains model."""
        reviews = reviews_df.copy()
        
        reviews['cond_idx'] = self.cond_encoder.fit_transform(reviews['condition'])
        reviews['drug_idx'] = self.drug_encoder.fit_transform(reviews['drugName'])
        
        n_conditions = reviews['cond_idx'].nunique()
        n_drugs = reviews['drug_idx'].nunique()
        
        self.build_model(n_conditions, n_drugs)
        
        # Target rating normalized to [0, 1]
        y = reviews['rating'].values / 10.0
        X_cond = reviews['cond_idx'].values
        X_drug = reviews['drug_idx'].values
        
        history = self.model.fit(
            [X_cond, X_drug],
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0
        )
        return history

    def predict_rating(self, condition, drug_name):
        """Predicts score (1-10 scale) for a condition and drug name."""
        try:
            cond_idx = self.cond_encoder.transform([condition])[0]
            drug_idx = self.drug_encoder.transform([drug_name])[0]
            
            pred = self.model.predict([np.array([cond_idx]), np.array([drug_idx])], verbose=0)[0][0]
            return pred * 10.0
        except Exception as e:
            # Fallback score
            return 5.0
            
    def save(self, filepath):
        self.model.save(filepath)
