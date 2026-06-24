# src/preprocessing.py
"""
Data preprocessing, cleaning, and feature engineering pipeline module.
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

class Preprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def fit_symptom_weights(self, severity_df):
        """Builds symptom to severity weight mapping dictionary."""
        severity_df['Symptom'] = severity_df['Symptom'].astype(str).str.replace('_', ' ').str.strip().str.lower()
        return dict(zip(severity_df['Symptom'], severity_df['weight']))

    def engineer_symptom_features(self, df, symptom_cols, severity_dict):
        """Computes symptom_count and total_severity_score for the symptoms dataframe."""
        df = df.copy()
        
        # Calculate active symptom count per row
        df['symptom_count'] = df[symptom_cols].sum(axis=1)
        
        # Clean symptom column names for matching
        cleaned_cols = [c.replace('_', ' ').strip().lower() for c in symptom_cols]
        weight_map = {orig: severity_dict.get(clean, 0) for orig, clean in zip(symptom_cols, cleaned_cols)}
        
        # Compute total severity score
        def row_severity(row):
            return sum(row[col] * weight_map[col] for col in symptom_cols)
            
        df['total_severity_score'] = df.apply(row_severity, axis=1)
        
        # Assign risk level (Low / Medium / High)
        mean_score = df['total_severity_score'].mean()
        std_score = df['total_severity_score'].std()
        
        def get_risk(score):
            if score < (mean_score - 0.5 * std_score):
                return 'Low'
            elif score > (mean_score + 0.5 * std_score):
                return 'High'
            else:
                return 'Medium'
                
        df['disease_risk_level'] = df['total_severity_score'].apply(get_risk)
        return df

    def preprocess_review_text(self, text):
        """Applies NLP cleaning pipeline: lowercases, removes punctuation, tokenizes, removes stop words, lemmatizes."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = nltk.word_tokenize(text)
        cleaned = [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words]
        return " ".join(cleaned)

    def extract_sentiment(self, df, review_col='review'):
        """Applies TextBlob to extract sentiment polarity and subjectivity."""
        df = df.copy()
        df['polarity'] = df[review_col].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        df['subjectivity'] = df[review_col].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
        
        def label_polarity(p):
            if p > 0.1: return 'Positive'
            elif p < -0.1: return 'Negative'
            else: return 'Neutral'
            
        df['sentiment_label'] = df['polarity'].apply(label_polarity)
        return df
