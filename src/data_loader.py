# src/data_loader.py
"""
Data loading and standardization utility module.
"""

import os
import pandas as pd

class DataLoader:
    def __init__(self, raw_data_dir='data/raw'):
        self.raw_data_dir = raw_data_dir

    def load_dataset(self, filename, sep=','):
        """Loads a specific dataset from raw data folder and standardizes columns."""
        filepath = os.path.join(self.raw_data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filename} not found in {self.raw_data_dir}")

        name = filename.split('.')[0]
        if name == 'symptom_Description':
            df = pd.read_csv(filepath, header=None, names=['Disease', 'Description'], sep=sep)
        elif name == 'symptom_precaution':
            df = pd.read_csv(filepath, header=None, names=['Disease', 'Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4'], sep=sep)
        else:
            df = pd.read_csv(filepath, sep=sep)

        # Drop index/unnamed columns
        unnamed = [c for c in df.columns if 'Unnamed' in c]
        if unnamed:
            df.drop(columns=unnamed, inplace=True)

        # Standardize prognosis / disease column names
        if 'prognosis' in df.columns:
            df.rename(columns={'prognosis': 'Disease'}, inplace=True)
        if 'disease' in df.columns:
            df.rename(columns={'disease': 'Disease'}, inplace=True)

        # Title-case and strip string representations of Disease
        if 'Disease' in df.columns:
            df['Disease'] = df['Disease'].astype(str).str.strip().str.title()

        return df

    def load_all_raw_data(self):
        """Loads and returns all 12 raw dataframes as a dictionary."""
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
        
        datasets = {}
        for filename, sep in file_list:
            name = filename.split('.')[0]
            datasets[name] = self.load_dataset(filename, sep=sep)
            
        return datasets
