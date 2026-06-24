# src/recommenders/graph_based.py
"""
Graph-Based Recommendation module using NetworkX.
"""

import networkx as nx
import pandas as pd
import ast

class GraphBasedRecommender:
    def __init__(self):
        self.G = nx.Graph()

    def build_graph(self, training_df, unified_df, workout_df):
        """Constructs heterogeneous knowledge graph."""
        # Nodes
        symptom_cols = [c for c in training_df.columns if c != 'Disease']
        
        for d in training_df['Disease'].unique():
            self.G.add_node(d, type='Disease')
        for s in symptom_cols:
            self.G.add_node(s.replace('_', ' ').title(), type='Symptom')
            
        # Add Disease-Symptom edges
        for _, row in training_df.iterrows():
            d = row['Disease']
            for col in symptom_cols:
                if row[col] == 1:
                    s_name = col.replace('_', ' ').title()
                    self.G.add_edge(d, s_name, relation='has_symptom')
                    
        # Add Medications, Diets, and Precautions
        for _, row in unified_df.iterrows():
            d = row['Disease']
            
            # Medications
            try:
                med_list = ast.literal_eval(row['Medication'])
                for m in med_list:
                    m_name = m.strip().title()
                    self.G.add_node(m_name, type='Medication')
                    self.G.add_edge(d, m_name, relation='treated_by_medication')
            except:
                pass
                
            # Diets
            try:
                diet_list = ast.literal_eval(row['Diet'])
                for dt in diet_list:
                    dt_name = dt.strip().title()
                    self.G.add_node(dt_name, type='Diet')
                    self.G.add_edge(d, dt_name, relation='dietary_recommendation')
            except:
                pass

            # Precautions
            for i in range(1, 5):
                prec_col = f'Precaution_{i}'
                if prec_col in row and pd.notnull(row[prec_col]):
                    p_name = str(row[prec_col]).strip().title()
                    self.G.add_node(p_name, type='Precaution')
                    self.G.add_edge(d, p_name, relation='precautionary_measure')

        # Add Workouts
        for _, row in workout_df.iterrows():
            d = row['Disease']
            w_name = str(row['workout']).strip().title()
            self.G.add_node(w_name, type='Workout')
            self.G.add_edge(d, w_name, relation='recommended_exercise')

    def recommend_for_symptoms(self, input_symptoms, num_recommendations=3):
        """Finds candidate diseases that match input symptoms based on path overlaps."""
        standardized = [s.replace('_', ' ').strip().title() for s in input_symptoms]
        valid_symptoms = [s for s in standardized if s in self.G and self.G.nodes[s]['type'] == 'Symptom']
        
        if not valid_symptoms:
            return "No matching symptoms exist in the knowledge graph."
            
        candidate_diseases = {}
        for sym in valid_symptoms:
            for neighbor in self.G.neighbors(sym):
                if self.G.nodes[neighbor]['type'] == 'Disease':
                    candidate_diseases[neighbor] = candidate_diseases.get(neighbor, 0) + 1
                    
        # Sort diseases by matched symptom counts
        sorted_candidates = sorted(candidate_diseases.items(), key=lambda x: x[1], reverse=True)
        
        recoms = []
        for d, score in sorted_candidates[:num_recommendations]:
            meds = [n for n in self.G.neighbors(d) if self.G.nodes[n]['type'] == 'Medication']
            diets = [n for n in self.G.neighbors(d) if self.G.nodes[n]['type'] == 'Diet']
            
            recoms.append({
                'Disease': d,
                'MatchedSymptoms': score,
                'Medications': meds[:3],
                'Diets': diets[:3]
            })
            
        return pd.DataFrame(recoms)
