# %% [markdown]
"""
# 🕸️ Notebook 10: Graph-Based Recommendation Engine
## Healthcare Recommendation System — Phase 5

This notebook builds a Knowledge Graph representing relationships between:
- Diseases (41)
- Symptoms (132)
- Medications
- Diets
- Workouts
- Precautions

We analyze graph properties, calculate PageRank importances, run modularity community detection,
and build a graph-based path recommendation system.
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
import ast
import pickle

os.makedirs('reports', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Set styling
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')

# %% [markdown]
"""
## 1. Build Knowledge Graph
"""

# %%
# Load data
training = pd.read_csv('data/raw/Training.csv')
training.rename(columns={'prognosis': 'Disease'}, inplace=True)
training['Disease'] = training['Disease'].astype(str).str.strip().str.title()

unified = pd.read_csv('data/processed/unified_healthcare_data.csv')
workout = pd.read_csv('data/raw/workout_df.csv')
workout.rename(columns={'disease': 'Disease'}, inplace=True)
workout['Disease'] = workout['Disease'].astype(str).str.strip().str.title()

# Instantiate Heterogeneous Graph
G = nx.Graph()

# Add Disease Nodes & Symptom Nodes + Edges from Training data (non-zero)
symptom_cols = [c for c in training.columns if c != 'Disease']

# Map unique diseases and symptoms
diseases = training['Disease'].unique()
for d in diseases:
    G.add_node(d, type='Disease')
for s in symptom_cols:
    G.add_node(s.replace('_', ' ').title(), type='Symptom')

# Add edges based on training data associations (non-zero means present)
# To avoid duplicate edges, we only add if it's a valid link
for _, row in training.iterrows():
    d = row['Disease']
    for col in symptom_cols:
        if row[col] == 1:
            s_name = col.replace('_', ' ').title()
            G.add_edge(d, s_name, relation='has_symptom')

# %%
# Add Medications, Diets, and Precautions nodes and edges
for _, row in unified.iterrows():
    d = row['Disease']
    
    # Medications
    try:
        med_list = ast.literal_eval(row['Medication'])
        for m in med_list:
            m_name = m.strip().title()
            G.add_node(m_name, type='Medication')
            G.add_edge(d, m_name, relation='treated_by_medication')
    except:
        pass
        
    # Diets
    try:
        diet_list = ast.literal_eval(row['Diet'])
        for dt in diet_list:
            dt_name = dt.strip().title()
            G.add_node(dt_name, type='Diet')
            G.add_edge(d, dt_name, relation='dietary_recommendation')
    except:
        pass

    # Precautions
    for i in range(1, 5):
        prec_col = f'Precaution_{i}'
        if prec_col in row and pd.notnull(row[prec_col]):
            p_name = str(row[prec_col]).strip().title()
            G.add_node(p_name, type='Precaution')
            G.add_edge(d, p_name, relation='precautionary_measure')

# Add Workouts
for _, row in workout.iterrows():
    d = row['Disease']
    w_name = str(row['workout']).strip().title()
    G.add_node(w_name, type='Workout')
    G.add_edge(d, w_name, relation='recommended_exercise')

# %% [markdown]
"""
## 2. Graph Statistics & Properties
"""

# %%
n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges()
density = nx.density(G)
avg_degree = np.mean([val for (node, val) in G.degree()])

print("\n📊 KNOWLEDGE GRAPH STATISTICS")
print("="*60)
print(f"Number of Nodes: {n_nodes}")
print(f"Number of Edges: {n_edges}")
print(f"Graph Density  : {density:.6f}")
print(f"Average Degree : {avg_degree:.2f}")

# Node counts by type
node_types = nx.get_node_attributes(G, 'type')
type_counts = pd.Series(node_types.values()).value_counts()
print("\nNode Counts by Type:")
print(type_counts.to_string())

# Plot Degree Distribution
degrees = [d for n, d in G.degree()]
plt.figure(figsize=(10, 5))
sns.histplot(degrees, bins=30, kde=True, color='teal')
plt.title('Node Degree Distribution in Knowledge Graph', fontsize=14, fontweight='bold')
plt.xlabel('Degree')
plt.ylabel('Node Count')
plt.savefig('reports/graph_degree_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# %% [markdown]
"""
## 3. Subgraph Visualization
"""

# %%
# Select 3 sample diseases and visualize their local networks
selected_diseases = ['Fungal Infection', 'Allergy', 'Gerd']
sub_nodes = list(selected_diseases)

for d in selected_diseases:
    sub_nodes.extend(list(G.neighbors(d)))

subgraph = G.subgraph(sub_nodes)

# Visual styling for node types
color_map = {
    'Disease': 'tomato',
    'Symptom': 'gold',
    'Medication': 'lightskyblue',
    'Diet': 'lightgreen',
    'Precaution': 'violet',
    'Workout': 'orange'
}

node_colors = []
for node in subgraph.nodes():
    ntype = G.nodes[node].get('type', 'Symptom')
    node_colors.append(color_map.get(ntype, 'gray'))

plt.figure(figsize=(14, 12))
pos = nx.spring_layout(subgraph, k=0.35, seed=42)
nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=800, edgecolors='black')
nx.draw_networkx_edges(subgraph, pos, alpha=0.3, width=1.5)
nx.draw_networkx_labels(subgraph, pos, font_size=9, font_weight='bold')

# Create legend
markers = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=col, markersize=10) for col in color_map.values()]
plt.legend(markers, color_map.keys(), title="Node Types", loc='upper left')

plt.title('Knowledge Graph Subgraph for 3 Selected Diseases', fontsize=16, fontweight='bold')
plt.axis('off')
plt.savefig('reports/graph_disease_subgraph.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved subgraph visualization to reports/graph_disease_subgraph.png")

# %% [markdown]
"""
## 4. Graph-Based Recommendations & PageRank
"""

# %%
# A. PageRank analysis
pagerank = nx.pagerank(G)
top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:15]

print("\n📊 TOP 15 NODES BY PAGERANK IMPORTANCE")
print("="*60)
for node, score in top_nodes:
    ntype = G.nodes[node].get('type', 'Unknown')
    print(f" - {node:<35} | Type: {ntype:<12} | PageRank: {score:.5f}")

# %%
# B. Modularity-Based Community Detection
from networkx.algorithms import community
communities = list(community.greedy_modularity_communities(G))
print(f"\nDetected {len(communities)} communities inside the knowledge graph.")

# C. Path-Based Recommendation for Symptoms
# Given input symptoms, find which diseases share the most paths / common symptom nodes
def recommend_diseases_graph(input_symptoms, num_recommendations=3):
    standardized = [s.replace('_', ' ').strip().title() for s in input_symptoms]
    valid_symptoms = [s for s in standardized if s in G and G.nodes[s]['type'] == 'Symptom']
    
    if not valid_symptoms:
        return "None of the input symptoms exist in the knowledge graph."
        
    print(f"Querying graph for symptoms: {valid_symptoms}")
    
    # Track candidate diseases and match counts
    candidate_diseases = {}
    for sym in valid_symptoms:
        # Neighbors of a symptom node are the diseases connected to it
        for neighbor in G.neighbors(sym):
            if G.nodes[neighbor]['type'] == 'Disease':
                candidate_diseases[neighbor] = candidate_diseases.get(neighbor, 0) + 1
                
    # Sort diseases by matched symptom counts
    sorted_candidates = sorted(candidate_diseases.items(), key=lambda x: x[1], reverse=True)
    
    recoms = []
    for d, score in sorted_candidates[:num_recommendations]:
        # Collect medications, diets, workouts connected to this disease
        meds = [n for n in G.neighbors(d) if G.nodes[n]['type'] == 'Medication']
        diets = [n for n in G.neighbors(d) if G.nodes[n]['type'] == 'Diet']
        
        recoms.append({
            'Disease': d,
            'Matched Symptoms Count': score,
            'Medications': meds[:3],
            'Diets': diets[:3]
        })
        
    return pd.DataFrame(recoms)

print("\n" + "="*80)
print("TESTING GRAPH-BASED RECOMMENDATIONS")
print("="*80)
input_syms = ['Itching', 'Skin Rash', 'Nodal Skin Eruptions']
res_graph = recommend_diseases_graph(input_syms, 3)
print(res_graph.to_string())

# Save the graph model to models folder
with open('models/knowledge_graph.pkl', 'wb') as f:
    pickle.dump(G, f)
print("\nSaved knowledge graph pickle to models/knowledge_graph.pkl")

print("\n✅ Graph-based recommendation phase completed successfully!")
