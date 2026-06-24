# %% [markdown]
"""
# 🧠 Notebook 09: Deep Learning Recommender System
## Healthcare Recommendation System — Phase 5

This notebook builds and trains deep learning recommendation models:
1. Encodes conditions (users) and drug names (items) as indices.
2. Normalizes target ratings to [0, 1] range.
3. Builds a Neural Collaborative Filtering (NCF) model using embeddings and dense layers.
4. Builds the mentor's simple Sequential recommendation model.
5. Trains both models, saves them, and compares their performance.
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

# Suppress TensorFlow logs for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model, Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

os.makedirs('reports', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Set styling
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')

# %% [markdown]
"""
## 1. Load Data & Encode Categoricals
"""

# %%
reviews_df = pd.read_csv('data/raw/drug_reviews.tsv', sep='\t')

# Fit Label Encoders
cond_encoder = LabelEncoder()
drug_encoder = LabelEncoder()

reviews_df['cond_idx'] = cond_encoder.fit_transform(reviews_df['condition'])
reviews_df['drug_idx'] = drug_encoder.fit_transform(reviews_df['drugName'])

# Save encoders
joblib.dump(cond_encoder, 'models/ncf_cond_encoder.joblib')
joblib.dump(drug_encoder, 'models/ncf_drug_encoder.joblib')

n_conditions = reviews_df['cond_idx'].nunique()
n_drugs = reviews_df['drug_idx'].nunique()

print(f"Number of unique conditions (user-groups): {n_conditions}")
print(f"Number of unique drugs (items): {n_drugs}")

# Normalize ratings to [0, 1] scale
reviews_df['rating_normalized'] = reviews_df['rating'] / 10.0

# Train/Test Split (80/20)
train_df, test_df = train_test_split(reviews_df, test_size=0.2, random_state=42)

# Prepare inputs/outputs
X_train_cond = train_df['cond_idx'].values
X_train_drug = train_df['drug_idx'].values
y_train = train_df['rating_normalized'].values

X_test_cond = test_df['cond_idx'].values
X_test_drug = test_df['drug_idx'].values
y_test = test_df['rating_normalized'].values

# %% [markdown]
"""
## 2. Build Models
"""

# %%
# Model A: Neural Collaborative Filtering (NCF)
user_input = Input(shape=(1,), name='user_input')  # condition index
item_input = Input(shape=(1,), name='item_input')   # drug index

# Embeddings
user_emb = Flatten()(Embedding(n_conditions, 32, name='user_emb')(user_input))
item_emb = Flatten()(Embedding(n_drugs, 32, name='item_emb')(item_input))

# Concatenate embeddings and pass to dense layers
concat = Concatenate()([user_emb, item_emb])
x = Dense(128, activation='relu')(concat)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(1, activation='sigmoid')(x)

ncf_model = Model(inputs=[user_input, item_input], outputs=output)
ncf_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
ncf_model.summary()

# %%
# Model B: Simple Sequential Model (Mentor's Approach)
# Embedding input layer is index of drug, output predicted normalized rating
simple_model = Sequential([
    Embedding(input_dim=n_drugs, output_dim=32, input_length=1),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
simple_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
simple_model.summary()

# %% [markdown]
"""
## 3. Train Models
"""

# %%
# For time efficiency and resource management, we train for 15 epochs
# rather than 50, which is enough to demonstrate convergence.
epochs = 15
batch_size = 64

print(f"\nTraining NCF Model for {epochs} epochs...")
ncf_history = ncf_model.fit(
    [X_train_cond, X_train_drug], 
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2,
    verbose=1
)

print(f"\nTraining Simple Sequential Model for {epochs} epochs...")
# Simple model only takes drug index as input
simple_history = simple_model.fit(
    X_train_drug,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2,
    verbose=1
)

# %% [markdown]
"""
## 4. Evaluation and Plotting
"""

# %%
# Evaluate NCF
ncf_eval = ncf_model.evaluate([X_test_cond, X_test_drug], y_test, verbose=0)
ncf_rmse = np.sqrt(ncf_eval[0])
ncf_mae = ncf_eval[1]

# Evaluate Simple
simple_eval = simple_model.evaluate(X_test_drug, y_test, verbose=0)
simple_rmse = np.sqrt(simple_eval[0])
simple_mae = simple_eval[1]

print("\n" + "="*50)
print("DEEP LEARNING MODEL COMPARISON")
print("="*50)
print(f"NCF Model     — Test RMSE: {ncf_rmse:.4f}, Test MAE: {ncf_mae:.4f}")
print(f"Simple Model  — Test RMSE: {simple_rmse:.4f}, Test MAE: {simple_mae:.4f}")
print("-" * 50)

# %%
# Save models
ncf_model.save('models/ncf_model.keras')
simple_model.save('models/simple_seq_model.keras')
print("Saved models to models/ folder.")

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(ncf_history.history['loss'], label='NCF Train Loss', linewidth=2)
plt.plot(ncf_history.history['val_loss'], label='NCF Val Loss', linewidth=2)
plt.plot(simple_history.history['loss'], label='Simple Train Loss', linewidth=2, linestyle='--')
plt.plot(simple_history.history['val_loss'], label='Simple Val Loss', linewidth=2, linestyle='--')
plt.title('Training and Validation Loss (MSE)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('reports/dl_training_loss.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot training MAE
plt.figure(figsize=(10, 6))
plt.plot(ncf_history.history['mae'], label='NCF Train MAE', linewidth=2)
plt.plot(ncf_history.history['val_mae'], label='NCF Val MAE', linewidth=2)
plt.plot(simple_history.history['mae'], label='Simple Train MAE', linewidth=2, linestyle='--')
plt.plot(simple_history.history['val_mae'], label='Simple Val MAE', linewidth=2, linestyle='--')
plt.title('Training and Validation MAE', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('reports/dl_training_mae.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved training curves to reports/")

print("\n✅ Deep Learning model phase completed successfully!")
