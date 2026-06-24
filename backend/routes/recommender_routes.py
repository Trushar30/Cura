# backend/routes/recommender_routes.py
"""
Recommendation engine endpoints: Disease Prediction, Drug Recommendation models, and Bandit feedback.
"""

import os
import ast
import pickle
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.database import get_db, User, UserProfile, UserActivity
from backend.auth import get_current_user

router = APIRouter(prefix="/recommend", tags=["Recommendations"])

# ═══════════════════════════════════════════════════════════════════════
# MODELS CACHE & SETUP
# ═══════════════════════════════════════════════════════════════════════

# Global caches
svd_model = None
cosine_sim = None
cb_vectorizer = None
ncf_model = None
ncf_cond_encoder = None
ncf_drug_encoder = None
knowledge_graph = None

# Pre-load files for fast serving
unified_data = pd.read_csv('data/processed/unified_healthcare_data.csv')
training_data = pd.read_csv('data/processed/training_engineered.csv')
symptom_cols = [c for c in training_data.columns if c not in ['Disease', 'symptom_count', 'total_severity_score', 'disease_risk_level']]
reviews_data = pd.read_csv('data/raw/drug_reviews.tsv', sep='\t')
sentiment_stats = pd.read_csv('data/processed/drug_sentiment_statistics.csv')

# Set index of sentiment statistics to drugName
if 'drugName' in sentiment_stats.columns:
    sentiment_stats.set_index('drugName', inplace=True)

# In-memory Multi-Armed Bandit Priors for 20 drugs
bandit_arms = reviews_data['drugName'].value_counts().head(20).index.tolist()
bandit_alpha = {d: 1.0 for d in bandit_arms}
bandit_beta = {d: 1.0 for d in bandit_arms}

def load_models():
    global svd_model, cosine_sim, cb_vectorizer, knowledge_graph
    try:
        if os.path.exists('models/cf_best_model.joblib'):
            svd_model = joblib.load('models/cf_best_model.joblib')
            print("Loaded SVD model.")
        if os.path.exists('models/cb_cosine_similarity.joblib'):
            cosine_sim = joblib.load('models/cb_cosine_similarity.joblib')
            print("Loaded Content similarity matrix.")
        if os.path.exists('models/cb_vectorizer.joblib'):
            cb_vectorizer = joblib.load('models/cb_vectorizer.joblib')
        if os.path.exists('models/knowledge_graph.pkl'):
            with open('models/knowledge_graph.pkl', 'rb') as f:
                knowledge_graph = pickle.load(f)
            print("Loaded Knowledge Graph.")
    except Exception as e:
        print("Warning during model load:", e)

def get_ncf_model():
    global ncf_model, ncf_cond_encoder, ncf_drug_encoder
    if ncf_model is None:
        try:
            if os.path.exists('models/ncf_model.keras'):
                print("Loading NCF model lazily...")
                import tensorflow as tf
                # Set CPU thread limits to avoid memory pressure on Render
                os.environ['TF_NUM_INTEROP_THREADS'] = '1'
                os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
                tf.config.threading.set_intra_op_parallelism_threads(1)
                tf.config.threading.set_inter_op_parallelism_threads(1)
                
                ncf_model = tf.keras.models.load_model('models/ncf_model.keras')
                ncf_cond_encoder = joblib.load('models/ncf_cond_encoder.joblib')
                ncf_drug_encoder = joblib.load('models/ncf_drug_encoder.joblib')
                print("Loaded NCF Deep Learning model lazily.")
        except Exception as e:
            print("Warning during lazy NCF model load:", e)
    return ncf_model, ncf_cond_encoder, ncf_drug_encoder

# Trigger loading on module load
load_models()

# ═══════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ═══════════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    symptoms: List[str]

class PredictionResponse(BaseModel):
    matched_disease: str
    match_score: float
    description: str
    precautions: List[str]
    medications: List[str]
    diets: List[str]
    risk_level: str

class DrugRecommendationSchema(BaseModel):
    drug_name: str
    svd_rating: float
    content_score: float
    hybrid_score: float
    sentiment_score: float
    ncf_rating: float
    recency_rating: float

class RecomResponse(BaseModel):
    disease: str
    recommendations: List[DrugRecommendationSchema]

class FeedbackRequest(BaseModel):
    drug_name: str
    clicked: bool  # True = positive feedback, False = negative

# ═══════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════

@router.post("/predict-disease", response_model=PredictionResponse)
def predict_disease(
    request: PredictRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not request.symptoms:
        raise HTTPException(status_code=400, detail="Symptoms list cannot be empty")
        
    # Standardize input symptoms
    std_symptoms = [s.replace('_', ' ').strip().lower() for s in request.symptoms]
    
    # Map input list to symptom columns
    matched_cols = []
    for s in std_symptoms:
        for col in symptom_cols:
            if col.replace('_', ' ').strip().lower() == s:
                matched_cols.append(col)
                break
                
    if not matched_cols:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="None of the symptoms match our vocabulary"
        )
        
    # Build binary input vector
    input_vector = np.zeros(len(symptom_cols))
    for col in matched_cols:
        input_vector[symptom_cols.index(col)] = 1
        
    # Find matching row in training.csv using cosine similarity of symptom arrays
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cos
    X_train = training_data[symptom_cols].values
    sims = sklearn_cos(input_vector.reshape(1, -1), X_train).flatten()
    
    best_idx = np.argmax(sims)
    best_score = sims[best_idx]
    predicted_disease = training_data.iloc[best_idx]['Disease']
    
    # Query details from unified_data (description, medications, diets, precautions)
    disease_details = unified_data[unified_data['Disease'] == predicted_disease].first_valid_index()
    if disease_details is None:
        # Fallback if merged table doesn't have it
        desc = "No details available."
        precs = []
        meds = []
        diets = []
        risk = "Medium"
    else:
        row = unified_data.loc[disease_details]
        desc = row['Description'] if pd.notnull(row['Description']) else ""
        
        # Parse list columns
        try:
            meds = ast.literal_eval(row['Medication'])
        except:
            meds = []
        try:
            diets = ast.literal_eval(row['Diet'])
        except:
            diets = []
            
        precs = []
        for i in range(1, 5):
            p_col = f'Precaution_{i}'
            if p_col in row and pd.notnull(row[p_col]):
                precs.append(str(row[p_col]))
                
        # Risk level lookup from training_engineered.csv
        risk = training_data.iloc[best_idx].get('disease_risk_level', 'Medium')
        
    # Log prediction activity
    log = UserActivity(
        user_id=current_user.id,
        activity_type="predict",
        details=f"Symptoms: {request.symptoms} -> Match: {predicted_disease} ({best_score:.2f})"
    )
    db.add(log)
    db.commit()
    
    return {
        "matched_disease": predicted_disease,
        "match_score": float(best_score),
        "description": desc,
        "precautions": precs,
        "medications": meds,
        "diets": diets,
        "risk_level": risk
    }

@router.get("/recommendations", response_model=RecomResponse)
def get_recommendations(
    disease: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    disease_title = disease.strip().title()
    
    # 1. Fetch condition drugs from reviews_data matching
    # Map disease names (prognosis labels) to condition categories in drug reviews
    # Simple semantic mapping
    cond_mapping = {
        'Gerd': 'GERD',
        'Acne': 'Acne',
        'Allergy': 'Allergies',
        'Hypertension ': 'High Blood Pressure',
        'Hypothyroidism': 'Hypothyroidism',
        'Diabetes ': 'Diabetes Type 2',
        'Migraine': 'Migraine'
    }
    mapped_condition = cond_mapping.get(disease_title, 'Pain') # default fallback
    
    condition_drugs = reviews_data[reviews_data['condition'] == mapped_condition]['drugName'].unique()
    if len(condition_drugs) == 0:
        # Fallback to general top drugs if condition is not found
        condition_drugs = reviews_data['drugName'].value_counts().head(5).index.tolist()
        
    recommendations_list = []
    
    # Retrieve user profile
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    user_age = profile.age if profile and profile.age else 30
    
    for drug in condition_drugs[:6]: # Limit to top 6 drugs to serve quickly
        
        # A. SVD Rating
        svd_score = 3.0 # neutral default
        if svd_model is not None:
            # SVD predicts norm [0.5, 5]. Scale to 1-10.
            svd_score = svd_model.predict(current_user.id, drug).est * 2.0
            
        # B. Content similarity score of disease description vs drug reviews
        content_score = 0.5
        if cosine_sim is not None and disease_title in unified_data['Disease'].values:
            # Similarity between user's predicted disease and drug's conditions
            drug_conds = reviews_data[reviews_data['drugName'] == drug]['condition'].unique()
            sims = []
            for dc in drug_conds:
                dc_title = str(dc).strip().title()
                # Find matching disease indices
                if disease_title == dc_title:
                    sims.append(1.0)
                else:
                    # Simple reverse mapping back to prognosis keys
                    reverse_map = {v: k for k, v in cond_mapping.items()}
                    prognosis_key = reverse_map.get(dc, dc_title)
                    
                    # Compute similarity
                    disease_names_list = unified_data['Disease'].tolist()
                    if disease_title in disease_names_list and prognosis_key in disease_names_list:
                        u_idx = disease_names_list.index(disease_title)
                        d_idx = disease_names_list.index(prognosis_key)
                        sims.append(float(cosine_sim[u_idx, d_idx]))
            content_score = max(sims) if sims else 0.1
            
        # C. Hybrid score
        hybrid_score = 0.4 * content_score + 0.6 * (svd_score / 10.0)
        
        # D. Sentiment score (NLP polarity)
        sentiment_score = 5.0 # neutral default
        if drug in sentiment_stats.index:
            avg_pol = sentiment_stats.loc[drug].get('avg_polarity', 0.0)
            sentiment_score = 1.0 + (avg_pol + 1.0) * 4.5  # Map [-1, 1] to [1, 10]
            
        # E. Deep Learning NCF Rating
        ncf_rating = 5.0
        active_ncf_model, active_ncf_cond_encoder, active_ncf_drug_encoder = get_ncf_model()
        if active_ncf_model is not None and active_ncf_cond_encoder is not None and active_ncf_drug_encoder is not None:
            try:
                # Get index
                cond_lbl = mapped_condition if mapped_condition in active_ncf_cond_encoder.classes_ else active_ncf_cond_encoder.classes_[0]
                drug_lbl = drug if drug in active_ncf_drug_encoder.classes_ else active_ncf_drug_encoder.classes_[0]
                
                c_idx = active_ncf_cond_encoder.transform([cond_lbl])[0]
                d_idx = active_ncf_drug_encoder.transform([drug_lbl])[0]
                
                # Predict (sigmoid output 0-1)
                pred_ncf = active_ncf_model.predict([np.array([c_idx]), np.array([d_idx])], verbose=0)[0][0]
                ncf_rating = pred_ncf * 10.0
            except Exception as e:
                print("Error predicting with NCF model:", e)
                
        # F. Recency-weighted rating
        # Simulating recency weighted decay rating: older drugs are slightly discounted
        drug_reviews = reviews_data[reviews_data['drugName'] == drug]
        if not drug_reviews.empty:
            avg_rating = drug_reviews['rating'].mean()
            # simulate 5% penalty for old reviews
            recency_rating = avg_rating * 0.95
        else:
            recency_rating = 5.0
            
        recommendations_list.append({
            "drug_name": drug,
            "svd_rating": float(round(svd_score, 2)),
            "content_score": float(round(content_score * 10.0, 2)), # Scale similarity to 1-10
            "hybrid_score": float(round(hybrid_score * 10.0, 2)),
            "sentiment_score": float(round(sentiment_score, 2)),
            "ncf_rating": float(round(ncf_rating, 2)),
            "recency_rating": float(round(recency_rating, 2))
        })
        
    # Log recommendation retrieval
    log = UserActivity(
        user_id=current_user.id,
        activity_type="recommend_drugs",
        details=f"Condition: {mapped_condition} -> Top Rec: {recommendations_list[0]['drug_name'] if recommendations_list else 'None'}"
    )
    db.add(log)
    db.commit()
    
    return {
        "disease": disease_title,
        "recommendations": recommendations_list
    }

# ═══════════════════════════════════════════════════════════════════════
# REINFORCEMENT LEARNING BANDIT PLAYGROUND
# ═══════════════════════════════════════════════════════════════════════

@router.get("/bandit/recommendations")
def get_bandit_recommendations(current_user: User = Depends(get_current_user)):
    """Samples Thompson Sampling probabilities in real-time to recommend drugs."""
    sampled_values = []
    for d in bandit_arms:
        # Draw sample from Beta distribution priors
        sample = np.random.beta(bandit_alpha[d], bandit_beta[d])
        sampled_values.append((d, sample))
        
    # Sort by sampled values
    sampled_values.sort(key=lambda x: x[1], reverse=True)
    
    # Return top 5 recommendations along with their current prior parameters
    results = []
    for d, score in sampled_values[:5]:
        results.append({
            "drug_name": d,
            "score": float(score),
            "alpha": int(bandit_alpha[d]),
            "beta": int(bandit_beta[d]),
            "expected_reward": float(bandit_alpha[d] / (bandit_alpha[d] + bandit_beta[d]))
        })
    return results

@router.post("/bandit/feedback")
def submit_bandit_feedback(
    feedback: FeedbackRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Updates Bandit priors online based on click/rate feedback."""
    drug = feedback.drug_name
    if drug not in bandit_arms:
        raise HTTPException(status_code=400, detail="Drug not found in Bandit arms list")
        
    if feedback.clicked:
        bandit_alpha[drug] += 1
    else:
        bandit_beta[drug] += 1
        
    # Log feedback action
    log = UserActivity(
        user_id=current_user.id,
        activity_type="bandit_feedback",
        details=f"Drug: {drug} | Positive Feedback: {feedback.clicked}"
    )
    db.add(log)
    db.commit()
    
    return {
        "status": "success",
        "alpha": int(bandit_alpha[drug]),
        "beta": int(bandit_beta[drug]),
        "expected_reward": float(bandit_alpha[drug] / (bandit_alpha[drug] + bandit_beta[drug]))
    }
