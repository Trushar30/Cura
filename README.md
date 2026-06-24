# 🏥 Personalized Healthcare Recommendation System

A comprehensive, production-grade data science pipeline and recommendation system for personalized healthcare. 

This project implements eight different recommendation strategies—spanning traditional similarity models, collaborative filtering matrix factorization, context-aware popularity trends, NLP sentiment-enhanced ranking, deep learning Neural Collaborative Filtering (NCF), graph path-based traversals, and Reinforcement Learning Multi-Armed Bandits.

---

## 📂 Project Directory Structure

```
Health Recom/
├── .venv/                            # Local python virtual environment
├── data/
│   ├── raw/                          # 12 Raw open-source Kaggle/GitHub datasets
│   │   ├── Training.csv
│   │   ├── Testing.csv
│   │   ├── Symptom-severity.csv
│   │   ├── description.csv
│   │   ├── medications.csv
│   │   ├── diets.csv
│   │   ├── workout_df.csv
│   │   ├── precautions_df.csv
│   │   ├── symptoms_df.csv
│   │   ├── symptom_Description.csv
│   │   ├── symptom_precaution.csv
│   │   └── drug_reviews.tsv
│   └── processed/                    # Cleaned & feature-engineered outputs
│       ├── training_engineered.csv
│       ├── testing_engineered.csv
│       ├── unified_healthcare_data.csv
│       └── user_item_matrix.csv
├── notebooks/                        # Jupyter-compatible pipeline notebooks
│   ├── 01_data_collection_and_loading.py
│   ├── 02_exploratory_data_analysis.py
│   ├── 03_data_preprocessing.py
│   ├── 04_content_based_filtering.py
│   ├── 05_collaborative_filtering.py
│   ├── 06_hybrid_filtering.py
│   ├── 07_context_aware_recommendations.py
│   ├── 08_sentiment_analysis.py
│   ├── 09_deep_learning_model.py
│   ├── 10_graph_based_recommendation.py
│   ├── 11_reinforcement_learning.py
│   └── 12_model_evaluation_comparison.py
├── src/                              # Reusable production modules
│   ├── __init__.py
│   ├── data_loader.py                # Standardized loading utilities
│   ├── preprocessing.py              # NLP & feature engineering pipelines
│   ├── evaluation.py                 # Common scoring metrics (RMSE, Precision)
│   └── recommenders/                 # Individual recommender classes
│       ├── __init__.py
│       ├── content_based.py
│       ├── collaborative.py
│       ├── hybrid.py
│       ├── deep_learning.py
│       └── graph_based.py
├── models/                           # Serialized joblib/keras/pickle models
├── reports/                          # Generated plots and evaluation summaries
├── requirements.txt                  # Project dependencies
└── README.md                         # Project documentation
```

---

## 🛠️ Setup & Execution Instructions

Ensure you use the local Python virtual environment to execute the scripts:

1. **Activate the Virtual Environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Install/Verify Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Data Science Pipeline:**
   You can run any notebook script in order to verify or compile predictions:
   ```bash
   python notebooks/01_data_collection_and_loading.py
   python notebooks/03_data_preprocessing.py
   python notebooks/12_model_evaluation_comparison.py
   ```

---

## 📓 Notebook Pipeline Overview

- **01: Data Loading & Collection:** Merges disease datasets on disease labels and standardizes formatting.
- **02: Exploratory Data Analysis (EDA):** Generates 16 plots analyzing symptom correlations, ratings, and conditions.
- **03: Preprocessing & Feature Engineering:** Mapped symptom severity weights, computed risk levels, pivoted drug ratings, and vectorized descriptions.
- **04: Content-Based Filtering:** TF-IDF + Cosine Similarity recommendation function and matching symptom arrays.
- **05: Collaborative Filtering:** Benchmarks SVD, SVDpp, NMF, User-KNN, and Item-KNN with 5-fold cross-validation.
- **06: Hybrid Filtering:** Blended strategy (weighted alpha tuning) and returning-user switching logic.
- **07: Context-Aware Recommendations:** Seasonal condition distributions and recency-decay rating weightings.
- **08: Sentiment Analysis:** Tokenization, lemmatization, TextBlob sentiment extraction, and sentiment-weighted ranking adjustments.
- **09: Deep Learning Model:** Trains a Keras Neural Collaborative Filtering (NCF) model using embeddings.
- **10: Graph-Based Recommendation:** Compiles a heterogeneous Knowledge Graph in NetworkX (602 nodes, 1293 edges).
- **11: Reinforcement Learning:** Multi-Armed Bandit simulation comparing ε-Greedy, UCB, and Thompson Sampling.
- **12: Model Evaluation & Comparison:** Summarizes the entire pipeline in a single comparison table.

---

## 📊 Final Model Comparison Results

Evaluation was executed over the 5,000 rating reviews (scaled to standard range):

| Model | RMSE | MAE | Avg Similarity | Coverage % | Status |
|---|---|---|---|---|---|
| **Content-Based (TF-IDF)** | *N/A* | *N/A* | 0.0863 | 100.0% | ✅ Trained |
| **CF: SVD** | 1.1940 | 0.9801 | *N/A* | 100.0% | ✅ Trained |
| **CF: SVDpp** | 1.1808 | 0.9733 | *N/A* | 100.0% | ✅ Trained |
| **CF: KNN (User)** | 1.1752 | 0.9720 | *N/A* | 100.0% | ✅ Trained |
| **CF: KNN (Item)** | 1.1750 | 0.9719 | *N/A* | 100.0% | ✅ Trained |
| **CF: NMF** | 1.1748 | 0.9716 | *N/A* | 100.0% | ✅ Trained |
| **Hybrid (best α=0.3)** | 1.1816 | *N/A* | *N/A* | 100.0% | ✅ Trained |
| **Deep Learning (NCF)** | **0.2317** | **0.1907** | *N/A* | 100.0% | ✅ Trained |
| **Sentiment Analysis** | *N/A* | *N/A* | *N/A* | 100.0% | ✅ Computed (60.7% Acc) |

### 🏆 Key Takeaways & Recommendations:
1. **Core Engine:** A **Hybrid Filtering Model** is recommended for production. It achieves the best balance between personalization (Collaborative SVD) and cold-start robustness (Content-Based TF-IDF).
2. **Deep Learning:** Neural Collaborative Filtering (NCF) achieves the lowest prediction RMSE on rating vectors, capturing complex non-linear combinations.
3. **Enhancement Layer:** Use **Sentiment Analysis** to refine ranking metrics by adjusting average ratings based on text review polarity scores.
