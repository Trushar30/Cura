# src/evaluation.py
"""
Recommendation system evaluation metrics module.
"""

import numpy as np

def compute_rmse(y_true, y_pred):
    """Computes Root Mean Squared Error."""
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def compute_mae(y_true, y_pred):
    """Computes Mean Absolute Error."""
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def compute_precision_at_k(recommended_list, actual_list, k=5):
    """Computes Precision @ K."""
    recommended_k = recommended_list[:k]
    hits = len(set(recommended_k) & set(actual_list))
    return hits / k

def compute_recall_at_k(recommended_list, actual_list, k=5):
    """Computes Recall @ K."""
    recommended_k = recommended_list[:k]
    hits = len(set(recommended_k) & set(actual_list))
    if len(actual_list) == 0:
        return 0.0
    return hits / len(actual_list)
