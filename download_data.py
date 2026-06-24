#!/usr/bin/env python3
"""
============================================================================
Healthcare Recommendation System — Open Source Data Collection (v2)
============================================================================
Downloads REAL open-source healthcare datasets from verified GitHub repos.

Sources:
  1. sohamvsonar/Disease-Prediction-and-Medical-Recommendation-System (GitHub)
  2. itachi9604/healthcare-chatbot (GitHub)
  3. UCI Drug Review Dataset mirrors (GitHub)

Datasets collected:
  - Training.csv              — Binary symptom-disease matrix (4920 rows × 133 symptoms)
  - symptoms_df.csv           — Symptom-disease mapping (long format)
  - Symptom-severity.csv      — Symptom severity weights
  - description.csv           — Disease descriptions
  - symptom_precaution.csv    — Disease precautions (source 1)
  - precautions_df.csv        — Disease precautions (source 2)
  - medications.csv           — Medication recommendations
  - diets.csv                 — Diet recommendations
  - workout_df.csv            — Exercise recommendations
  - drug_reviews.tsv          — 160K+ real patient drug reviews with ratings
"""

import os
import urllib.request
import ssl
import sys

# ── Configuration ────────────────────────────────────────────────────────────
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# SSL context
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

# ── VERIFIED Dataset URLs ───────────────────────────────────────────────────
# All URLs verified via GitHub API tree listing
DATASETS = [
    # ─── From sohamvsonar/Disease-Prediction-and-Medical-Recommendation-System ───
    {
        "filename": "Training.csv",
        "urls": [
            "https://raw.githubusercontent.com/sohamvsonar/Disease-Prediction-and-Medical-Recommendation-System/main/dataset/Training.csv",
            "https://raw.githubusercontent.com/itachi9604/healthcare-chatbot/master/Data/Training.csv",
        ],
        "description": "Binary symptom-disease matrix (4920 rows × 133 symptoms + prognosis)"
    },
    {
        "filename": "symptoms_df.csv",
        "urls": [
            "https://raw.githubusercontent.com/sohamvsonar/Disease-Prediction-and-Medical-Recommendation-System/main/dataset/symptoms_df.csv",
        ],
        "description": "Symptom-disease mapping in long format (columns: Disease, Symptom_1..Symptom_17)"
    },
    {
        "filename": "Symptom-severity.csv",
        "urls": [
            "https://raw.githubusercontent.com/sohamvsonar/Disease-Prediction-and-Medical-Recommendation-System/main/dataset/Symptom-severity.csv",
        ],
        "description": "Severity weight (1-7) for each symptom"
    },
    {
        "filename": "description.csv",
        "urls": [
            "https://raw.githubusercontent.com/sohamvsonar/Disease-Prediction-and-Medical-Recommendation-System/main/dataset/description.csv",
        ],
        "description": "Text description for each disease"
    },
    {
        "filename": "precautions_df.csv",
        "urls": [
            "https://raw.githubusercontent.com/sohamvsonar/Disease-Prediction-and-Medical-Recommendation-System/main/dataset/precautions_df.csv",
        ],
        "description": "4 precaution recommendations per disease"
    },
    {
        "filename": "medications.csv",
        "urls": [
            "https://raw.githubusercontent.com/sohamvsonar/Disease-Prediction-and-Medical-Recommendation-System/main/dataset/medications.csv",
        ],
        "description": "Medication recommendations per disease"
    },
    {
        "filename": "diets.csv",
        "urls": [
            "https://raw.githubusercontent.com/sohamvsonar/Disease-Prediction-and-Medical-Recommendation-System/main/dataset/diets.csv",
        ],
        "description": "Diet recommendations per disease"
    },
    {
        "filename": "workout_df.csv",
        "urls": [
            "https://raw.githubusercontent.com/sohamvsonar/Disease-Prediction-and-Medical-Recommendation-System/main/dataset/workout_df.csv",
        ],
        "description": "Exercise/workout recommendations per disease"
    },
    # ─── Additional from itachi9604/healthcare-chatbot ───
    {
        "filename": "symptom_Description.csv",
        "urls": [
            "https://raw.githubusercontent.com/itachi9604/healthcare-chatbot/master/MasterData/symptom_Description.csv",
        ],
        "description": "Alternate disease descriptions source"
    },
    {
        "filename": "symptom_precaution.csv",
        "urls": [
            "https://raw.githubusercontent.com/itachi9604/healthcare-chatbot/master/MasterData/symptom_precaution.csv",
        ],
        "description": "Alternate disease precautions source"
    },
    {
        "filename": "Testing.csv",
        "urls": [
            "https://raw.githubusercontent.com/itachi9604/healthcare-chatbot/master/Data/Testing.csv",
        ],
        "description": "Test split for model evaluation (42 rows)"
    },
    # ─── UCI Drug Review Dataset (Real patient reviews) ───
    {
        "filename": "drug_reviews.tsv",
        "urls": [
            "https://raw.githubusercontent.com/Narasimha-007/Drug-Review-analysis-using-NLP-technique/main/drugsComTrain_raw.tsv",
            "https://raw.githubusercontent.com/MohammedAlJaff/drug_review_analysis/main/data/drugsComTrain_raw.tsv",
        ],
        "description": "160K+ real patient drug reviews with 10-star ratings (UCI ML Repository)"
    },
]


def download_file(url, filepath, filename):
    """Download a file from URL with error handling."""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=60) as response:
            data = response.read()
            with open(filepath, "wb") as f:
                f.write(data)
            size_kb = len(data) / 1024
            repo = "/".join(url.split("/")[3:5])
            print(f"   ✅ {filename:<30} {size_kb:>10.1f} KB  ← {repo}")
            return True
    except Exception as e:
        short_err = str(e)[:60]
        print(f"   ⚠️  {filename:<30} FAILED ({short_err})")
        return False


# ============================================================================
# DOWNLOAD ALL DATASETS
# ============================================================================
print("=" * 72)
print("  📥 DOWNLOADING OPEN-SOURCE HEALTHCARE DATASETS")
print("=" * 72)
print(f"\n  Target: {RAW_DIR}\n")

success_count = 0
fail_count = 0
downloaded_files = []

for ds in DATASETS:
    filename = ds["filename"]
    filepath = os.path.join(RAW_DIR, filename)

    downloaded = False
    for url in ds["urls"]:
        if download_file(url, filepath, filename):
            downloaded = True
            success_count += 1
            downloaded_files.append(filename)
            break
        else:
            print(f"      ↳ Trying next mirror...")

    if not downloaded:
        fail_count += 1
        print(f"      ❌ All mirrors failed for {filename}")

# ============================================================================
# VERIFY ALL DOWNLOADS
# ============================================================================
print(f"\n{'=' * 72}")
print("  🔍 VERIFICATION & DATA PREVIEW")
print(f"{'=' * 72}\n")

import pandas as pd

print(f"  {'File':<35} {'Rows':>8} {'Cols':>6} {'Size':>10}")
print(f"  {'─' * 63}")

total_rows = 0
total_size = 0

for filename in sorted(os.listdir(RAW_DIR)):
    filepath = os.path.join(RAW_DIR, filename)
    if not os.path.isfile(filepath):
        continue
    try:
        sep = "\t" if filename.endswith(".tsv") else ","
        df = pd.read_csv(filepath, sep=sep, encoding="utf-8", on_bad_lines="skip")
        size_kb = os.path.getsize(filepath) / 1024
        total_rows += len(df)
        total_size += size_kb
        print(f"  ✅ {filename:<33} {len(df):>8} {df.shape[1]:>6} {size_kb:>8.1f} KB")
    except Exception as e:
        print(f"  ❌ {filename:<33} Error: {e}")

print(f"  {'─' * 63}")
print(f"  {'TOTAL':<33} {total_rows:>8} {'':>6} {total_size:>8.1f} KB")
print(f"  {'':>33} {'':>8} {'':>6} {total_size/1024:>7.1f} MB")

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'=' * 72}")
print(f"  ✅ DOWNLOAD COMPLETE: {success_count} succeeded, {fail_count} failed")
print(f"{'=' * 72}")

print(f"""
  📊 Dataset Summary:
  ────────────────────
  • Training.csv       → {total_rows} total rows across all files
  • 41 diseases with binary symptom encoding (133 features)
  • Symptom severity weights for prioritization
  • Disease descriptions for NLP / content-based filtering
  • Medications, diets, workouts for recommendation output
  • Real drug reviews with ratings for sentiment analysis
  
  📂 All data saved to: {RAW_DIR}
""")
