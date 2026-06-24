# %% [markdown]
"""
# 📊 Notebook 02: Exploratory Data Analysis
## Healthcare Recommendation System — Phase 2

This notebook conducts a deep exploratory analysis of the healthcare data, generating rich
visualizations to understand symptoms, disease prevalence, user reviews, and recommendation patterns.
All charts are saved directly to the `reports/` directory.
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('reports', exist_ok=True)

# Set styling
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')

# Load the raw datasets we need for EDA
training_df = pd.read_csv('data/raw/Training.csv')
testing_df = pd.read_csv('data/raw/Testing.csv')
severity_df = pd.read_csv('data/raw/Symptom-severity.csv')
description_df = pd.read_csv('data/raw/description.csv')
meds_df = pd.read_csv('data/raw/medications.csv')
diets_df = pd.read_csv('data/raw/diets.csv')
workout_df = pd.read_csv('data/raw/workout_df.csv')
reviews_df = pd.read_csv('data/raw/drug_reviews.tsv', sep='\t')

# Clean Unnamed columns
for df in [training_df, testing_df, severity_df, description_df, meds_df, diets_df, workout_df, reviews_df]:
    unnamed = [c for c in df.columns if 'Unnamed' in c]
    if unnamed:
        df.drop(columns=unnamed, inplace=True)

# Standardize prognosis to Disease
training_df.rename(columns={'prognosis': 'Disease'}, inplace=True)
testing_df.rename(columns={'prognosis': 'Disease'}, inplace=True)
workout_df.rename(columns={'disease': 'Disease'}, inplace=True)

# %% [markdown]
"""
## 2.1 Data Quality Analysis
"""

# %%
# Missing values check
plt.figure(figsize=(10, 6))
sns.heatmap(training_df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Missing Values Heatmap (Training.csv)', fontsize=14, fontweight='bold')
plt.savefig('reports/eda_01_missing_values_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# Missing % bar chart across datasets
missing_data = {
    'Dataset': ['Training', 'Testing', 'Severity', 'Description', 'Meds', 'Diets', 'Workout', 'Reviews'],
    'Missing_Pct': [
        training_df.isnull().mean().mean() * 100,
        testing_df.isnull().mean().mean() * 100,
        severity_df.isnull().mean().mean() * 100,
        description_df.isnull().mean().mean() * 100,
        meds_df.isnull().mean().mean() * 100,
        diets_df.isnull().mean().mean() * 100,
        workout_df.isnull().mean().mean() * 100,
        reviews_df.isnull().mean().mean() * 100,
    ]
}
missing_df = pd.DataFrame(missing_data)
plt.figure(figsize=(10, 5))
sns.barplot(x='Dataset', y='Missing_Pct', data=missing_df, palette='Blues_r')
plt.title('Missing Values Percentage by Dataset', fontsize=14, fontweight='bold')
plt.ylabel('Missing Percentage (%)')
plt.ylim(0, 10)
plt.savefig('reports/eda_02_missing_pct_bar.png', dpi=150, bbox_inches='tight')
plt.close()

# %% [markdown]
"""
## 2.2 Disease & Symptom Analysis
"""

# %%
# Disease frequency distribution
plt.figure(figsize=(12, 8))
disease_counts = training_df['Disease'].value_counts()
sns.barplot(x=disease_counts.values, y=disease_counts.index, palette='crest')
plt.title('Disease Frequency Distribution in Training Dataset', fontsize=14, fontweight='bold')
plt.xlabel('Count')
plt.savefig('reports/eda_03_disease_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Top 20 most common symptoms
symptoms = [c for c in training_df.columns if c != 'Disease']
symptom_sums = training_df[symptoms].sum().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=symptom_sums.head(20).values, y=symptom_sums.head(20).index, palette='flare')
plt.title('Top 20 Most Common Symptoms (Training Data)', fontsize=14, fontweight='bold')
plt.xlabel('Frequency')
plt.savefig('reports/eda_04_top_20_symptoms.png', dpi=150, bbox_inches='tight')
plt.close()

# Symptom severity distribution
plt.figure(figsize=(10, 5))
sns.histplot(severity_df['weight'], bins=np.arange(1, 9) - 0.5, kde=False, color='teal', edgecolor='black')
plt.title('Symptom Severity Weights Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Severity Weight')
plt.ylabel('Symptom Count')
plt.xticks(range(1, 8))
plt.savefig('reports/eda_05_symptom_severity.png', dpi=150, bbox_inches='tight')
plt.close()

# %% [markdown]
"""
## 2.3 Symptom Co-occurrence Analysis
"""

# %%
# Correlation of top 20 symptoms
top_symptom_names = symptom_sums.head(20).index
corr_matrix = training_df[top_symptom_names].corr()
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap of Top 20 Symptoms', fontsize=16, fontweight='bold', pad=20)
plt.savefig('reports/eda_06_symptom_correlation.png', dpi=150, bbox_inches='tight')
plt.close()

# %% [markdown]
"""
## 2.4 Drug Review Analysis
"""

# %%
# Rating distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='rating', data=reviews_df, palette='viridis')
plt.title('Drug Review Ratings Distribution (1-10 Scale)', fontsize=14, fontweight='bold')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig('reports/eda_07_rating_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Top 20 drugs by review count
plt.figure(figsize=(10, 6))
top_drugs = reviews_df['drugName'].value_counts().head(20)
sns.barplot(x=top_drugs.values, y=top_drugs.index, palette='magma')
plt.title('Top 20 Drugs by Review Count', fontsize=14, fontweight='bold')
plt.xlabel('Number of Reviews')
plt.savefig('reports/eda_08_top_20_drugs.png', dpi=150, bbox_inches='tight')
plt.close()

# Top 10 conditions by review count
plt.figure(figsize=(10, 5))
top_conds = reviews_df['condition'].value_counts().head(10)
sns.barplot(x=top_conds.values, y=top_conds.index, palette='viridis')
plt.title('Top 10 Conditions by Review Count', fontsize=14, fontweight='bold')
plt.xlabel('Number of Reviews')
plt.savefig('reports/eda_09_top_10_conditions.png', dpi=150, bbox_inches='tight')
plt.close()

# Average rating by condition
plt.figure(figsize=(12, 6))
avg_rating_cond = reviews_df.groupby('condition')['rating'].mean().loc[top_conds.index].sort_values(ascending=False)
sns.barplot(x=avg_rating_cond.values, y=avg_rating_cond.index, palette='mako')
plt.title('Average Rating by Condition (Top 10 Conditions)', fontsize=14, fontweight='bold')
plt.xlabel('Average Rating (1-10)')
plt.xlim(0, 10)
plt.savefig('reports/eda_10_avg_rating_condition.png', dpi=150, bbox_inches='tight')
plt.close()

# Review length and rating vs useful count
reviews_df['review_length'] = reviews_df['review'].fillna('').apply(len)
plt.figure(figsize=(10, 5))
sns.histplot(reviews_df['review_length'], bins=50, color='purple', kde=True)
plt.title('Review Character Length Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Length')
plt.savefig('reports/eda_11_review_length.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='rating', y='usefulCount', data=reviews_df, alpha=0.5, color='coral')
plt.title('Review Rating vs. Useful Count', fontsize=14, fontweight='bold')
plt.xlabel('Rating (1-10)')
plt.ylabel('Useful Count')
plt.savefig('reports/eda_12_rating_vs_useful.png', dpi=150, bbox_inches='tight')
plt.close()

# %% [markdown]
"""
## 2.5 Word Clouds
"""

# %%
try:
    from wordcloud import WordCloud
    
    # Disease descriptions
    desc_text = " ".join(description_df['Description'].dropna())
    wc_desc = WordCloud(width=800, height=400, background_color='white', colormap='cool').generate(desc_text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc_desc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Disease Descriptions', fontsize=16, fontweight='bold', pad=15)
    plt.savefig('reports/eda_13_wordcloud_descriptions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Drug reviews
    review_text = " ".join(reviews_df['review'].dropna().head(1000))
    wc_reviews = WordCloud(width=800, height=400, background_color='black', colormap='Accent').generate(review_text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc_reviews, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Drug Reviews', fontsize=16, fontweight='bold', pad=15)
    plt.savefig('reports/eda_14_wordcloud_reviews.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Word clouds generated successfully")
except Exception as e:
    print(f"   ⚠️ WordCloud generation skipped or failed: {e}")

# %% [markdown]
"""
## 2.6 Treatment Recommendation Distribution
"""

# %%
# Most recommended medications
import ast
med_counts = {}
for med_str in meds_df['Medication'].dropna():
    try:
        med_list = ast.literal_eval(med_str)
        for m in med_list:
            med_counts[m] = med_counts.get(m, 0) + 1
    except:
        pass
        
if med_counts:
    med_counts_df = pd.DataFrame(list(med_counts.items()), columns=['Medication', 'Count']).sort_values('Count', ascending=False)
    plt.figure(figsize=(12, 5))
    sns.barplot(x='Count', y='Medication', data=med_counts_df.head(15), palette='viridis')
    plt.title('Top 15 Most Frequently Prescribed Medications (Overall)', fontsize=14, fontweight='bold')
    plt.savefig('reports/eda_15_top_medications.png', dpi=150, bbox_inches='tight')
    plt.close()

# Diet recommendations
diet_counts = {}
for diet_str in diets_df['Diet'].dropna():
    try:
        diet_list = ast.literal_eval(diet_str)
        for d in diet_list:
            diet_counts[d] = diet_counts.get(d, 0) + 1
    except:
        pass

if diet_counts:
    diet_counts_df = pd.DataFrame(list(diet_counts.items()), columns=['Diet', 'Count']).sort_values('Count', ascending=False)
    plt.figure(figsize=(12, 5))
    sns.barplot(x='Count', y='Diet', data=diet_counts_df.head(15), palette='mako')
    plt.title('Top 15 Most Frequently Recommended Diets', fontsize=14, fontweight='bold')
    plt.savefig('reports/eda_16_top_diets.png', dpi=150, bbox_inches='tight')
    plt.close()

print("\n📊 EDA Notebook successfully generated and executed plots saved to reports/")
