# %% [markdown]
"""
# 📅 Notebook 07: Context-Aware Recommendation Engine
## Healthcare Recommendation System — Phase 4

This notebook implements context-aware recommendation techniques focused on time and trends:
1. Parses dates to identify seasonal popularity.
2. Implements trend detection for growing/trending conditions.
3. Implements recency-weighted scoring: ratings are discounted over time using exponential decay.
4. Compares all-time vs. recent popularity rankings.
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

# Load raw reviews
reviews_df = pd.read_csv('data/raw/drug_reviews.tsv', sep='\t')
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

print("Reviews date range:", reviews_df['date'].min(), "to", reviews_df['date'].max())

# %% [markdown]
"""
## 1. Extract Date Features (Seasonality)
"""

# %%
# Extract Month, Year, and Season
reviews_df['month'] = reviews_df['date'].dt.month
reviews_df['year'] = reviews_df['date'].dt.year

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

reviews_df['season'] = reviews_df['month'].apply(get_season)

# Plot seasonal popularity for top 5 conditions
top_5_conditions = reviews_df['condition'].value_counts().head(5).index.tolist()
seasonal_df = reviews_df[reviews_df['condition'].isin(top_5_conditions)]
seasonal_counts = seasonal_df.groupby(['condition', 'season']).size().unstack(fill_value=0)

# Normalize by row sum to see proportion per season
seasonal_prop = seasonal_counts.div(seasonal_counts.sum(axis=1), axis=0)

plt.figure(figsize=(12, 6))
seasonal_prop.plot(kind='bar', stacked=True, colormap='viridis', figsize=(12, 6))
plt.title('Seasonal Distribution of Reviews for Top 5 Conditions', fontsize=14, fontweight='bold')
plt.xlabel('Condition')
plt.ylabel('Proportion of Reviews')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Season')
plt.tight_layout()
plt.savefig('reports/context_seasonality.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved seasonality chart to reports/context_seasonality.png")

# %% [markdown]
"""
## 2. Trend Detection (Rolling Averages & Growth)
"""

# %%
# Group by Month-Year to see monthly reviews count for top conditions
reviews_df['year_month'] = reviews_df['date'].dt.to_period('M')
monthly_counts = reviews_df[reviews_df['condition'].isin(top_5_conditions)].groupby(['year_month', 'condition']).size().unstack(fill_value=0)

# Apply 3-month rolling average
rolling_counts = monthly_counts.rolling(window=3).mean()

plt.figure(figsize=(14, 7))
for cond in top_5_conditions:
    plt.plot(rolling_counts.index.astype(str), rolling_counts[cond], label=cond, linewidth=2.5)
    
plt.title('3-Month Rolling Average of Review Counts (Top 5 Conditions)', fontsize=15, fontweight='bold')
plt.xlabel('Year-Month')
plt.ylabel('Review Count (Rolling Average)')
# Rotate and sub-select x-labels to avoid clutter
plt.xticks(rotation=45, ha='right')
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(20))
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('reports/context_trends.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved trend detection chart to reports/context_trends.png")

# %% [markdown]
"""
## 3. Popularity Rankings: Recent vs. All-Time
"""

# %%
# Define "Recent" as the last 12 months of data in the dataset
max_date = reviews_df['date'].max()
cutoff_date = max_date - pd.DateOffset(months=12)

recent_df = reviews_df[reviews_df['date'] >= cutoff_date]

all_time_drugs = reviews_df['drugName'].value_counts().head(10).index.tolist()
recent_drugs = recent_df['drugName'].value_counts().head(10).index.tolist()

popularity_compare = pd.DataFrame({
    'Rank': range(1, 11),
    'All-Time Most Popular': all_time_drugs,
    'Recent Most Popular (Last 12M)': recent_drugs
})

print("\n📊 POPULARITY RANKING COMPARISON")
print("="*60)
print(popularity_compare.to_string(index=False))

# %% [markdown]
"""
## 4. Recency-Weighted Scoring
"""

# %%
# score = base_rating * exp(-lambda * age_in_days)
# Let's define a lambda decay factor of 0.0005 (half-life of ~1386 days or ~3.8 years)
decay_lambda = 0.0005

# Reference date is the latest date in the reviews dataset
ref_date = reviews_df['date'].max()
reviews_df['age_in_days'] = (ref_date - reviews_df['date']).dt.days
reviews_df['decay_weight'] = np.exp(-decay_lambda * reviews_df['age_in_days'])
reviews_df['recency_weighted_rating'] = reviews_df['rating'] * reviews_df['decay_weight']

# Example: show top drugs for a specific condition with and without recency weighting
target_condition = 'Depression'
dep_reviews = reviews_df[reviews_df['condition'] == target_condition]

# Average rating
avg_ratings = dep_reviews.groupby('drugName')['rating'].mean()
# Recency weighted average rating
weighted_ratings = dep_reviews.groupby('drugName').apply(
    lambda x: np.sum(x['rating'] * x['decay_weight']) / np.sum(x['decay_weight'])
)
review_counts = dep_reviews.groupby('drugName').size()

compare_drugs = pd.DataFrame({
    'All-Time Avg Rating': avg_ratings,
    'Recency-Weighted Avg Rating': weighted_ratings,
    'Review Count': review_counts
}).loc[review_counts[review_counts >= 5].index].sort_values('Recency-Weighted Avg Rating', ascending=False)

print(f"\n📊 DRUG RANKING FOR '{target_condition}' (Min 5 reviews)")
print("="*70)
print(compare_drugs.head(10).to_string())

print("\n✅ Context-aware recommendation phase completed successfully!")
