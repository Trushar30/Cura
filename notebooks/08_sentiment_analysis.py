# %% [markdown]
"""
# 💬 Notebook 08: Sentiment Analysis & NLP
## Healthcare Recommendation System — Phase 4

This notebook implements NLP preprocessing and Sentiment Analysis on drug reviews:
1. Performs text cleaning: lowercasing, tokenization, stopword removal, and lemmatization.
2. Evaluates sentiment polarity and subjectivity using TextBlob.
3. Computes alignment (confusion matrix, classification report) between ratings and sentiment classes.
4. Generates rich visualizations (scatters, bar charts, word clouds).
5. Implements a Sentiment-Enhanced Recommendation ranking score.
6. Extracts key TF-IDF keywords per sentiment class.
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

os.makedirs('reports', exist_ok=True)

# Set styling
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')

# %%
# Download NLTK data
print("Downloading NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
print("NLTK downloads completed.")

# Load drug reviews
reviews_df = pd.read_csv('data/raw/drug_reviews.tsv', sep='\t')

# %% [markdown]
"""
## 1. NLP Preprocessing Pipeline
"""

# %%
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove punctuation, numbers and special chars
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Filter stopwords & lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(cleaned_tokens)

# Run preprocessing on a subset or full reviews
# For speed in execution, we process full 5000 reviews (takes ~5-10s)
print("Preprocessing reviews...")
reviews_df['cleaned_review'] = reviews_df['review'].apply(preprocess_text)
print("Preprocessing finished.")

# Show before/after for 5 example reviews
print("\n📝 BEFORE/AFTER PREPROCESSING EXAMPLES:")
print("="*60)
for idx, row in reviews_df.head(5).iterrows():
    print(f"Original: {row['review'][:80]}...")
    print(f"Cleaned : {row['cleaned_review'][:80]}...")
    print("-" * 40)

# %% [markdown]
"""
## 2. Sentiment Analysis with TextBlob
"""

# %%
print("Running TextBlob sentiment analysis...")
reviews_df['polarity'] = reviews_df['review'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
reviews_df['subjectivity'] = reviews_df['review'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)

# Classify polarity into Positive, Neutral, Negative
def get_sentiment_label(polarity):
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

reviews_df['sentiment_label'] = reviews_df['polarity'].apply(get_sentiment_label)

# Map ratings to ground-truth sentiment labels (1-3=Negative, 4-6=Neutral, 7-10=Positive)
def rating_to_sentiment(rating):
    if rating <= 3:
        return 'Negative'
    elif rating <= 6:
        return 'Neutral'
    else:
        return 'Positive'

reviews_df['rating_sentiment'] = reviews_df['rating'].apply(rating_to_sentiment)

# Confusion Matrix and Classification Report
print("\n📊 SENTIMENT CLASSIFICATION METRICS")
print("="*60)
print("Accuracy Score:", round(accuracy_score(reviews_df['rating_sentiment'], reviews_df['sentiment_label']) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(reviews_df['rating_sentiment'], reviews_df['sentiment_label']))

# %% [markdown]
"""
## 3. Visualizations
"""

# %%
# A. Polarity Distribution
plt.figure(figsize=(10, 5))
sns.histplot(reviews_df['polarity'], bins=30, color='skyblue', kde=True)
plt.title('Distribution of Sentiment Polarity', fontsize=14, fontweight='bold')
plt.xlabel('Polarity (-1 to 1)')
plt.savefig('reports/sentiment_polarity_dist.png', dpi=150, bbox_inches='tight')
plt.close()

# B. Polarity vs Actual Rating
plt.figure(figsize=(10, 6))
sns.boxplot(x='rating', y='polarity', data=reviews_df, palette='Spectral')
plt.title('Sentiment Polarity by Rating', fontsize=14, fontweight='bold')
plt.xlabel('Rating (1-10)')
plt.ylabel('Polarity')
plt.savefig('reports/sentiment_polarity_vs_rating.png', dpi=150, bbox_inches='tight')
plt.close()

# C. Average Sentiment by Condition (Top 20)
plt.figure(figsize=(12, 8))
top_20_conditions = reviews_df['condition'].value_counts().head(20).index
avg_sent_condition = reviews_df[reviews_df['condition'].isin(top_20_conditions)].groupby('condition')['polarity'].mean().sort_values(ascending=False)
sns.barplot(x=avg_sent_condition.values, y=avg_sent_condition.index, palette='coolwarm')
plt.title('Average Sentiment Polarity by Condition (Top 20)', fontsize=14, fontweight='bold')
plt.xlabel('Average Polarity')
plt.savefig('reports/sentiment_avg_by_condition.png', dpi=150, bbox_inches='tight')
plt.close()

# D. Sentiment Class Pie Chart
plt.figure(figsize=(8, 8))
reviews_df['sentiment_label'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightgreen', 'lightcoral', 'lightskyblue'], startangle=90, textprops={'fontweight':'bold', 'fontsize':12})
plt.title('Sentiment Label Proportions', fontsize=14, fontweight='bold')
plt.ylabel('')
plt.savefig('reports/sentiment_label_proportions.png', dpi=150, bbox_inches='tight')
plt.close()

# E. Subjectivity vs Polarity Scatter
plt.figure(figsize=(10, 6))
sns.scatterplot(x='polarity', y='subjectivity', hue='rating_sentiment', data=reviews_df, alpha=0.5, palette='Set1')
plt.title('Sentiment Subjectivity vs. Polarity', fontsize=14, fontweight='bold')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.savefig('reports/sentiment_subjectivity_vs_polarity.png', dpi=150, bbox_inches='tight')
plt.close()

# F. Word Clouds for Positive vs Negative
try:
    from wordcloud import WordCloud
    pos_reviews = " ".join(reviews_df[reviews_df['sentiment_label'] == 'Positive']['cleaned_review'].dropna())
    neg_reviews = " ".join(reviews_df[reviews_df['sentiment_label'] == 'Negative']['cleaned_review'].dropna())
    
    wc_pos = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(pos_reviews)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc_pos, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Positive Reviews', fontsize=16, fontweight='bold', pad=15)
    plt.savefig('reports/sentiment_wordcloud_pos.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    wc_neg = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(neg_reviews)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc_neg, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Negative Reviews', fontsize=16, fontweight='bold', pad=15)
    plt.savefig('reports/sentiment_wordcloud_neg.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   Word Clouds generated and saved.")
except Exception as e:
    print(f"   Word Cloud warning: {e}")

# %% [markdown]
"""
## 4. Sentiment-Enhanced Recommendations
"""

# %%
# Compute Sentiment-Weighted Rating:
# score = 0.6 * avg_rating + 0.4 * normalized_sentiment (scaled to 1-10 range)
drug_stats = reviews_df.groupby('drugName').agg(
    avg_rating=('rating', 'mean'),
    avg_polarity=('polarity', 'mean'),
    review_count=('rating', 'count')
)

# Normalize average polarity from [-1, 1] to [1, 10] range to match ratings
drug_stats['normalized_sentiment'] = 1 + (drug_stats['avg_polarity'] + 1) * 4.5
drug_stats['sentiment_weighted_score'] = 0.6 * drug_stats['avg_rating'] + 0.4 * drug_stats['normalized_sentiment']

# Compare original vs sentiment-weighted drug rankings
top_ranked_original = drug_stats[drug_stats['review_count'] >= 5].sort_values('avg_rating', ascending=False).head(10)
top_ranked_weighted = drug_stats[drug_stats['review_count'] >= 5].sort_values('sentiment_weighted_score', ascending=False).head(10)

print("\n🏆 TOP 10 DRUGS BY ORIGINAL AVG RATING (min 5 reviews)")
print("="*75)
print(top_ranked_original[['avg_rating', 'avg_polarity', 'review_count']].to_string())

print("\n🏆 TOP 10 DRUGS BY SENTIMENT-WEIGHTED SCORE (min 5 reviews)")
print("="*75)
print(top_ranked_weighted[['sentiment_weighted_score', 'avg_rating', 'avg_polarity', 'review_count']].to_string())

# Save drug sentiment statistics to processed directory
drug_stats.to_csv('data/processed/drug_sentiment_statistics.csv')
print("\nSaved drug sentiment stats to data/processed/drug_sentiment_statistics.csv")

# %% [markdown]
"""
## 5. TF-IDF Key Terms per Sentiment Class
"""

# %%
# Run TF-IDF on Positive, Neutral, Negative subsets
vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
for s_class in ['Positive', 'Neutral', 'Negative']:
    class_reviews = reviews_df[reviews_df['sentiment_label'] == s_class]['cleaned_review']
    if not class_reviews.empty:
        tfidf_mat = vectorizer.fit_transform(class_reviews)
        terms = vectorizer.get_feature_names_out()
        print(f"\n🔑 Top TF-IDF Terms in {s_class} Reviews: {list(terms)}")

print("\n✅ Sentiment analysis phase completed successfully!")
