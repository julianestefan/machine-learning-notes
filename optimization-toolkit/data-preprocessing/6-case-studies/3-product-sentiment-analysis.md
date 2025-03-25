# Case Study: Product Sentiment Analysis

This case study demonstrates a complete preprocessing and feature engineering workflow for product sentiment analysis, focusing on text data processing.

## Problem Statement

Analyze customer reviews to predict sentiment (positive, negative, or neutral) about products. This is a text classification problem with significant applications in e-commerce, brand monitoring, and customer experience management.

## Dataset Overview

For this case study, we'll use the "amazon_reviews_multi" dataset from Hugging Face's datasets library. This dataset contains Amazon product reviews in multiple languages. We'll focus on the English subset.

## Preprocessing Pipeline

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Load the dataset from Hugging Face
try:
    from datasets import load_dataset
    
    # Load the English Amazon reviews dataset
    dataset = load_dataset('amazon_reviews_multi', 'en')
    
    # Convert to pandas dataframe for easier handling
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    
    # Combine for exploration
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Select relevant columns
    df = df[['review_title', 'review_body', 'stars', 'product_category']]
    
    # Create sentiment labels based on star rating
    df['sentiment'] = pd.cut(
        df['stars'], 
        bins=[0, 2, 3, 5], 
        labels=['negative', 'neutral', 'positive']
    )
    
    print(f"Successfully loaded Amazon Reviews dataset with {df.shape[0]} reviews")

except Exception as e:
    print(f"Could not load dataset from Hugging Face: {e}")
    print("Creating a synthetic product reviews dataset...")
    
    # Create synthetic product reviews
    np.random.seed(42)
    
    # Synthetic product categories
    categories = ['Electronics', 'Books', 'Clothing', 'Home', 'Beauty']
    
    # Generate synthetic star ratings with realistic distribution (skewed positive)
    stars = np.random.choice([1, 2, 3, 4, 5], size=5000, p=[0.05, 0.1, 0.15, 0.3, 0.4])
    
    # Synthetic review templates
    positive_templates = [
        "I love this {product}. It's {adj} and {adj}.",
        "This {product} is {adj}! I'm very satisfied with my purchase.",
        "Excellent {product}, worth every penny. It's very {adj} and {adj}.",
        "I'm impressed with this {product}. {adj} quality and {adj} performance.",
        "Great {product}! It's {adj} and {adj}. Highly recommend."
    ]
    
    neutral_templates = [
        "This {product} is okay. It's {adj} but not {adj}.",
        "Average {product}. Nothing special but gets the job done.",
        "The {product} works as expected. Not {adj}, but not {adj} either.",
        "Decent {product}. Has some {adj} features but some {adj} aspects too.",
        "Neutral about this {product}. Some {adj} points, some {adj} points."
    ]
    
    negative_templates = [
        "Disappointed with this {product}. It's {adj} and {adj}.",
        "Don't buy this {product}. Very {adj} and {adj} quality.",
        "This {product} is terrible. It's {adj} and not worth the money.",
        "Avoid this {adj} {product}. Had a {adj} experience with it.",
        "Unhappy with my purchase. The {product} is {adj} and {adj}."
    ]
    
    positive_adj = ['excellent', 'amazing', 'fantastic', 'perfect', 'reliable', 'high-quality', 'durable', 'convenient', 'worth it', 'innovative']
    neutral_adj = ['decent', 'acceptable', 'adequate', 'standard', 'typical', 'expected', 'average', 'okay', 'modest', 'fair']
    negative_adj = ['disappointing', 'poor', 'faulty', 'unreliable', 'flimsy', 'defective', 'cheap', 'uncomfortable', 'low-quality', 'frustrating']
    
    products = {
        'Electronics': ['laptop', 'headphones', 'smartphone', 'camera', 'TV', 'speaker', 'tablet', 'monitor', 'mouse', 'keyboard'],
        'Books': ['novel', 'textbook', 'cookbook', 'biography', 'self-help book', 'reference book', 'magazine', 'journal', 'children\'s book', 'comic book'],
        'Clothing': ['shirt', 'pants', 'dress', 'jacket', 'shoes', 'hat', 'socks', 'gloves', 'sweater', 'scarf'],
        'Home': ['furniture', 'decoration', 'pillow', 'blanket', 'lamp', 'rug', 'curtain', 'kitchenware', 'appliance', 'tool'],
        'Beauty': ['shampoo', 'lotion', 'makeup', 'perfume', 'skincare', 'haircare', 'face wash', 'cream', 'serum', 'mask']
    }
    
    # Generate synthetic reviews
    reviews = []
    titles = []
    categories_list = []
    
    for star in stars:
        # Select category and product
        category = np.random.choice(categories)
        product = np.random.choice(products[category])
        categories_list.append(category)
        
        # Generate review based on star rating
        if star >= 4:  # Positive
            template = np.random.choice(positive_templates)
            adj_list = np.random.choice(positive_adj, size=2, replace=False)
            title_prefix = np.random.choice(['Love this', 'Great', 'Excellent', 'Amazing', 'Perfect'])
        elif star == 3:  # Neutral
            template = np.random.choice(neutral_templates)
            adj_list = np.random.choice(neutral_adj, size=2, replace=False)
            title_prefix = np.random.choice(['Okay', 'Decent', 'Average', 'Acceptable', 'Fair'])
        else:  # Negative
            template = np.random.choice(negative_templates)
            adj_list = np.random.choice(negative_adj, size=2, replace=False)
            title_prefix = np.random.choice(['Disappointing', 'Poor', 'Avoid', 'Terrible', 'Not worth it'])
        
        # Fill in the template
        review = template.format(product=product, adj=adj_list[0], adj2=adj_list[1])
        reviews.append(review)
        
        # Generate title
        title = f"{title_prefix} {product}"
        titles.append(title)
    
    # Create DataFrame
    df = pd.DataFrame({
        'review_title': titles,
        'review_body': reviews,
        'stars': stars,
        'product_category': categories_list,
        'sentiment': pd.cut(stars, bins=[0, 2, 3, 5], labels=['negative', 'neutral', 'positive'])
    })
    
    print(f"Created synthetic product reviews dataset with {df.shape[0]} reviews")

# Download necessary NLTK resources
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    print("Could not download NLTK resources. Proceeding without them.")
    
# Preview the data
print("\nData preview:")
print(df.head())

# Basic statistics
print("\nSentiment distribution:")
print(df['sentiment'].value_counts())

print("\nProduct category distribution:")
print(df['product_category'].value_counts().head(10))  # Show top 10 categories

print("\nRating distribution:")
print(df['stars'].value_counts().sort_index())
```

## Exploratory Data Analysis

```python
# Plot sentiment distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='sentiment', data=df, palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Plot star rating distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='stars', data=df, palette='viridis')
plt.title('Rating Distribution')
plt.xlabel('Stars')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Review length analysis
df['review_length'] = df['review_body'].apply(len)
df['title_length'] = df['review_title'].apply(len)
df['word_count'] = df['review_body'].apply(lambda x: len(str(x).split()))

# Review length by sentiment
plt.figure(figsize=(12, 6))
sns.boxplot(x='sentiment', y='word_count', data=df)
plt.title('Review Word Count by Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Word Count')
plt.show()

# Review length distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='word_count', bins=50, kde=True)
plt.title('Review Word Count Distribution')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.xlim(0, df['word_count'].quantile(0.99))  # Limit x-axis to 99th percentile to handle outliers
plt.show()

# Category vs. sentiment
plt.figure(figsize=(14, 8))
categories_to_plot = df['product_category'].value_counts().head(8).index  # Top 8 categories
category_sentiment = pd.crosstab(
    df[df['product_category'].isin(categories_to_plot)]['product_category'], 
    df[df['product_category'].isin(categories_to_plot)]['sentiment'], 
    normalize='index'
).sort_values(by='positive', ascending=False)

category_sentiment.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Sentiment Distribution Across Top Product Categories')
plt.xlabel('Product Category')
plt.ylabel('Proportion')
plt.xticks(rotation=45)
plt.legend(title='Sentiment')
plt.tight_layout()
plt.show()

# Most common words by sentiment
from collections import Counter
import string

def get_top_words(text_series, n=20, min_length=3):
    # Combine all text
    all_text = ' '.join(text_series.astype(str))
    
    # Remove punctuation and convert to lowercase
    all_text = all_text.lower()
    all_text = all_text.translate(str.maketrans('', '', string.punctuation))
    
    # Split into words
    words = all_text.split()
    
    # Remove short words and stopwords
    try:
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if len(word) >= min_length and word not in stop_words]
    except:
        # If NLTK resources aren't available
        filtered_words = [word for word in words if len(word) >= min_length]
    
    # Count and return top N
    word_counts = Counter(filtered_words)
    return word_counts.most_common(n)

# Get top words for each sentiment
plt.figure(figsize=(18, 12))

for i, sentiment in enumerate(['positive', 'neutral', 'negative']):
    top_words = get_top_words(df[df['sentiment'] == sentiment]['review_body'])
    
    plt.subplot(3, 1, i+1)
    words, counts = zip(*top_words)
    sns.barplot(x=list(counts), y=list(words))
    plt.title(f'Top Words in {sentiment.capitalize()} Reviews')
    plt.xlabel('Count')
    
plt.tight_layout()
plt.show()
```

## Text Preprocessing and Feature Engineering

```python
# Define text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    try:
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        return ' '.join(tokens)
    except:
        # If NLTK resources aren't available
        return text

# Apply preprocessing
df['processed_review'] = df['review_body'].apply(preprocess_text)
df['processed_title'] = df['review_title'].apply(preprocess_text)

# Feature engineering from text data
df['combined_text'] = df['processed_title'] + ' ' + df['processed_review']
df['contains_exclamation'] = df['review_body'].apply(lambda x: 1 if '!' in str(x) else 0)
df['contains_question'] = df['review_body'].apply(lambda x: 1 if '?' in str(x) else 0)
df['title_caps_ratio'] = df['review_title'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0)

# Generate additional features
df['caps_ratio'] = df['review_body'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0)
df['sentiment_score'] = df['stars'] - 3  # Simple sentiment score: -2 to +2

# Advanced n-gram features
df['bigrams'] = df['processed_review'].apply(lambda x: ' '.join([x[i:i+2] for i in range(len(x)-1)]) if len(x) > 1 else '')

# One-hot encode product categories (if there aren't too many)
if df['product_category'].nunique() < 20:
    category_dummies = pd.get_dummies(df['product_category'], prefix='category')
    df = pd.concat([df, category_dummies], axis=1)

print("\nFeatures created:")
print(f"- Text features: processed_review, processed_title, combined_text, bigrams")
print(f"- Numeric features: review_length, title_length, word_count, contains_exclamation, contains_question, title_caps_ratio, caps_ratio, sentiment_score")
print(f"- One-hot encoded categories: {df['product_category'].nunique()} categories")

# Preview processed data
print("\nProcessed data sample:")
print(df[['processed_review', 'sentiment']].head(3))
```

## Model Training and Evaluation

```python
# Prepare data for modeling
X = df['combined_text']  # Use combined preprocessed text
y = df['sentiment']  # Target variable

# Create train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create a text classification pipeline
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42)),
])

# Train the model
text_clf.fit(X_train, y_train)

# Make predictions
y_pred = text_clf.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=text_clf.classes_, 
            yticklabels=text_clf.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Get feature importance - top words for each class
tfidf = text_clf.named_steps['tfidf']
clf = text_clf.named_steps['clf']

# Get feature names
feature_names = tfidf.get_feature_names_out()

# For each sentiment, find the most important features
for i, sentiment in enumerate(text_clf.classes_):
    # For Random Forest, we use feature importances
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print(f"\nTop 10 features for {sentiment}:")
        for j in range(10):
            if j < len(indices):
                print(f"{feature_names[indices[j]]}: {importances[indices[j]]:.4f}")
```

## Advanced Feature Engineering for Text

```python
# Let's add more sophisticated text features
from sklearn.feature_extraction.text import CountVectorizer

# Create enhanced feature dataset
X_enhanced = df.copy()

# 1. TF-IDF vectorizer to get top N words for each review
def get_top_tfidf_words(corpus, n=5):
    tfidf = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf.fit_transform(corpus)
    feature_names = tfidf.get_feature_names_out()
    
    top_words = []
    for i in range(len(corpus)):
        if i < tfidf_matrix.shape[0]:
            feature_index = tfidf_matrix[i,:].nonzero()[1]
            tfidf_scores = zip(feature_index, [tfidf_matrix[i, x] for x in feature_index])
            top_word_indices = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:n]
            top_words.append(' '.join([feature_names[idx] for idx, score in top_word_indices]))
        else:
            top_words.append('')
            
    return top_words

# Add top TF-IDF words feature
X_enhanced['top_tfidf_words'] = get_top_tfidf_words(df['processed_review'])

# 2. Sentiment-specific features
positive_words = ['excellent', 'great', 'love', 'best', 'amazing', 'perfect', 'good', 'wonderful', 'fantastic', 'happy']
negative_words = ['bad', 'poor', 'terrible', 'worst', 'waste', 'disappointed', 'awful', 'horrible', 'hate', 'useless']

# Count occurrences of positive and negative words
def count_sentiment_words(text, word_list):
    if not isinstance(text, str):
        return 0
    text = text.lower()
    count = sum(1 for word in word_list if word in text.split())
    return count / max(len(text.split()), 1)  # Normalize by word count

X_enhanced['positive_word_ratio'] = df['review_body'].apply(lambda x: count_sentiment_words(str(x), positive_words))
X_enhanced['negative_word_ratio'] = df['review_body'].apply(lambda x: count_sentiment_words(str(x), negative_words))
X_enhanced['sentiment_polarity'] = X_enhanced['positive_word_ratio'] - X_enhanced['negative_word_ratio']

# 3. Additional text statistics
X_enhanced['avg_word_length'] = df['review_body'].apply(lambda x: np.mean([len(word) for word in str(x).split()]) if len(str(x).split()) > 0 else 0)
X_enhanced['sentence_count'] = df['review_body'].apply(lambda x: len(re.split(r'[.!?]+', str(x))))
X_enhanced['avg_sentence_length'] = X_enhanced['word_count'] / (X_enhanced['sentence_count'] + 1)  # Add 1 to avoid division by zero

# 4. Product-sentiment interaction features
if 'product_category' in X_enhanced.columns:
    product_sentiment = df.groupby('product_category')['stars'].mean().to_dict()
    X_enhanced['category_avg_sentiment'] = df['product_category'].map(product_sentiment)
    X_enhanced['rating_vs_category'] = df['stars'] - X_enhanced['category_avg_sentiment']

# Prepare enhanced data for modeling
features_to_drop = ['review_title', 'review_body', 'processed_review', 'processed_title', 
                    'combined_text', 'bigrams', 'sentiment', 'top_tfidf_words']
X_numeric = X_enhanced.drop(features_to_drop, axis=1, errors='ignore')

# Handle categorical features if any remain
categorical_cols = X_numeric.select_dtypes(include=['object', 'category']).columns
if len(categorical_cols) > 0:
    X_numeric = pd.get_dummies(X_numeric, columns=categorical_cols, drop_first=True)

# Convert to numeric (some columns might have mixed types)
for col in X_numeric.columns:
    X_numeric[col] = pd.to_numeric(X_numeric[col], errors='coerce')
X_numeric = X_numeric.fillna(0)

# Combine TF-IDF features and numeric features
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['combined_text'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
)
X_train_numeric, X_test_numeric, _, _ = train_test_split(
    X_numeric, df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
)

# Create a TF-IDF matrix for the training and test sets
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

# Convert sparse matrix to DataFrame
X_train_tfidf_df = pd.DataFrame.sparse.from_spmatrix(
    X_train_tfidf, 
    columns=tfidf.get_feature_names_out()
)
X_test_tfidf_df = pd.DataFrame.sparse.from_spmatrix(
    X_test_tfidf, 
    columns=tfidf.get_feature_names_out()
)

# Combine text features and numeric features
X_train_combined = pd.concat([X_train_tfidf_df.reset_index(drop=True), X_train_numeric.reset_index(drop=True)], axis=1)
X_test_combined = pd.concat([X_test_tfidf_df.reset_index(drop=True), X_test_numeric.reset_index(drop=True)], axis=1)

# Train model with combined features
from sklearn.ensemble import RandomForestClassifier

enhanced_model = RandomForestClassifier(n_estimators=100, random_state=42)
enhanced_model.fit(X_train_combined, y_train)

# Make predictions
y_pred_enhanced = enhanced_model.predict(X_test_combined)

# Evaluate enhanced model
print("\nEnhanced Model Evaluation:")
print(classification_report(y_test, y_pred_enhanced))

# Confusion matrix for enhanced model
cm_enhanced = confusion_matrix(y_test, y_pred_enhanced)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_enhanced, annot=True, fmt='d', cmap='Blues', 
            xticklabels=enhanced_model.classes_, 
            yticklabels=enhanced_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Enhanced Model')
plt.show()

# Compare feature importance
feature_importance = pd.DataFrame({
    'Feature': X_train_combined.columns,
    'Importance': enhanced_model.feature_importances_
}).sort_values('Importance', ascending=False)

# Plot top 20 features
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
plt.title('Top 20 Features by Importance')
plt.tight_layout()
plt.show()

print("\nTop 20 most important features:")
print(feature_importance.head(20))
```

## Advanced Analysis and Insights

```python
# Misclassification analysis
misclassified = X_test_text[y_test != y_pred_enhanced].reset_index(drop=True)
misclassified_labels = pd.DataFrame({
    'Actual': y_test[y_test != y_pred_enhanced].reset_index(drop=True),
    'Predicted': y_pred_enhanced[y_test != y_pred_enhanced]
})

print("\nMisclassification analysis:")
print(f"Total misclassified instances: {len(misclassified)} ({len(misclassified)/len(y_test):.2%} of test set)")

# Most common misclassification patterns
misclass_patterns = misclassified_labels.groupby(['Actual', 'Predicted']).size().reset_index(name='Count')
print("\nMost common misclassification patterns:")
print(misclass_patterns.sort_values('Count', ascending=False))

# Sample of misclassified instances
print("\nSample of misclassified instances:")
misclass_sample = pd.concat([
    misclassified.rename('Review Text'),
    misclassified_labels
], axis=1).sample(min(5, len(misclassified)))
print(misclass_sample)

# Word clouds by sentiment
try:
    from wordcloud import WordCloud
    
    plt.figure(figsize=(15, 15))
    
    for i, sentiment in enumerate(['positive', 'neutral', 'negative']):
        plt.subplot(3, 1, i+1)
        
        # Combine all text for this sentiment
        text = ' '.join(df[df['sentiment'] == sentiment]['processed_review'].dropna())
        
        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(text)
        
        # Display
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {sentiment.capitalize()} Reviews')
    
    plt.tight_layout()
    plt.show()
except:
    print("WordCloud package not available. Skipping word cloud visualization.")

# Cross-validation performance
from sklearn.model_selection import cross_val_score

# We'll use just the text-based model for simplicity in cross-validation
text_model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42)),
])

cv_scores = cross_val_score(text_model, X, y, cv=5, scoring='accuracy')
print(f"\nCross-validation accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
```

## Key Insights and Preprocessing Lessons

From this sentiment analysis case study, we've learned several important lessons:

1. **Text Preprocessing Importance**:
   - Cleaning text data (lowercasing, removing punctuation, etc.) significantly improves model performance
   - Removing stopwords and lemmatization helps focus on meaningful content
   - Feature extraction from text data is critical for sentiment analysis success

2. **Feature Engineering for Text**:
   - Combining title and review text provides more context for sentiment prediction
   - Engineered features like exclamation mark counts and capitalization ratios capture emotional intensity
   - TF-IDF vectorization effectively captures important words and n-grams
   - Product category context enhances sentiment prediction accuracy

3. **Model Selection Considerations**:
   - Random Forest works well for multi-class sentiment classification
   - The pipeline approach ensures consistent preprocessing between training and prediction
   - Combining text-based and engineered features yields the best performance

4. **Business Applications**:
   - Product categories show different sentiment profiles that can guide product development
   - Word importance analysis reveals specific aspects customers care about
   - Misclassification analysis helps identify ambiguous or mixed sentiment reviews

## Next Steps

For further improvement of the sentiment analysis system:

1. **Advanced Models**:
   - Implement deep learning models like BERT or RoBERTa for improved text understanding
   - Explore ensemble methods combining multiple model types

2. **Feature Enhancement**:
   - Add aspect-based sentiment analysis to identify sentiments about specific product features
   - Incorporate emoji and emoticon analysis for additional sentiment signals

3. **Additional Data Sources**:
   - Include product metadata for more context
   - Consider time-based analysis to track sentiment changes over product lifecycle

4. **Deployment Considerations**:
   - Create a real-time sentiment analysis pipeline for new reviews
   - Build dashboards to track sentiment trends across product categories
</rewritten_file> 