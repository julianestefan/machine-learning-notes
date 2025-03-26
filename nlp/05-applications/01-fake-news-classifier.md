# Fake News Classification with NLP

Fake news classification uses NLP techniques to automatically identify misleading or fabricated news articles. This application combines text preprocessing, feature engineering, and machine learning to distinguish between legitimate and false information.

## Problem Statement

Misinformation spreads rapidly online, making manual fact-checking insufficient. Automated systems can help by:

1. Flagging potentially false content for human review
2. Warning users about questionable sources
3. Reducing the spread of misinformation
4. Providing trust scores for news articles

## Dataset Preparation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load dataset (example using Kaggle's "Fake and Real News" dataset)
# https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset
fake = pd.read_csv('Fake.csv')
real = pd.read_csv('True.csv')

# Add labels
fake['label'] = 'FAKE'
real['label'] = 'REAL'

# Combine datasets
df = pd.concat([fake, real]).reset_index(drop=True)

# Basic exploration
print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution:\n{df['label'].value_counts()}")
print(f"\nColumns: {df.columns.tolist()}")

# Text length analysis
df['text_length'] = df['text'].apply(len)
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='text_length', hue='label', bins=50, kde=True)
plt.title('Text Length Distribution by Label')
plt.xlabel('Text Length (characters)')
plt.ylabel('Count')
plt.show()

# Create a sample dataframe with relevant columns
df = df[['text', 'title', 'label']]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df[['text', 'title']], df['label'], 
    test_size=0.2, random_state=42, stratify=df['label']
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")
```

## Feature Engineering

### Bag of Words (CountVectorizer)

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np

# Initialize a CountVectorizer
count_vectorizer = CountVectorizer(
    stop_words='english',
    max_features=5000,
    min_df=5,          # Minimum document frequency
    max_df=0.7,        # Maximum document frequency (as a proportion)
    ngram_range=(1, 2) # Include unigrams and bigrams
)

# Transform training data
count_train = count_vectorizer.fit_transform(X_train['text'])

# Transform testing data
count_test = count_vectorizer.transform(X_test['text'])

# Display some features
feature_names = count_vectorizer.get_feature_names_out()
print(f"Number of features: {len(feature_names)}")
print(f"Sample features: {feature_names[:20]}")

# View the first few vectors
count_df = pd.DataFrame(count_train.toarray()[:5, :20], columns=feature_names[:20])
print("\nSample vectors (first 5 documents, first 20 features):")
print(count_df)
```

### TF-IDF Vectorization

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    min_df=5,
    max_df=0.7,
    ngram_range=(1, 2),
    norm='l2'
)

# Transform training data
tfidf_train = tfidf_vectorizer.fit_transform(X_train['text'])

# Transform testing data
tfidf_test = tfidf_vectorizer.transform(X_test['text'])

# Compare the first document's representation in both models
doc_id = 0
count_vector = count_train.toarray()[doc_id]
tfidf_vector = tfidf_train.toarray()[doc_id]

# Get indices of top 10 features in each representation
count_top_indices = np.argsort(count_vector)[-10:]
tfidf_top_indices = np.argsort(tfidf_vector)[-10:]

# Get feature names
count_features = [feature_names[i] for i in count_top_indices]
tfidf_features = [tfidf_vectorizer.get_feature_names_out()[i] for i in tfidf_top_indices]

print("Top Count features:", count_features)
print("Top TF-IDF features:", tfidf_features)
```

### Feature Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Compare the CountVectorizer and TfidfVectorizer features
count_df = pd.DataFrame(count_train.toarray(), columns=count_vectorizer.get_feature_names_out())
tfidf_df = pd.DataFrame(tfidf_train.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Add labels
count_df['label'] = y_train.values
tfidf_df['label'] = y_train.values

# Function to find most common words by class
def get_top_n_words(corpus, labels, class_label, n=20):
    # Filter corpus by class
    class_corpus = corpus[corpus['label'] == class_label]
    
    # Drop the label column
    class_corpus = class_corpus.drop('label', axis=1)
    
    # Sum up feature frequencies
    class_sum = class_corpus.sum(axis=0)
    
    # Get top N words
    top_n_words = class_sum.nlargest(n)
    
    return top_n_words

# Get top words for fake and real news (Count)
fake_top_words = get_top_n_words(count_df, y_train, 'FAKE')
real_top_words = get_top_n_words(count_df, y_train, 'REAL')

# Plot comparison
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
fake_top_words.plot(kind='bar', color='red', alpha=0.7)
plt.title('Top Words in FAKE News')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')

plt.subplot(2, 1, 2)
real_top_words.plot(kind='bar', color='green', alpha=0.7)
plt.title('Top Words in REAL News')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()
```

## Building a Basic Classifier

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize classifiers
nb_classifier = MultinomialNB()
lr_classifier = LogisticRegression(max_iter=1000, C=1.0, solver='liblinear')

# Train Naive Bayes with Count Vectors
nb_classifier.fit(count_train, y_train)
nb_count_pred = nb_classifier.predict(count_test)

# Train Logistic Regression with TF-IDF
lr_classifier.fit(tfidf_train, y_train)
lr_tfidf_pred = lr_classifier.predict(tfidf_test)

# Evaluate Naive Bayes model
print("Naive Bayes with Count Vectors:")
print(f"Accuracy: {accuracy_score(y_test, nb_count_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, nb_count_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, nb_count_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['FAKE', 'REAL'], 
            yticklabels=['FAKE', 'REAL'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix: Naive Bayes with Count Vectors')
plt.show()

# Evaluate Logistic Regression model
print("\nLogistic Regression with TF-IDF:")
print(f"Accuracy: {accuracy_score(y_test, lr_tfidf_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, lr_tfidf_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, lr_tfidf_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['FAKE', 'REAL'], 
            yticklabels=['FAKE', 'REAL'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix: Logistic Regression with TF-IDF')
plt.show()
```

## Feature Importance Analysis

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Get feature names
feature_names = tfidf_vectorizer.get_feature_names_out()

# Get feature importance from Logistic Regression
if hasattr(lr_classifier, 'coef_'):
    # For binary classification, take the first (and only) class
    coefficients = lr_classifier.coef_[0]
    
    # Create a DataFrame with features and their coefficients
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })
    
    # Sort by absolute coefficient value
    coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
    coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
    
    # Plot top positive and negative coefficients
    plt.figure(figsize=(12, 8))
    
    # Top 20 features that indicate REAL news (positive coefficients)
    plt.subplot(2, 1, 1)
    coef_df.nlargest(20, 'Coefficient').set_index('Feature')['Coefficient'].plot(
        kind='bar', color='green', alpha=0.7
    )
    plt.title('Top 20 Features Indicating REAL News')
    plt.xlabel('Feature')
    plt.ylabel('Coefficient')
    plt.xticks(rotation=45, ha='right')
    
    # Top 20 features that indicate FAKE news (negative coefficients)
    plt.subplot(2, 1, 2)
    coef_df.nsmallest(20, 'Coefficient').set_index('Feature')['Coefficient'].plot(
        kind='bar', color='red', alpha=0.7
    )
    plt.title('Top 20 Features Indicating FAKE News')
    plt.xlabel('Feature')
    plt.ylabel('Coefficient')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
```

## Model Tuning with Grid Search

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Create a pipeline with TF-IDF and Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('classifier', LogisticRegression(solver='liblinear'))
])

# Define parameter grid
param_grid = {
    'tfidf__max_features': [3000, 5000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__min_df': [2, 5],
    'tfidf__max_df': [0.7, 0.9],
    'classifier__C': [0.1, 1.0, 10.0]
}

# Initialize grid search
grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=5, 
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

# Fit grid search
grid_search.fit(X_train['text'], y_train)

# Best parameters
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Evaluate on test set
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test['text'])
print("\nTest set accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))
```

## Advanced Model: Adding Title Features

```python
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer

# Define functions to extract columns
def get_text(X):
    return X['text']

def get_title(X):
    return X['title']

# Create feature union
features = FeatureUnion([
    ('text_features', Pipeline([
        ('selector', FunctionTransformer(get_text, validate=False)),
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2)))
    ])),
    ('title_features', Pipeline([
        ('selector', FunctionTransformer(get_title, validate=False)),
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=2000, ngram_range=(1, 2)))
    ]))
])

# Create pipeline
pipeline = Pipeline([
    ('features', features),
    ('classifier', LogisticRegression(solver='liblinear', C=1.0))
])

# Fit model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("Accuracy with text + title features:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

## Adding Linguistic Features

```python
import spacy
import pandas as pd
import numpy as np

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def extract_linguistic_features(texts, max_texts=None):
    """Extract linguistic features from texts using spaCy"""
    if max_texts:
        texts = texts[:max_texts]
        
    features = []
    
    for text in texts:
        # Process text with spaCy
        doc = nlp(text[:10000])  # Limit text length to avoid memory issues
        
        # Extract features
        feature_dict = {
            'num_tokens': len(doc),
            'num_sentences': len(list(doc.sents)),
            'avg_token_length': np.mean([len(token.text) for token in doc]) if len(doc) > 0 else 0,
            'num_entities': len(doc.ents),
            'num_person': len([ent for ent in doc.ents if ent.label_ == 'PERSON']),
            'num_org': len([ent for ent in doc.ents if ent.label_ == 'ORG']),
            'num_gpe': len([ent for ent in doc.ents if ent.label_ == 'GPE']),
            'num_dates': len([ent for ent in doc.ents if ent.label_ == 'DATE']),
            'num_verbs': len([token for token in doc if token.pos_ == 'VERB']),
            'num_nouns': len([token for token in doc if token.pos_ == 'NOUN']),
            'num_adjs': len([token for token in doc if token.pos_ == 'ADJ']),
            'num_advs': len([token for token in doc if token.pos_ == 'ADV']),
            'num_exclamations': text.count('!'),
            'num_questions': text.count('?'),
            'avg_sentence_length': np.mean([len(sent) for sent in doc.sents]) if len(list(doc.sents)) > 0 else 0
        }
        
        features.append(feature_dict)
        
    return pd.DataFrame(features)

# Extract features for a sample of documents
sample_size = 1000  # Adjust based on your computational resources
X_train_sample = X_train['text'].iloc[:sample_size]
X_test_sample = X_test['text'].iloc[:sample_size]

# Extract linguistic features
print("Extracting linguistic features from training data...")
train_ling_features = extract_linguistic_features(X_train_sample)
print("Extracting linguistic features from test data...")
test_ling_features = extract_linguistic_features(X_test_sample)

# Add label
train_ling_features['label'] = y_train.iloc[:sample_size]

# Analyze feature differences between fake and real news
fake_features = train_ling_features[train_ling_features['label'] == 'FAKE']
real_features = train_ling_features[train_ling_features['label'] == 'REAL']

# Calculate mean values
fake_means = fake_features.mean()
real_means = real_features.mean()

# Compare features
feature_comparison = pd.DataFrame({
    'FAKE': fake_means,
    'REAL': real_means,
    'Difference': real_means - fake_means
})

print("\nLinguistic Feature Comparison:")
print(feature_comparison.sort_values('Difference', ascending=False))

# Visualize some interesting features
features_to_plot = ['num_entities', 'num_person', 'num_org', 'num_exclamations', 'avg_sentence_length']

plt.figure(figsize=(14, 10))
for i, feature in enumerate(features_to_plot):
    plt.subplot(len(features_to_plot), 1, i+1)
    sns.boxplot(x='label', y=feature, data=train_ling_features)
    plt.title(f'Distribution of {feature} by News Type')
    plt.tight_layout()

plt.show()
```

## Building a Model with Text and Linguistic Features

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer to extract linguistic features
class LinguisticFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, max_texts=None):
        self.max_texts = max_texts
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return extract_linguistic_features(X, self.max_texts).values

# Create pipeline for text features
text_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        max_features=3000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.7
    ))
])

# Create pipeline for linguistic features
ling_pipeline = Pipeline([
    ('extractor', LinguisticFeatureExtractor(max_texts=1000)),
    ('scaler', StandardScaler())
])

# Combine features
features = FeatureUnion([
    ('text_features', text_pipeline),
    ('linguistic_features', ling_pipeline)
])

# Create full pipeline
pipeline = Pipeline([
    ('features', features),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Fit model on sample
print("Training model with combined features...")
pipeline.fit(X_train_sample, y_train.iloc[:sample_size])

# Evaluate
y_pred = pipeline.predict(X_test_sample)
print("Accuracy with text + linguistic features:", accuracy_score(y_test.iloc[:sample_size], y_pred))
print("\nClassification Report:")
print(classification_report(y_test.iloc[:sample_size], y_pred))
```

## Model Deployment

```python
import joblib

# Save the best model
joblib.dump(best_model, 'fake_news_classifier.pkl')

# Function to preprocess and classify a news article
def classify_news(title, text, model_path='fake_news_classifier.pkl'):
    # Load model
    model = joblib.load(model_path)
    
    # Create DataFrame with the article
    article_df = pd.DataFrame({
        'title': [title],
        'text': [text]
    })
    
    # Make prediction
    prediction = model.predict(article_df)
    probability = model.predict_proba(article_df)
    
    # Get confidence score for the predicted class
    confidence = probability[0][0] if prediction[0] == 'FAKE' else probability[0][1]
    
    return {
        'prediction': prediction[0],
        'confidence': confidence,
        'probability_fake': probability[0][0],
        'probability_real': probability[0][1]
    }

# Example usage
sample_title = "Scientists Make Breakthrough Discovery"
sample_text = """Researchers at the University of Science announced today 
that they have discovered a new treatment for cancer that shows promising 
results in early trials. The study, published in the Journal of Medicine, 
describes how the treatment targets specific cancer cells while leaving 
healthy cells intact. Clinical trials are expected to begin next year."""

result = classify_news(sample_title, sample_text)
print("Classification result:")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Probability Fake: {result['probability_fake']:.2f}")
print(f"Probability Real: {result['probability_real']:.2f}")
```

## Challenges in Fake News Detection

1. **Evolving misinformation tactics**: As detection systems improve, fake news creators adapt their techniques
2. **Domain-specific language**: News articles from different domains (politics, science, etc.) have different linguistic patterns
3. **Satire vs. misinformation**: Distinguishing between intentional satire and fake news
4. **Bias in training data**: Models may learn the biases present in the data
5. **Limited contextual understanding**: NLP models may miss broader context needed for fact-checking
6. **Multimodal misinformation**: Fake news often combines text, images, and videos

## Future Improvements

1. **Incorporate external knowledge**: Use fact-checking databases and knowledge graphs
2. **Source credibility features**: Include information about the source's reputation
3. **Temporal patterns**: Analyze how stories evolve over time
4. **Deep learning architectures**: Use transformer models like BERT for better language understanding
5. **Multimodal analysis**: Combine text analysis with image and video verification
6. **Cross-lingual models**: Detect fake news across multiple languages 