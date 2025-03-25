# Text Feature Engineering

Text data requires specialized feature engineering techniques to convert unstructured text into meaningful numerical representations that machine learning models can process.

## Text Preprocessing Pipeline

A comprehensive text preprocessing pipeline typically includes these steps:

```python
import re
import string
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text, 
                    lowercase=True, 
                    remove_urls=True,
                    remove_html=True,
                    remove_punctuation=True,
                    remove_numbers=True,
                    remove_stopwords=True,
                    stemming=False,
                    lemmatization=True):
    """
    Complete text preprocessing pipeline
    
    Args:
        text (str): Raw text input
        lowercase (bool): Convert to lowercase
        remove_urls (bool): Remove URLs
        remove_html (bool): Remove HTML tags
        remove_punctuation (bool): Remove punctuation
        remove_numbers (bool): Remove numbers
        remove_stopwords (bool): Remove common stopwords
        stemming (bool): Apply stemming
        lemmatization (bool): Apply lemmatization
        
    Returns:
        list: Preprocessed tokens
    """
    if not isinstance(text, str):
        return []
    
    # Convert to lowercase
    if lowercase:
        text = text.lower()
    
    # Remove URLs
    if remove_urls:
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    if remove_html:
        text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    
    # Apply stemming
    if stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
    
    # Apply lemmatization
    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Remove short words (optional)
    tokens = [word for word in tokens if len(word) > 2]
    
    return tokens

# Example usage
sample_text = "This is an example text with some URLs https://example.com and HTML <div>tags</div> and numbers 12345!"
processed_tokens = preprocess_text(sample_text)
print(processed_tokens)
```

### Understanding Text Preprocessing Steps

1. **Lowercasing**: Standardizes text to avoid treating identical words as different

2. **URL and HTML Removal**: Removes web-specific elements that add noise

3. **Punctuation Removal**: Removes punctuation marks that typically don't contribute to meaning

4. **Number Removal**: Removes numeric characters unless they're significant for the domain

5. **Tokenization**: Breaks text into individual tokens (usually words)

6. **Stopword Removal**: Eliminates common words like "the", "is", "and" that add little meaning

7. **Stemming**: Reduces words to their root form (e.g., "running" → "run") but can be aggressive

8. **Lemmatization**: More sophisticated form of stemming that ensures the root is a valid word

## Basic Text Statistics Features

Extract simple numerical features from text:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load example text data
df = pd.DataFrame({
    'text': ["This is a short text.", 
             "This text is a bit longer and contains more information.",
             "This is a very long text with many words and might contain complex sentences with sophisticated vocabulary."]
})

# Text length features
df['char_count'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['sentence_count'] = df['text'].str.count(r'[.!?]+')
df['avg_word_length'] = df['text'].apply(lambda x: np.mean([len(word) for word in x.split()]))
df['avg_sentence_length'] = df['word_count'] / df['sentence_count'].replace(0, 1)

# Lexical diversity (unique words / total words)
df['unique_word_ratio'] = df['text'].apply(
    lambda x: len(set(x.lower().split())) / len(x.split()) if len(x.split()) > 0 else 0
)

# Character-level features
df['uppercase_char_ratio'] = df['text'].apply(
    lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
)
df['digit_char_ratio'] = df['text'].apply(
    lambda x: sum(1 for c in x if c.isdigit()) / len(x) if len(x) > 0 else 0
)
df['special_char_ratio'] = df['text'].apply(
    lambda x: sum(1 for c in x if not c.isalnum() and not c.isspace()) / len(x) if len(x) > 0 else 0
)

# Display the features
print(df[['char_count', 'word_count', 'sentence_count', 'avg_word_length', 
          'unique_word_ratio', 'uppercase_char_ratio']])

# Visualize the statistics
plt.figure(figsize=(12, 8))
for i, col in enumerate(['char_count', 'word_count', 'avg_word_length', 'unique_word_ratio']):
    plt.subplot(2, 2, i+1)
    sns.barplot(x=df.index, y=col, data=df)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()
```

## Text Vectorization Techniques

### 1. Bag of Words (CountVectorizer)

Represents text as word frequency counts:

```python
from sklearn.feature_extraction.text import CountVectorizer

# Sample corpus
corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third document.",
    "Is this the first document?",
]

# Create CountVectorizer with parameters
vectorizer = CountVectorizer(
    min_df=1,           # Minimum document frequency (ignore terms that appear in fewer than min_df documents)
    max_df=0.9,         # Maximum document frequency (ignore terms that appear in more than max_df documents)
    max_features=None,  # Maximum number of features (None means all)
    stop_words='english', # Remove English stop words
    ngram_range=(1, 1), # Range of n-grams (1, 1) means unigrams only
    lowercase=True,     # Convert to lowercase
    strip_accents='unicode', # Remove accents
    token_pattern=r'\b\w+\b' # Pattern for tokenizing
)

# Fit and transform the corpus
X = vectorizer.fit_transform(corpus)

# Get feature names
feature_names = vectorizer.get_feature_names_out()
print("Feature names:", feature_names)

# Convert to array for viewing
print("\nDocument-Term Matrix:")
print(X.toarray())

# Create DataFrame from the document-term matrix
bow_df = pd.DataFrame(X.toarray(), columns=feature_names)
print("\nBag of Words DataFrame:")
print(bow_df)
```

#### Advantages and Limitations of Bag of Words

**Advantages**:
- Simple and intuitive
- Fast to compute
- Works with any model type
- Preserves word specificity

**Limitations**:
- Loses word order and context
- Can result in very high dimensional, sparse matrices
- Doesn't capture semantics or relations between words
- Doesn't handle words not seen during training

### 2. TF-IDF (Term Frequency-Inverse Document Frequency)

Weights terms by their importance across the corpus:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(
    min_df=1, 
    max_df=0.9,
    max_features=None,
    stop_words='english',
    ngram_range=(1, 1),
    norm='l2'  # Apply L2 normalization to the vectors
)

# Fit and transform the corpus
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# Get feature names
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

# Convert to DataFrame for viewing
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf_feature_names
)

print("TF-IDF Matrix:")
print(tfidf_df)

# Visualize important terms for a document
def plot_important_terms(document_idx, vectorizer, feature_names, X, n_top_terms=5):
    """Plot most important terms for a specific document"""
    if document_idx >= X.shape[0]:
        print(f"Document index {document_idx} out of range.")
        return
    
    # Get document vector
    doc_vector = X[document_idx].toarray().flatten()
    
    # Get term importance
    term_importance = [(feature_names[i], doc_vector[i]) for i in range(len(doc_vector))]
    term_importance.sort(key=lambda x: x[1], reverse=True)
    
    # Plot top terms
    plt.figure(figsize=(10, 5))
    terms, values = zip(*term_importance[:n_top_terms])
    plt.barh(terms, values)
    plt.title(f'Top {n_top_terms} TF-IDF Terms in Document {document_idx}')
    plt.xlabel('TF-IDF Score')
    plt.tight_layout()
    plt.show()

# Plot top terms for the first document
plot_important_terms(0, tfidf_vectorizer, tfidf_feature_names, tfidf_matrix)
```

### 3. N-grams

Capture sequences of words to preserve some context:

```python
# Create vectorizer with different n-gram ranges
unigram_vectorizer = CountVectorizer(ngram_range=(1, 1))
bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
trigram_vectorizer = CountVectorizer(ngram_range=(3, 3))
combined_vectorizer = CountVectorizer(ngram_range=(1, 3))  # Unigrams, bigrams, and trigrams

# Transform corpus
unigram_matrix = unigram_vectorizer.fit_transform(corpus)
bigram_matrix = bigram_vectorizer.fit_transform(corpus)
trigram_matrix = trigram_vectorizer.fit_transform(corpus)
combined_matrix = combined_vectorizer.fit_transform(corpus)

# Compare feature counts
print(f"Unigram features: {len(unigram_vectorizer.get_feature_names_out())}")
print(f"Bigram features: {len(bigram_vectorizer.get_feature_names_out())}")
print(f"Trigram features: {len(trigram_vectorizer.get_feature_names_out())}")
print(f"Combined features: {len(combined_vectorizer.get_feature_names_out())}")

# Display bigram features
print("\nBigram features:", bigram_vectorizer.get_feature_names_out())

# Create DataFrame with combined n-grams
combined_df = pd.DataFrame(
    combined_matrix.toarray(),
    columns=combined_vectorizer.get_feature_names_out()
)
print("\nSample of combined n-gram features:")
print(combined_df.iloc[:, :10])  # Show first 10 columns
```

### 4. Word Embeddings

Use pre-trained word vectors like Word2Vec, GloVe, or FastText:

```python
# Install gensim if needed: pip install gensim
import gensim.downloader
import numpy as np

# Load pre-trained Word2Vec embeddings
word2vec_model = gensim.downloader.load('word2vec-google-news-300')

# Function to create document vectors by averaging word vectors
def document_vector(doc, model, vector_size=300):
    """Create a document vector by averaging word vectors"""
    # Initialize empty word vector
    doc_vector = np.zeros(vector_size)
    
    # Tokenize and preprocess
    words = preprocess_text(doc)
    
    # Count words with vectors in the model
    word_count = 0
    
    # Sum vectors for all words in the document
    for word in words:
        if word in model:
            doc_vector += model[word]
            word_count += 1
    
    # Average the summed vectors
    if word_count > 0:
        doc_vector /= word_count
        
    return doc_vector

# Apply to corpus
document_vectors = np.array([document_vector(doc, word2vec_model) for doc in corpus])

# Create DataFrame of document vectors
doc_vec_df = pd.DataFrame(
    document_vectors,
    columns=[f'dim_{i}' for i in range(300)]
)
print("Document vectors (first 5 dimensions):")
print(doc_vec_df.iloc[:, :5])

# Calculate similarity between documents
from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(document_vectors)
print("\nDocument similarity matrix:")
print(similarity_matrix)
```

### 5. Topic Modeling with LDA

Identify latent topics in a text corpus:

```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Create a larger corpus for demonstration
larger_corpus = [
    "The sky is blue and clouds are white.",
    "Machine learning models need data to train.",
    "Neural networks have revolutionized AI.",
    "The weather is sunny with clear skies today.",
    "Deep learning is a subset of machine learning.",
    "Clouds bring rain and thunder during storms.",
    "Computers can learn from historical data examples.",
    "The atmosphere has different layers including troposphere."
]

# Vectorize the corpus
count_vec = CountVectorizer(
    max_df=0.9, 
    min_df=1, 
    stop_words='english'
)
doc_term_matrix = count_vec.fit_transform(larger_corpus)

# Create LDA model
n_topics = 2
lda_model = LatentDirichletAllocation(
    n_components=n_topics,
    max_iter=10,
    random_state=42
)

# Fit LDA model
lda_model.fit(doc_term_matrix)

# Get feature names
feature_names = count_vec.get_feature_names_out()

# Print topics with top words
def print_topics(model, feature_names, top_n=10):
    for topic_idx, topic in enumerate(model.components_):
        top_words_idx = topic.argsort()[:-top_n-1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        print(f"Topic #{topic_idx}: {' '.join(top_words)}")

print("LDA Topics:")
print_topics(lda_model, feature_names)

# Transform documents to topic space
doc_topics = lda_model.transform(doc_term_matrix)
print("\nDocument topic distributions:")
for i, doc_topic in enumerate(doc_topics):
    print(f"Document {i}: {doc_topic}")
```

## Sentiment Analysis Features

Extract sentiment from text:

```python
# For simple sentiment analysis, we can use TextBlob
# Install if needed: pip install textblob
from textblob import TextBlob

def extract_sentiment_features(text):
    """Extract sentiment features from text"""
    blob = TextBlob(text)
    
    return {
        'polarity': blob.sentiment.polarity,  # Range: -1 (negative) to 1 (positive)
        'subjectivity': blob.sentiment.subjectivity,  # Range: 0 (objective) to 1 (subjective)
        'is_positive': 1 if blob.sentiment.polarity > 0 else 0,
        'is_negative': 1 if blob.sentiment.polarity < 0 else 0,
        'is_neutral': 1 if blob.sentiment.polarity == 0 else 0,
    }

# Apply to corpus
sentiment_features = [extract_sentiment_features(doc) for doc in larger_corpus]
sentiment_df = pd.DataFrame(sentiment_features)
print("Sentiment features:")
print(sentiment_df)

# Visualize sentiment distribution
plt.figure(figsize=(10, 6))
plt.scatter(sentiment_df['polarity'], sentiment_df['subjectivity'])
for i, txt in enumerate(larger_corpus):
    plt.annotate(f"Doc {i}", (sentiment_df['polarity'][i], sentiment_df['subjectivity'][i]))
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.title('Sentiment Distribution')
plt.grid(True)
plt.show()
```

## Named Entity Recognition (NER) Features

Extract named entities as features:

```python
# Install if needed: pip install spacy
# Also run: python -m spacy download en_core_web_sm
import spacy

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def extract_entity_features(text):
    """Extract named entity features from text"""
    doc = nlp(text)
    
    # Count entities by type
    entity_counts = {}
    for ent in doc.ents:
        entity_type = ent.label_
        entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
    
    # Calculate total entities
    total_entities = sum(entity_counts.values())
    
    # Add total count
    entity_counts['TOTAL'] = total_entities
    
    # Add entity density feature
    entity_counts['ENTITY_DENSITY'] = total_entities / len(doc) if len(doc) > 0 else 0
    
    return entity_counts

# Apply to corpus
entity_features = [extract_entity_features(doc) for doc in larger_corpus]
entity_df = pd.DataFrame(entity_features).fillna(0)
print("Entity features:")
print(entity_df)

# Function to extract entity text
def extract_entity_text(text):
    """Extract entity text from document"""
    doc = nlp(text)
    entities = {ent.label_: [] for ent in doc.ents}
    
    for ent in doc.ents:
        entities[ent.label_].append(ent.text)
        
    return entities

# Print entities from a document
sample_doc = "Apple Inc. was founded by Steve Jobs in California on April 1, 1976."
print("\nNamed entities in sample document:")
print(extract_entity_text(sample_doc))
```

## Feature Selection for Text

Reducing dimensionality for text features:

```python
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a sample classification problem
texts = [
    "This movie is amazing and I loved it",
    "Great film with excellent acting",
    "Wonderful cinema experience, highly recommend",
    "Terrible movie, complete waste of time",
    "I hated this film, very disappointing",
    "Awful acting and poor direction"
]
labels = [1, 1, 1, 0, 0, 0]  # 1 = positive, 0 = negative

# Vectorize the text
tfidf_vec = TfidfVectorizer(min_df=1, max_df=0.9, stop_words='english')
X_tfidf = tfidf_vec.fit_transform(texts)

# Apply chi-squared feature selection
k = 5  # Select top k features
selector = SelectKBest(chi2, k=k)
X_selected = selector.fit_transform(X_tfidf, labels)

# Get selected feature names
feature_names = tfidf_vec.get_feature_names_out()
selected_indices = selector.get_support(indices=True)
selected_features = [feature_names[i] for i in selected_indices]

print(f"Selected features: {selected_features}")

# Get chi2 scores for all features
scores = zip(feature_names, selector.scores_)
sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

print("\nFeature importance (chi2):")
for feature, score in sorted_scores[:10]:  # Print top 10
    print(f"{feature}: {score:.4f}")
```

## Model-Ready Text Features

Putting it all together in a pipeline for a model:

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Create a sample text classification dataset
texts = larger_corpus + [
    "Data science combines statistics and programming.",
    "Cloud formations can predict weather patterns.",
    "Programming is essential for machine learning."
]
# Create binary labels: 1 if related to technology, 0 otherwise
labels = [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42
)

# Create a pipeline with text preprocessing and classification
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(
        max_df=0.9,
        min_df=1,
        stop_words='english',
        ngram_range=(1, 2)
    )),
    ('classifier', RandomForestClassifier(
        n_estimators=100, 
        random_state=42
    ))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate on test set
accuracy = pipeline.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.4f}")

# Predict on new text
new_texts = [
    "The sky is beautiful today with fluffy clouds.",
    "Artificial intelligence is transforming industries."
]
predictions = pipeline.predict(new_texts)
print("\nPredictions for new texts:")
for text, pred in zip(new_texts, predictions):
    print(f"Text: {text}")
    print(f"Prediction: {'Technology' if pred == 1 else 'Nature'}")
    print()
```

## Text Feature Engineering Best Practices

1. **Start simple**: Begin with basic bag-of-words or TF-IDF before more complex representations

2. **Clean thoroughly**: Text preprocessing significantly impacts feature quality

3. **Experiment with n-grams**: Try different n-gram ranges to capture phrases

4. **Balance vocabulary size**: Too small misses information, too large adds noise

5. **Consider domain-specific stopwords**: Standard stopword lists may not be appropriate for all domains

6. **Use dimensionality reduction**: Most text features are sparse and high-dimensional

7. **Combine representations**: Mix different feature types (lexical, semantic, sentiment) for better performance

8. **Cross-validate hyperparameters**: Test different vectorization settings to find optimal parameters

9. **Watch for data leakage**: Fit vectorizers only on training data

10. **Consider the downstream model**: Different models work better with different text representations 