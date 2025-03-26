# Topic Modeling

Topic modeling is an unsupervised machine learning technique that discovers abstract "topics" in a collection of documents. It identifies patterns of word co-occurrences and clusters related words into themes.

## Applications of Topic Modeling

- **Content Organization**: Organizing large document collections
- **Content Recommendation**: Finding similar articles or documents
- **Trend Analysis**: Tracking how topics evolve over time
- **Customer Feedback Analysis**: Categorizing reviews and feedback
- **Academic Research**: Analyzing research papers and publications
- **Content Summarization**: Providing thematic summaries of documents

## Latent Dirichlet Allocation (LDA)

LDA is the most popular topic modeling algorithm. It assumes that:

1. Each document is a mixture of topics
2. Each topic is a mixture of words

The goal is to discover the hidden (latent) topic structure.

### How LDA Works

1. Randomly assign each word in each document to one of K topics
2. For each document, calculate topic proportions and update assignments
3. For each topic, calculate word probabilities and update assignments
4. Repeat steps 2-3 until convergence

### Implementing LDA with Gensim

```python
import gensim
from gensim import corpora
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

# Example corpus
documents = [
    "Machine learning algorithms automatically improve through experience",
    "Deep learning is a subset of machine learning using neural networks",
    "Natural language processing helps computers understand human language",
    "Computer vision enables machines to interpret visual information",
    "Neural networks are inspired by the human brain's structure",
    "Supervised learning uses labeled data for training algorithms",
    "Unsupervised learning identifies patterns in unlabeled data",
    "Reinforcement learning trains agents through rewards and penalties",
    "NLP applications include translation, sentiment analysis, and chatbots",
    "Image recognition is a key application of computer vision technology"
]

# Preprocessing
def preprocess(docs):
    # Tokenize and lowercase
    texts = [[word.lower() for word in doc.split()] for doc in docs]
    
    # Create a dictionary
    dictionary = corpora.Dictionary(texts)
    
    # Create document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    return texts, dictionary, corpus

# Preprocess the documents
processed_texts, dictionary, corpus = preprocess(documents)

# Train LDA model
lda_model = gensim.models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=3,  # Number of topics
    random_state=42,
    passes=10       # Number of passes through corpus
)

# Print topics
print("LDA Topics:")
pprint(lda_model.print_topics())

# Visualize topics
from gensim.models import CoherenceModel

# Compute coherence score
coherence_model = CoherenceModel(
    model=lda_model, 
    texts=processed_texts, 
    dictionary=dictionary, 
    coherence='c_v'
)
coherence_score = coherence_model.get_coherence()
print(f"Coherence Score: {coherence_score:.4f}")

# Document-topic distribution
def format_topics_sentences(ldamodel, corpus):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: x[1], reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series(
                        [int(topic_num), round(prop_topic, 4), topic_keywords]
                    ), 
                    ignore_index=True
                )
            else:
                break
    
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    return sent_topics_df

df_topic_sents_keywords = format_topics_sentences(lda_model, corpus)

# Add original text
df_topic_sents_keywords['Text'] = documents

# Print document-topic distribution
print("\nDocument Topic Distribution:")
print(df_topic_sents_keywords[['Dominant_Topic', 'Perc_Contribution', 'Text']])
```

### Visualizing LDA Topics with pyLDAvis

```python
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# Visualize the topics
lda_display = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(lda_display)
```

### Finding Optimal Number of Topics

```python
def compute_coherence_values(dictionary, corpus, texts, start=2, limit=10, step=1):
    """
    Compute coherence values for different numbers of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        # Train model
        model = gensim.models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=10
        )
        model_list.append(model)
        
        # Compute coherence
        coherence_model = CoherenceModel(
            model=model, 
            texts=texts, 
            dictionary=dictionary, 
            coherence='c_v'
        )
        coherence_values.append(coherence_model.get_coherence())
    
    return model_list, coherence_values

# Compute coherence for different numbers of topics
model_list, coherence_values = compute_coherence_values(
    dictionary=dictionary, 
    corpus=corpus, 
    texts=processed_texts,
    start=2, 
    limit=11, 
    step=1
)

# Plot coherence scores
limit=11; start=2; step=1;
x = range(start, limit, step)
plt.figure(figsize=(12, 6))
plt.plot(x, coherence_values)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence Score")
plt.title("Coherence Score by Number of Topics")
plt.xticks(x)
plt.grid(True)
plt.show()

# Find the optimal number of topics
optimal_model_index = coherence_values.index(max(coherence_values))
optimal_model = model_list[optimal_model_index]
optimal_topics = start + step * optimal_model_index
print(f"Optimal number of topics: {optimal_topics}")
```

## Non-Negative Matrix Factorization (NMF)

NMF is another popular topic modeling algorithm that works well with TF-IDF features. Unlike LDA, NMF is a linear algebra approach that decomposes the document-term matrix.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_df=0.95, 
    min_df=2, 
    stop_words='english'
)

# Generate TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

# Train NMF model
num_topics = 3
nmf_model = NMF(
    n_components=num_topics, 
    random_state=42,
    alpha=.1, 
    l1_ratio=.5
)
nmf_model.fit(tfidf_matrix)

# Display topics
def display_topics(model, feature_names, no_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict[topic_idx] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict).T

# Display top 10 words per topic
top_words = 10
topics_df = display_topics(nmf_model, feature_names, top_words)
topics_df.columns = [f'Word {i}' for i in range(1, top_words + 1)]
print("\nNMF Topics:")
print(topics_df)

# Document-topic distribution
doc_topic_dist = nmf_model.transform(tfidf_matrix)
doc_topic_df = pd.DataFrame(doc_topic_dist, columns=[f'Topic {i+1}' for i in range(num_topics)])
doc_topic_df['Dominant Topic'] = doc_topic_df.idxmax(axis=1)
doc_topic_df['Document'] = documents
print("\nDocument Topic Distribution:")
print(doc_topic_df[['Document', 'Dominant Topic']])
```

## BERTopic: Modern Topic Modeling with Transformers

BERTopic combines BERT embeddings with clustering and TF-IDF to create a more semantic approach to topic modeling.

```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

# Get some example data
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
docs = newsgroups.data[:1000]  # Limit to 1000 docs for speed

# Create and fit BERTopic model
topic_model = BERTopic()
topics, probabilities = topic_model.fit_transform(docs)

# Get most frequent topics
freq = topic_model.get_topic_info()
print("Most frequent topics:")
print(freq.head(10))

# Show top terms for a specific topic
print("\nTop terms for Topic 1:")
print(topic_model.get_topic(1))

# Visualize topics
topic_model.visualize_topics()

# Visualize topic hierarchy
topic_model.visualize_hierarchy()

# Visualize topic similarity
topic_model.visualize_heatmap()

# Visualize a specific topic's terms as a bar chart
topic_model.visualize_barchart(top_n_topics=5)
```

## Topic Model Evaluation

### Coherence Score

Measures the semantic similarity between high-scoring words in a topic.

```python
from gensim.models import CoherenceModel

# Calculate coherence using 'c_v' (based on sliding window, segment size 110 words, cosine similarity)
coherence_model_cv = CoherenceModel(
    model=lda_model, 
    texts=processed_texts, 
    dictionary=dictionary, 
    coherence='c_v'
)
coherence_cv = coherence_model_cv.get_coherence()
print(f"Coherence Score (c_v): {coherence_cv}")

# Calculate coherence using 'u_mass' (based on document co-occurrence)
coherence_model_umass = CoherenceModel(
    model=lda_model, 
    corpus=corpus,
    dictionary=dictionary, 
    coherence='u_mass'
)
coherence_umass = coherence_model_umass.get_coherence()
print(f"Coherence Score (u_mass): {coherence_umass}")
```

### Perplexity

Measures how well a probability model predicts a sample.

```python
# Compute perplexity
perplexity = lda_model.log_perplexity(corpus)
print(f"Perplexity: {perplexity}")
```

### Human Evaluation

Manual evaluation by domain experts is often the most reliable approach.

```python
# Example of gathering human evaluations
topic_ratings = {
    "Topic Coherence": [
        "Does the topic contain words that make sense together?",
        "Rate from 1 (unrelated words) to 5 (highly coherent words)"
    ],
    "Topic Diversity": [
        "How distinct are the topics from each other?",
        "Rate from 1 (very similar) to 5 (completely distinct)"
    ],
    "Topic Interpretability": [
        "How easy is it to assign a label to each topic?",
        "Rate from 1 (impossible to label) to 5 (obvious label)"
    ]
}

for criteria, [question, scale] in topic_ratings.items():
    print(f"\n{criteria}: {question}")
    print(f"Scale: {scale}")
```

## Dynamic Topic Modeling

Dynamic Topic Modeling tracks how topics evolve over time.

```python
import gensim
from gensim.models import LdaSeqModel
from gensim.corpora import Dictionary
import numpy as np
import matplotlib.pyplot as plt

# Example documents with time information
docs_by_time = [
    # Year 1 documents
    ["ai future robots automation learning"],
    ["machine learning algorithms data prediction"],
    ["natural language processing text understanding"],
    
    # Year 2 documents
    ["deep learning neural networks training models"],
    ["convolutional networks image recognition computer vision"],
    ["recurrent networks sequential data prediction"],
    
    # Year 3 documents
    ["transformer models attention self-supervised learning"],
    ["bert gpt language models pretraining finetuning"],
    ["multimodal models vision language fusion"],
]

# Preprocess: tokenize documents and create dictionary
tokenized_docs = [[word for word in doc[0].split()] for doc in docs_by_time]
dictionary = Dictionary(tokenized_docs)

# Create corpus
corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

# Organize corpus into time slices
time_slices = [3, 3, 3]  # 3 documents for each year

# Train DTM model
dtm_model = LdaSeqModel(
    corpus=corpus,
    id2word=dictionary,
    time_slices=time_slices,
    num_topics=2,
    chunksize=1
)

# Visualize topic evolution
for topic_id in range(2):
    print(f"\nTopic {topic_id} evolution:")
    for t in range(3):  # 3 time periods
        print(f"Time {t}:")
        topic_terms = dtm_model.print_topic_times(topic=topic_id, time=t)
        print(topic_terms)
```

## Topic Modeling for Document Summarization

```python
def extract_topic_sentences(df, topic_id):
    """Extract sentences most representative of a topic"""
    # Filter documents with the specific topic
    topic_docs = df[df['Dominant_Topic'] == topic_id]
    
    # Sort by contribution percentage
    topic_docs = topic_docs.sort_values('Perc_Contribution', ascending=False)
    
    # Return top 3 representative documents
    return topic_docs.head(3)['Text'].tolist()

# Generate summaries for each topic
for topic_id in range(3):
    print(f"\nTopic {topic_id} Summary:")
    representative_docs = extract_topic_sentences(df_topic_sents_keywords, topic_id)
    for i, doc in enumerate(representative_docs):
        print(f"{i+1}. {doc}")
```

## Best Practices for Topic Modeling

1. **Data Preparation**:
   - Remove stopwords, punctuation, and special characters
   - Consider stemming or lemmatization
   - Remove very frequent and very rare words
   - Try different preprocessing strategies and evaluate their impact

2. **Model Selection**:
   - LDA: General-purpose topic modeling with probabilistic foundations
   - NMF: Works well with TF-IDF and tends to produce more coherent topics
   - BERTopic: Modern approach incorporating contextual embeddings

3. **Hyperparameter Tuning**:
   - Number of topics: Critical parameter, use coherence scores to optimize
   - Alpha/Beta in LDA: Control document-topic and topic-word distribution sparsity
   - Number of iterations: More iterations often lead to better convergence

4. **Evaluation**:
   - Use coherence scores to evaluate topic quality
   - Consider human evaluation for real-world applications
   - Compare different models and parameters

5. **Interpretation**:
   - Label topics based on top words
   - Examine documents highly associated with each topic
   - Visualize topics using tools like pyLDAvis 