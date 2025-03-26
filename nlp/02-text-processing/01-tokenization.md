# Tokenization in NLP

Tokenization is the process of breaking down text into smaller units called tokens. These tokens can be words, characters, or subwords, which form the basic building blocks for further NLP analysis.

## Why Tokenization Matters

- Foundation for all NLP tasks
- Allows computational processing of text
- Enables feature extraction for machine learning models
- Facilitates analysis of text structure and meaning

## Types of Tokenization

### Word Tokenization

Splits text into individual words or tokens based on delimiters like spaces or punctuation.

```python
import nltk
from nltk.tokenize import word_tokenize

# Ensure NLTK resources are available
nltk.download('punkt')

text = "NLTK provides powerful tokenization functionality. It works across multiple languages!"

# Simple word tokenization
tokens = word_tokenize(text)
print("Word tokens:", tokens)
```

Output:
```
Word tokens: ['NLTK', 'provides', 'powerful', 'tokenization', 'functionality', '.', 'It', 'works', 'across', 'multiple', 'languages', '!']
```

### Sentence Tokenization

Splits text into individual sentences.

```python
from nltk.tokenize import sent_tokenize

paragraph = """Hello there! How are you doing today? NLTK is a powerful library.
It provides many tools for NLP tasks. Let's explore more."""

# Sentence tokenization
sentences = sent_tokenize(paragraph)
print("Sentences:")
for i, sent in enumerate(sentences):
    print(f"{i+1}. {sent}")
```

Output:
```
Sentences:
1. Hello there!
2. How are you doing today?
3. NLTK is a powerful library.
4. It provides many tools for NLP tasks.
5. Let's explore more.
```

## Tokenization with spaCy

spaCy provides high-performance tokenization that integrates with its linguistic pipeline.

```python
import spacy

# Load the English language model
nlp = spacy.load('en_core_web_sm')

# Process the text
text = "SpaCy offers industrial-strength NLP capabilities! It's fast and accurate."
doc = nlp(text)

# Extract tokens
tokens = [token.text for token in doc]
print("spaCy tokens:", tokens)

# Extract sentences
sentences = [sent.text for sent in doc.sents]
print("spaCy sentences:", sentences)
```

Output:
```
spaCy tokens: ['SpaCy', 'offers', 'industrial', '-', 'strength', 'NLP', 'capabilities', '!', 'It', "'s", 'fast', 'and', 'accurate', '.']
spaCy sentences: ['SpaCy offers industrial-strength NLP capabilities!', "It's fast and accurate."]
```

## Specialized Tokenizers

### TweetTokenizer for Social Media Text

```python
from nltk.tokenize import TweetTokenizer

tweet = "Just updated my #NLP toolkit! @friend check it out :) #python #machinelearning"

# Initialize the tokenizer
tweet_tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

# Tokenize the tweet
tokens = tweet_tokenizer.tokenize(tweet)
print("Tweet tokens:", tokens)
```

Output:
```
Tweet tokens: ['just', 'updated', 'my', '#nlp', 'toolkit', '!', 'check', 'it', 'out', ':)', '#python', '#machinelearning']
```

### Regular Expression Tokenizer

Allows custom tokenization patterns:

```python
from nltk.tokenize import RegexpTokenizer

# Create a tokenizer that splits on non-alphanumeric characters
tokenizer = RegexpTokenizer(r'\w+')
text = "Hello, world! This is a test; with some punctuation."
tokens = tokenizer.tokenize(text)
print("Regexp tokens:", tokens)

# Create a tokenizer that captures hashtags and mentions
social_tokenizer = RegexpTokenizer(r'#\w+|@\w+|\w+')
social_text = "Check out #NLPtools with @janedoe and @johnsmith #AI"
social_tokens = social_tokenizer.tokenize(social_text)
print("Social media tokens:", social_tokens)
```

Output:
```
Regexp tokens: ['Hello', 'world', 'This', 'is', 'a', 'test', 'with', 'some', 'punctuation']
Social media tokens: ['Check', 'out', '#NLPtools', 'with', '@janedoe', 'and', '@johnsmith', '#AI']
```

## Subword Tokenization

Modern NLP models often use subword tokenization to handle out-of-vocabulary words.

### WordPiece (used by BERT)

```python
from transformers import BertTokenizer

# Load pre-trained tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize text
text = "The transformer architecture revolutionized NLP in 2017."
tokens = tokenizer.tokenize(text)
print("BERT tokens:", tokens)

# Uncommon word example
uncommon = "The antidisestablishmentarianism movement was controversial."
uncommon_tokens = tokenizer.tokenize(uncommon)
print("Uncommon word tokens:", uncommon_tokens)
```

Output:
```
BERT tokens: ['the', 'transformer', 'architecture', 'revolution', '##ized', 'nl', '##p', 'in', '2017', '.']
Uncommon word tokens: ['the', 'anti', '##dis', '##establish', '##ment', '##arian', '##ism', 'movement', 'was', 'controversial', '.']
```

## Tokenization Challenges

| Challenge | Description | Example |
|-----------|-------------|---------|
| Ambiguity | Words with unclear boundaries | "New York-based company" |
| Contractions | Shortened forms of words | "don't", "I'm", "we've" |
| Hyphenation | Words connected with hyphens | "state-of-the-art" |
| Apostrophes | Possessive forms and contractions | "John's book" vs "Johns" |
| Compound words | Words that can be written as one or multiple tokens | "desktop" vs "desk top" |
| Special characters | Handling punctuation and symbols | "#hashtags", "@mentions", "semi-colons;" |
| Multiple languages | Different tokenization rules for languages | English vs Chinese (no spaces) |

## Tokenization Best Practices

1. **Choose the right tokenizer** for your specific application and language
2. **Preprocess text** before tokenization (e.g., normalize case, handle special characters)
3. **Consider domain-specific needs** (e.g., social media, scientific papers, legal documents)
4. **Be consistent** in how you apply tokenization across training and inference
5. **Evaluate tokenization quality** for your specific task
6. **Handle edge cases** like URLs, email addresses, dates, and numbers appropriately 