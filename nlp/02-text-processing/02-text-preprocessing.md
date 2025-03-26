# Text Preprocessing in NLP

Text preprocessing is the process of cleaning and transforming raw text into a format that can be effectively used for NLP tasks. Well-prepared data leads to better model performance and more accurate results.

## Complete Preprocessing Pipeline

A typical text preprocessing pipeline includes these steps:

1. Text normalization (lowercasing, removing accents)
2. Noise removal (HTML tags, special characters)
3. Tokenization (converting text to tokens)
4. Stop word removal (common words with little semantic value)
5. Lemmatization/Stemming (reducing words to base forms)
6. Part-of-speech tagging (optional)
7. Named entity recognition (optional)

## Text Cleaning

### Lowercasing

```python
import spacy

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Example text
text = "The QUICK Brown Fox Jumps Over the Lazy Dog."

# Convert to lowercase
text_lower = text.lower()
print("Lowercase text:", text_lower)
```

### Removing Special Characters and Numbers

```python
import re

# Remove special characters and numbers
def remove_special_chars(text):
    # Keep only letters and spaces
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

cleaned_text = remove_special_chars(text)
print("Cleaned text:", cleaned_text)
```

### Removing HTML Tags

```python
import re

html_text = "<p>This is <b>bold</b> and <i>italic</i> text with a <a href='https://example.com'>link</a>.</p>"

# Remove HTML tags
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

clean_html = remove_html_tags(html_text)
print("HTML removed:", clean_html)
```

## Stop Word Removal

Stop words are common words (like "the", "a", "an", "in") that typically don't contribute much meaning.

```python
import spacy

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Example text
text = "This is an example of text preprocessing for natural language processing tasks."

# Create Doc object
doc = nlp(text)

# Remove stop words
filtered_tokens = [token.text for token in doc if not token.is_stop]
print("After stop word removal:", ' '.join(filtered_tokens))

# Using NLTK for stop word removal
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download stop words
nltk.download('stopwords')
nltk.download('punkt')

# Get English stop words
stop_words = set(stopwords.words('english'))

# Tokenize and remove stop words
tokens = word_tokenize(text)
filtered_tokens_nltk = [word for word in tokens if word.lower() not in stop_words]
print("NLTK stop word removal:", ' '.join(filtered_tokens_nltk))
```

## Lemmatization

Lemmatization reduces words to their base or dictionary form (lemma).

```python
import spacy

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Example text
text = "The cats are running and jumping around the houses"

# Create Doc object
doc = nlp(text)

# Generate lemmas
lemmas = [token.lemma_ for token in doc]
print("Original:", [token.text for token in doc])
print("Lemmatized:", lemmas)

# Lemmatization with NLTK
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Download WordNet
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Tokenize and lemmatize
tokens = word_tokenize(text)
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
print("NLTK lemmatized:", lemmatized_tokens)
```

## Stemming

Stemming reduces words to their root form, often by removing affixes.

```python
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize
import nltk

# Download necessary resources
nltk.download('punkt')

# Example text
text = "The runners running quickly jumped over hurdles and were faster than walking contestants"

# Tokenize
tokens = word_tokenize(text)

# Initialize stemmers
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer('english')

# Apply stemmers
porter_stems = [porter.stem(token) for token in tokens]
lancaster_stems = [lancaster.stem(token) for token in tokens]
snowball_stems = [snowball.stem(token) for token in tokens]

print("Original:", tokens)
print("Porter Stemmer:", porter_stems)
print("Lancaster Stemmer:", lancaster_stems)
print("Snowball Stemmer:", snowball_stems)
```

## Comparison: Stemming vs Lemmatization

| Feature | Stemming | Lemmatization |
|---------|----------|---------------|
| Approach | Rules-based truncation | Dictionary-based linguistic analysis |
| Speed | Faster | Slower |
| Output quality | May produce non-words | Produces actual words |
| Example | "running" → "run" | "running" → "run" |
| Example | "better" → "better" or "bett" | "better" → "good" |
| Resource usage | Lower | Higher |
| Linguistic accuracy | Lower | Higher |
| Best use case | Search engines, information retrieval | Text generation, sentiment analysis |

## Complete Text Preprocessing Example

```python
import spacy
import re
from nltk.tokenize import word_tokenize

def preprocess_text(text, remove_stopwords=True, lemmatize=True):
    """
    Complete text preprocessing function
    
    Args:
        text (str): Text to preprocess
        remove_stopwords (bool): Whether to remove stop words
        lemmatize (bool): Whether to lemmatize tokens
        
    Returns:
        str: Preprocessed text
    """
    # Load spaCy model
    nlp = spacy.load('en_core_web_sm')
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Create Doc object
    doc = nlp(text)
    
    # Process tokens
    processed_tokens = []
    for token in doc:
        # Skip stop words if requested
        if remove_stopwords and token.is_stop:
            continue
        
        # Lemmatize if requested, otherwise use the token text
        if lemmatize:
            processed_tokens.append(token.lemma_)
        else:
            processed_tokens.append(token.text)
    
    # Join tokens back into text
    return ' '.join(processed_tokens)

# Example usage
raw_text = """<p>This is an <b>example</b> text with URLs like https://example.com.
It has some numbers like 123 and special characters like #$%. It also contains
stop words and words in different forms like running, ran, and runs!</p>"""

processed_text = preprocess_text(raw_text)
print("Raw text:\n", raw_text)
print("\nProcessed text:\n", processed_text)
```

## Advanced Techniques

### Text Normalization for Slang and Abbreviations

```python
# Define a dictionary of slang words and their replacements
slang_dict = {
    "u": "you",
    "r": "are",
    "y": "why", 
    "luv": "love",
    "btw": "by the way",
    "lol": "laughing out loud",
    "omg": "oh my god"
}

def normalize_slang(text, slang_dict):
    """Replace slang words with their standard form"""
    words = text.split()
    normalized_words = [slang_dict.get(word.lower(), word) for word in words]
    return ' '.join(normalized_words)

text = "OMG u r so funny lol btw how r u"
normalized_text = normalize_slang(text, slang_dict)
print("Original:", text)
print("Normalized:", normalized_text)
```

### Handling Contractions

```python
import re

contractions_dict = {
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it's": "it is",
    "let's": "let us",
    "shouldn't": "should not",
    "that's": "that is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who's": "who is",
    "who've": "who have",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
}

def expand_contractions(text, contractions_dict):
    """Expand contractions in text"""
    # Create pattern for matching contractions
    pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())), 
                         flags=re.IGNORECASE|re.DOTALL)
    
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded = contractions_dict.get(match.lower())
        if expanded:
            if first_char.isupper():
                expanded = expanded[0].upper() + expanded[1:]
            return expanded
        else:
            return match
    
    # Expand contractions
    expanded_text = pattern.sub(expand_match, text)
    return expanded_text

text = "I can't believe they're not coming! It's a shame."
expanded = expand_contractions(text, contractions_dict)
print("With contractions:", text)
print("Expanded:", expanded)
```

## Text Preprocessing in Practice

When preprocessing text, consider:

1. **Task-specific needs**: Different NLP tasks may require different preprocessing steps
2. **Domain considerations**: Technical, medical, or social media text may need specialized preprocessing
3. **Language considerations**: Different languages require different preprocessing approaches
4. **Balancing preprocessing**: Too much preprocessing can remove important information
5. **Documenting choices**: Maintain a record of preprocessing decisions for reproducibility 