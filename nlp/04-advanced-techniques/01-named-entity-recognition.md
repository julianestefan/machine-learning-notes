# Named Entity Recognition (NER)

Named Entity Recognition is an NLP technique that identifies and classifies named entities in text into predefined categories such as person names, organizations, locations, time expressions, quantities, monetary values, and more.

## Why NER Matters

NER addresses key information extraction questions like:
- **Who?** (Person)
- **Where?** (Location)
- **When?** (Time)
- **What organization?** (Organization)
- **How much?** (Money, Quantity)

Applications include:
- Information extraction
- Question answering
- Search engine optimization
- Content recommendation
- Knowledge graph construction
- Customer support automation

## Common Entity Types

| Entity Type | Description | Examples |
|-------------|-------------|----------|
| PERSON | People, including fictional characters | "Barack Obama", "Harry Potter" |
| ORG | Companies, agencies, institutions | "Apple Inc.", "United Nations" |
| GPE/LOC | Countries, cities, states, locations | "New York", "Europe", "Mount Everest" |
| DATE | Absolute or relative dates | "January 1st", "yesterday" |
| TIME | Times of day | "5pm", "noon" |
| MONEY | Monetary values | "$100", "fifty euros" |
| PERCENT | Percentage values | "25%", "one-third" |
| PRODUCT | Objects, vehicles, foods, etc. | "iPhone", "Boeing 747" |
| EVENT | Named events | "World War II", "Super Bowl" |
| WORK_OF_ART | Titles of books, songs, etc. | "The Great Gatsby" |

## NER with NLTK

NLTK provides NER capabilities through its chunking and named entity chunking functions:

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import ne_chunk
import matplotlib.pyplot as plt
from collections import defaultdict

# Download necessary resources
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

# Sample text
article = """
The European Union's foreign policy chief, Josep Borrell, visited Kyiv on Wednesday, 
where he met with Ukrainian President Volodymyr Zelenskyy to discuss the war with Russia. 
Apple Inc. announced its new iPhone 15 at an event in Cupertino, California last Friday.
Tesla CEO Elon Musk plans to visit China next month to discuss expanding operations there.
"""

# Tokenize the article into sentences and then into words
sentences = sent_tokenize(article)
token_sentences = [word_tokenize(sent) for sent in sentences]

# Tag each word with its part of speech
pos_sentences = [nltk.pos_tag(sent) for sent in token_sentences]

# Extract named entities
chunked_sentences = nltk.ne_chunk_sents(pos_sentences, binary=True)

# Extract named entities with their type
typed_chunked_sentences = nltk.ne_chunk_sents(pos_sentences, binary=False)

# Print named entities without types (binary=True)
print("Named Entities (binary):")
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, "label") and chunk.label() == "NE":
            entity = " ".join([token for token, pos in chunk.leaves()])
            print(f"- {entity}")

# Print named entities with types (binary=False)
print("\nNamed Entities with Types:")
for sent in typed_chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, "label"):
            entity = " ".join([token for token, pos in chunk.leaves()])
            entity_type = chunk.label()
            print(f"- {entity} ({entity_type})")
```

### Visualizing NER Distribution

```python
# Create a distribution of named entity types
ner_categories = defaultdict(int)

for sent in typed_chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, 'label'):
            ner_categories[chunk.label()] += 1

# Create a bar chart of entity types
labels = list(ner_categories.keys())
values = [ner_categories.get(label) for label in labels]

plt.figure(figsize=(12, 6))
plt.bar(labels, values)
plt.title('Distribution of Named Entity Types')
plt.xlabel('Entity Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create a pie chart
plt.figure(figsize=(10, 10))
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Named Entity Type Distribution')
plt.show()
```

## NER with spaCy

spaCy provides more accurate and efficient NER capabilities compared to NLTK:

```python
import spacy
from spacy import displacy
import pandas as pd

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Process the text
doc = nlp(article)

# Extract entities
entities = [(ent.text, ent.label_, spacy.explain(ent.label_)) for ent in doc.ents]

# Display entities in a table
entities_df = pd.DataFrame(entities, columns=['Entity', 'Type', 'Description'])
print(entities_df)

# Visualize entities in the browser or notebook
displacy.render(doc, style="ent", jupyter=True)

# Count entity types
entity_counts = defaultdict(int)
for ent in doc.ents:
    entity_counts[ent.label_] += 1

# Create a bar chart
plt.figure(figsize=(12, 6))
plt.bar(entity_counts.keys(), entity_counts.values())
plt.title('Named Entity Types in Text')
plt.xlabel('Entity Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Customizing spaCy's NER Visualization

```python
# Customize colors for entity visualization
colors = {"ORG": "#7aecec", "PERSON": "#faa", "GPE": "#9cc9cc", 
          "DATE": "#bfe1d9", "PRODUCT": "#e4e7d2", "EVENT": "#c887fb"}

options = {"ents": list(colors.keys()), "colors": colors}
displacy.render(doc, style="ent", options=options, jupyter=True)
```

## NER with Transformers (Hugging Face)

State-of-the-art NER using transformer models:

```python
from transformers import pipeline
import pandas as pd

# Load NER pipeline with a BERT-based model
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Run NER on the text
results = ner_pipeline(article)

# Group consecutive tokens with the same entity type
entities = []
current_entity = {"text": "", "entity": "", "score": 0.0, "count": 0}

for result in results:
    # If this is a continuation of the previous entity (same entity type and adjacent)
    if (result["entity"].startswith("B-") or 
        (result["entity"].startswith("I-") and current_entity["entity"] == result["entity"][2:] and 
         result["start"] == current_entity["end"])):
        
        # If this is the start of a new entity
        if result["entity"].startswith("B-"):
            # Save the previous entity if it exists
            if current_entity["count"] > 0:
                entities.append({
                    "text": current_entity["text"].strip(),
                    "entity": current_entity["entity"],
                    "score": current_entity["score"] / current_entity["count"]
                })
            # Start a new entity
            current_entity = {
                "text": result["word"],
                "entity": result["entity"][2:],
                "score": result["score"],
                "count": 1,
                "end": result["end"]
            }
        # If this is a continuation of the current entity
        else:
            current_entity["text"] += " " + result["word"]
            current_entity["score"] += result["score"]
            current_entity["count"] += 1
            current_entity["end"] = result["end"]
    # If this is a new entity or not a continuation
    else:
        # Save the previous entity if it exists
        if current_entity["count"] > 0:
            entities.append({
                "text": current_entity["text"].strip(),
                "entity": current_entity["entity"],
                "score": current_entity["score"] / current_entity["count"]
            })
        # Start a new entity if this is a B- tag
        if result["entity"].startswith("B-"):
            current_entity = {
                "text": result["word"],
                "entity": result["entity"][2:],
                "score": result["score"],
                "count": 1,
                "end": result["end"]
            }
        else:
            current_entity = {"text": "", "entity": "", "score": 0.0, "count": 0}

# Add the last entity if it exists
if current_entity["count"] > 0:
    entities.append({
        "text": current_entity["text"].strip(),
        "entity": current_entity["entity"],
        "score": current_entity["score"] / current_entity["count"]
    })

# Create a DataFrame for better visualization
entities_df = pd.DataFrame(entities)
entities_df["score"] = entities_df["score"].round(3)
print(entities_df.sort_values(by="entity"))
```

## Multilingual NER with Polyglot

Polyglot allows NER in 130+ languages:

```python
from polyglot.text import Text

# Example text in different languages
texts = {
    "English": "Barack Obama was born in Hawaii. He was the 44th President of the United States.",
    "Spanish": "Gabriel García Márquez, conocido como Gabo, nació en Colombia y escribió Cien años de soledad.",
    "French": "La Tour Eiffel est située à Paris, en France. Elle a été construite par Gustave Eiffel."
}

# Process each text
for language, text in texts.items():
    print(f"\n=== {language} ===")
    try:
        # Create a Text object
        text_obj = Text(text, hint_language_code=language.lower()[:2])
        
        # Extract entities
        entities = [(ent.tag, ' '.join(ent)) for ent in text_obj.entities]
        
        # Print entities
        for entity_type, entity_text in entities:
            print(f"- {entity_text} ({entity_type})")
    except Exception as e:
        print(f"Error processing {language}: {e}")
```

### Entity Percentage Analysis

```python
def calculate_entity_percentage(text, target_entities):
    """Calculate the percentage of entities matching the targets"""
    text_obj = Text(text)
    total_entities = len(text_obj.entities)
    matching_entities = 0
    
    for ent in text_obj.entities:
        entity_text = ' '.join(ent)
        if any(target in entity_text for target in target_entities):
            matching_entities += 1
            
    if total_entities > 0:
        percentage = (matching_entities / total_entities) * 100
    else:
        percentage = 0
        
    return matching_entities, total_entities, percentage

# Example usage
article = "Gabriel García Márquez, also known as Gabo, was a Colombian novelist. Gabo was born in Aracataca and lived in Mexico City. He won the Nobel Prize in Literature in 1982."
targets = ["Márquez", "Gabo"]

matches, total, percentage = calculate_entity_percentage(article, targets)
print(f"Found {matches} matches for target entities out of {total} total entities")
print(f"Percentage: {percentage:.1f}%")
```

## Custom NER Model Training

Creating a custom NER model with spaCy:

```python
import spacy
from spacy.tokens import DocBin
from spacy.training import Example

# Training data (text and entity annotations)
training_data = [
    ("Apple Inc. launched iPhone 15 in California yesterday.", 
     {"entities": [(0, 10, "ORG"), (20, 29, "PRODUCT"), (33, 43, "GPE")]}),
    ("Microsoft announced Windows 11 at their headquarters in Seattle.",
     {"entities": [(0, 9, "ORG"), (20, 30, "PRODUCT"), (52, 59, "GPE")]}),
    ("Tesla's Cybertruck will be manufactured in Texas.",
     {"entities": [(0, 5, "ORG"), (7, 17, "PRODUCT"), (40, 45, "GPE")]})
]

# Initialize spaCy model
nlp = spacy.blank("en")

# Create a DocBin to store training documents
doc_bin = DocBin()

# Convert training data to spaCy format
for text, annotations in training_data:
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, annotations)
    doc_bin.add(example.reference)

# Save the DocBin to a file
doc_bin.to_disk("train.spacy")

# Train the model (in real scenario)
'''
!python -m spacy train config.cfg --output ./model --paths.train train.spacy --paths.dev train.spacy
'''
```

## Best Practices for NER

1. **Data Preprocessing**:
   - Clean your text (remove unnecessary characters, normalize casing)
   - Handle special characters and punctuation appropriately

2. **Model Selection**:
   - Use spaCy for general-purpose NER with good accuracy and speed
   - Use transformer-based models for state-of-the-art accuracy
   - Use NLTK for basic NER tasks or educational purposes
   - Use Polyglot for multilingual NER

3. **Custom Entity Types**:
   - Define domain-specific entity types relevant to your task
   - Gather annotated training data for your custom entities

4. **Evaluation**:
   - Use precision, recall, and F1-score to evaluate NER performance
   - Create a test set with gold-standard annotations

5. **Post-processing**:
   - Merge adjacent entities of the same type if appropriate
   - Resolve entity overlaps based on confidence scores
   - Use gazetteer lists to validate extracted entities

6. **Domain Adaptation**:
   - Fine-tune NER models on domain-specific data
   - Use entity rulers to augment statistical models with rule-based approaches

## Challenges in NER

| Challenge | Description | Solution |
|-----------|-------------|----------|
| Ambiguity | Words can be entities in some contexts but not others | Context-aware models (e.g., BERT) |
| Nested entities | Entities contained within other entities | Hierarchical NER models |
| Entity boundary detection | Determining where entities start and end | BIO/BIOES tagging schemes |
| Out-of-vocabulary entities | Entities not seen during training | Character-level features, knowledge bases |
| Domain-specific entities | Technical terms, specialized nomenclature | Domain adaptation, custom training |
| Low-resource languages | Limited training data for some languages | Transfer learning, multilingual models | 