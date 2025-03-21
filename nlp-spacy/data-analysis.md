# Pipelines
When you don't want to run the default pipeline you can create an empty one and add some pipes using the `add_pipe` method

```python 
# Load a blank spaCy English model and add a sentencizer component
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

# Create Doc containers, store sentences and print its number of sentences
doc = nlp(texts)
sentences = [s for s in doc.sents]
print("Number of sentences: ", len(sentences), "\n")

# Print the list of tokens in the second sentence
print("Second sentence tokens: ", [words for words in sentences[1]])
```

You can analyze a pipeline as follows 

```python 
# Load a blank spaCy English model
nlp = spacy.blank("en")

# Add tagger and entity_linker pipeline components
nlp.add_pipe("tagger")
nlp.add_pipe("entity_linker")

# Analyze the pipeline
analysis = nlp.analyze_pipes(pretty=True)
```
Output: 
```shell
============================= Pipeline Overview =============================

#   Component       Assigns           Requires         Scores        Retokenizes
-   -------------   ---------------   --------------   -----------   -----------
0   tagger          token.tag                          tag_acc       False      
                                                                                
1   entity_linker   token.ent_kb_id   doc.ents         nel_micro_f   False      
                                      doc.sents        nel_micro_r              
                                      token.ent_iob    nel_micro_p              
                                      token.ent_type                            


================================ Problems (4) ================================
⚠ 'entity_linker' requirements not met: doc.ents, doc.sents,
token.ent_iob, token.ent_type
```

# Entity Ruler

Add entities to doc.ents. It could be used alone or with `EntityRecognizer`

```python 
nlp = spacy.blank("en")
patterns = [{"label": "ORG", "pattern": [{"LOWER": "openai"}]},
            {"label": "ORG", "pattern": [{"LOWER": "microsoft"}]}]
text = "OpenAI has joined forces with Microsoft."

# Add EntityRuler component to the model
entity_ruler = nlp.add_pipe("entity_ruler", before="ner")

# Add given patterns to the EntityRuler component
entity_ruler.add_patterns(patterns)

# Run the model on a given text
doc = nlp(text)

# Print entities text and type for all entities in the Doc container
print([(ent.text, ent.label_) for ent in doc.ents])
```

You can define it before or after the existing NER leading to different results. It will depend of the use case but you usually would like to give it precedence to existing NER to be more specific.

# Regex in spacy

```python 
text = "Our phone number is 4251234567."

# Define a pattern to match phone numbers
patterns = [
    {
        "label": "PHONE_NUMBERS", 
        "pattern": [
            {"TEXT": {"REGEX": "(\d){10}"}}
        ]
    }
]

# Load a blank model and add an EntityRuler
nlp = spacy.blank("en")
ruler = nlp.add_pipe("entity_ruler")

# Add the compiled patterns to the EntityRuler
ruler.add_patterns(patterns)

# Print the tuple of entities texts and types for the given text
doc = nlp(text)
print([(ent.text, ent.label_) for ent in doc.ents])
```

# Matcher and PhraseMatcher

Matcher is a more expressive way to find match in documents. IT allows the use of the following operators

![Matcher operators](./assets/operators-matcher.png)

```
nlp = spacy.load("en_core_web_sm")
doc = nlp(example_text)

# Define a matcher object
matcher = Matcher(nlp.vocab)
# Define a pattern to match tiny squares and tiny mouthful
pattern = [{"lower": "tiny"}, {"lower": {"IN": ["squares", "mouthful"]}}]

# Add the pattern to matcher object and find matches
matcher.add("CustomMatcher", [pattern])
matches = matcher(doc)

# Print out start and end token indices and the matched text span per match
for match_id, start, end in matches:
    print("Start token: ", start, " | End token: ", end, "| Matched text: ", doc[start:end].text)
```

When you have multiple patterns you can use `PhraseMatcher`. It allows to create patterns from a terms list.

```python 
text = "There are only a few acceptable IP addresses: (1) 127.100.0.1, (2) 123.4.1.0."
terms = ["110.0.0.0", "101.243.0.0"]

# Initialize a PhraseMatcher class to match to shapes of given terms
matcher = PhraseMatcher(nlp.vocab, attr = "SHAPE")

# Create patterns to add to the PhraseMatcher object
patterns = [nlp.make_doc(term) for term in terms]
matcher.add("IPAddresses", patterns)

# Find matches to the given patterns and print start and end characters and matches texts
doc = nlp(text)
matches = matcher(doc)
for match_id, start, end in matches:
    print("Start token: ", start, " | End token: ", end, "| Matched text: ", doc[start:end].text)
```