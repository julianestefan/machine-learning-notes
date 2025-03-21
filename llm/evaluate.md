# Evaluate library 

```python 
# Load the metrics
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

print(accuracy.description )
print(precision.description)
print(recall.description)
print(f1.description)
```
Output:

```shell
Precision is the fraction of correctly labeled positive examples out of all of the examples that were labeled as positive. It is computed via the equation:
Precision = TP / (TP + FP)
where TP is the True positives (i.e. the examples correctly labeled as positive) and FP is the False positive examples (i.e. the examples incorrectly labeled as positive).


Recall is the fraction of the positive examples that were correctly labeled by the model as positive. It can be computed with the equation:
Recall = TP / (TP + FN)
Where TP is the true positives and FN is the false negatives.


The F1 score is the harmonic mean of the precision and recall. It can be computed with the equation:
F1 = 2 * (precision * recall) / (precision + recall)
```

## Basic Classification Metrics

### Precision
$$\text{Precision} = \frac{TP}{TP + FP}$$

Where:
- TP = True Positives (correctly labeled as positive)
- FP = False Positives (incorrectly labeled as positive)

### Recall
$$\text{Recall} = \frac{TP}{TP + FN}$$

Where:
- TP = True Positives (correctly labeled as positive)
- FN = False Negatives (incorrectly labeled as negative)

### F1 Score
$$F1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Accuracy
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Where:
- TP = True Positives
- TN = True Negatives
- FP = False Positives
- FN = False Negatives

## Usage

```python 
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

# Extract the new predictions
predicted_labels = torch.argmax(outputs.logits, dim=1).tolist()

# Compute the metrics by comparing real and predicted labels
print(accuracy.compute(references=validate_labels, predictions=predicted_labels))
print(precision.compute(references=validate_labels, predictions=predicted_labels))
print(recall.compute(references=validate_labels, predictions=predicted_labels))
print(f1.compute(references=validate_labels, predictions=predicted_labels))
```

# Metrics for language tasks

![Metrics for LLM tasks](./assets/metrics-for-llm-task.png)

## Perplexity

Perplexity is a metric used to evaluate language models. It measures how well a probability distribution or probability model predicts a sample. A lower perplexity indicates a better predictive model. For a given sequence of words, perplexity is calculated as the exponentiation of the average negative log-likelihood of the sequence.

Mathematically, for a sequence of words $w_1, w_2, \ldots, w_N$, the perplexity $PP$ is defined as:

$$PP(W) = \exp \left( -\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_1, w_2, \ldots, w_{i-1}) \right)$$

where $P(w_i | w_1, w_2, \ldots, w_{i-1})$ is the conditional probability of the word $w_i$ given the previous words in the sequence.

In simpler terms, perplexity can be considered as the weighted branching factor of a language model, indicating how many options the model is considering at each step. Lower perplexity values suggest that the model is more confident in its predictions.

```python 
# Encode the input text, generate and decode it
input_text_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_text_ids, max_length=20)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Text: ", generated_text)

# Load and compute the perplexity score
perplexity = evaluate.load("perplexity", module_type="metric")
results = perplexity.compute(model_id="gpt2", predictions=generated_text)
print("Perplexity: ", results['mean_perplexity'])
```

## BLEU

BLEU (Bilingual Evaluation Understudy) is a metric for evaluating the quality of text which has been machine-translated from one language to another. It compares the machine-translated text to one or more reference translations. The BLEU score ranges from 0 to 1, with 1 indicating a perfect match between the machine translation and the reference.

The BLEU score is calculated based on the precision of n-grams (contiguous sequences of n items from a given sample of text) in the machine-translated text compared to the reference translations. It also includes a brevity penalty to penalize translations that are too short.

Mathematically, the BLEU score is computed as:

$$\text{BLEU} = \text{BP} \cdot \exp \left( \sum_{n=1}^{N} w_n \log p_n \right)$$

where:
- $\text{BP}$ is the brevity penalty,
- $w_n$ is the weight for n-gram precision (usually uniform weights are used),
- $p_n$ is the precision of n-grams.

The brevity penalty $\text{BP}$ is calculated as:

$$\text{BP} = 
\begin{cases} 
1 & \text{if } c > r \\
\exp(1 - \frac{r}{c}) & \text{if } c \leq r 
\end{cases}$$

where $c$ is the length of the candidate translation and $r$ is the length of the reference translation.

In summary, the BLEU score provides a quantitative measure of the accuracy of machine translations by comparing them to human-generated reference translations.

```python 
# Translate the input sentences, extract the translated text, and compute BLEU score
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")

translated_outputs = translator(input_sentences_2, clean_up_tokenization_spaces=True)

predictions = [translated_output["translation_text"] for translated_output in translated_outputs]
print(predictions)

results = bleu.compute(predictions=predictions, references=references_2)
print(results)
```

output:

```shell 
 {'bleu': 0.8627788640890415, 'precisions': [0.9090909090909091, 0.8888888888888888, 0.8571428571428571, 0.8], 'brevity_penalty': 1.0, 'length_ratio': 1.0, 'translation_length': 11, 'reference_length': 11}
```

## ROUGE

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics designed to evaluate automatic summarization and machine translation. It compares a generated summary or translation against reference texts.

The most common ROUGE metrics include:

### ROUGE-N
Measures the overlap of n-grams between the generated and reference texts:

$$\text{ROUGE-N} = \frac{\sum_{S \in \text{References}} \sum_{\text{gram}_n \in S} \text{Count}_{\text{match}}(\text{gram}_n)}{\sum_{S \in \text{References}} \sum_{\text{gram}_n \in S} \text{Count}(\text{gram}_n)}$$

Where Count_match is the maximum number of n-grams co-occurring in the candidate summary and the reference summaries.

### ROUGE-L
Measures the longest common subsequence (LCS) between the generated and reference texts:

$$\text{ROUGE-L}_{\text{precision}} = \frac{\text{LCS}(X,Y)}{|X|}$$

$$\text{ROUGE-L}_{\text{recall}} = \frac{\text{LCS}(X,Y)}{|Y|}$$

$$\text{ROUGE-L}_{\text{F-score}} = \frac{(1 + \beta^2)\text{ROUGE-L}_{\text{precision}} \times \text{ROUGE-L}_{\text{recall}}}{\text{ROUGE-L}_{\text{precision}} + \beta^2 \times \text{ROUGE-L}_{\text{recall}}}$$

Where X is the candidate summary, Y is the reference summary, and |X| and |Y| are their lengths respectively.

```python 
# Load the rouge metric
rouge = evaluate.load("rouge")

predictions = ["""Pluto is a dwarf planet in our solar system, located in the Kuiper Belt beyond Neptune, and was formerly considered the ninth planet until its reclassification in 2006."""]
references = ["""Pluto is a dwarf planet in the solar system, located in the Kuiper Belt beyond Neptune, and was previously deemed as a planet until it was reclassified in 2006."""]

# Calculate the rouge scores between the predicted and reference summaries
results = rouge.compute(predictions=predictions, references=references)
print("ROUGE results: ", results)
```

## METEOR

METEOR (Metric for Evaluation of Translation with Explicit ORdering) is an evaluation metric for machine translation that addresses some limitations of BLEU by considering stemming, synonymy, and word order.

The METEOR score is calculated using:

$$\text{METEOR} = (1 - \gamma \cdot \text{fragmentation}^{\beta}) \cdot \frac{P \cdot R}{\alpha \cdot P + (1 - \alpha) \cdot R}$$

Where:
- $P$ is precision (the ratio of the number of matched words to the total number of words in the candidate translation)
- $R$ is recall (the ratio of the number of matched words to the total number of words in the reference translation)
- $\text{fragmentation}$ measures how well-ordered the matched words are
- $\alpha$, $\beta$, and $\gamma$ are parameters typically set to 0.9, 3.0, and 0.5 respectively

METEOR considers exact matches, stemmed matches, and synonym matches to create alignment between candidate and reference translations.

```python 
meteor = evaluate.load("meteor")

generated = ["The burrow stretched forward like a narrow corridor for a while, then plunged abruptly downward, so quickly that Alice had no chance to stop herself before she was tumbling into an extremely deep shaft."]
reference = ["The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well."]

# Compute and print the METEOR score
results = meteor.compute(predictions=generated, references=reference)
print("Meteor: ", results)
```

## Exact Match

Exact Match (EM) is one of the simplest evaluation metrics, measuring the percentage of predictions that exactly match their references. The score is binary for each example (1 if the prediction matches exactly, 0 otherwise) and the final score is the average across all examples.

$$\text{Exact Match} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(prediction_i = reference_i)$$

Where:
- $N$ is the total number of examples
- $\mathbb{1}$ is the indicator function that returns 1 if the prediction exactly matches the reference, and 0 otherwise

```python 
# Load the metric
exact_match = evaluate.load("exact_match")

predictions = ["It's a wonderful day", "I love dogs", "DataCamp has great AI courses", "Sunshine and flowers"]
references = ["What a wonderful day", "I love cats", "DataCamp has great AI courses", "Sunsets and flowers"]

# Compute the exact match and print the results
results = exact_match.compute(predictions=predictions, references=references)
print("EM results: ", results)
```
Output:
```shelll
EM results:  {'exact_match': 0.25}
```

# Safeguarding 

## Toxicity

Toxicity measures how rude, disrespectful, or unreasonable a piece of text is. The metric provides different aggregation methods:

- Individual toxicity scores for each text
- Maximum toxicity score across all texts
- Toxicity ratio (percentage of texts with toxicity above a threshold)

The toxicity score ranges from 0 to 1, with higher values indicating more toxic content.

$$\text{Toxicity Ratio} = \frac{\text{Number of texts with toxicity > threshold}}{\text{Total number of texts}}$$

```python 
# Calculate the individual toxicities
toxicity_1 = toxicity_metric.compute(predictions=user_1)
toxicity_2 = toxicity_metric.compute(predictions=user_2)
print("Toxicities (user_1):", toxicity_1['toxicity'])
print("Toxicities (user_2): ", toxicity_2['toxicity'])

# Calculate the maximum toxicities
toxicity_1_max = toxicity_metric.compute(predictions=user_1, aggregation="maximum")
toxicity_2_max = toxicity_metric.compute(predictions=user_2, aggregation="maximum")
print(toxicity_1_max)
print("Maximum toxicity (user_1):", toxicity_1_max['max_toxicity'])
print("Maximum toxicity (user_2): ", toxicity_2_max['max_toxicity'])

# Calculate the toxicity ratios
toxicity_1_ratio =  toxicity_metric.compute(predictions=user_1, aggregation="ratio")
toxicity_2_ratio = toxicity_metric.compute(predictions=user_2, aggregation="ratio")
print("Toxicity ratio (user_1):", toxicity_1_ratio['toxicity_ratio'])
print("Toxicity ratio (user_2): ", toxicity_2_ratio['toxicity_ratio'])
```
Output:
```python 
Toxicities (user_1): [0.00013486345415003598, 0.00013348401989787817]
Toxicities (user_2):  [0.00013559251965489239, 0.00013771136582363397]
Maximum toxicity (user_1): 0.00013486345415003598
Maximum toxicity (user_2):  0.00013771136582363397
Toxicity ratio (user_1): 0.0
Toxicity ratio (user_2):  0.0
```

## Regard

Regard measures the sentiment analysis of text towards specific demographic groups. It classifies text into different sentiment categories:
- Neutral
- Positive
- Negative
- Other

The regard metric can also compare the sentiment difference between texts referencing different demographic groups:

$$\text{Regard Difference} = \text{Sentiment}_{group1} - \text{Sentiment}_{group2}$$

This helps identify potential bias in language models when discussing different demographic groups.

```python 
# Load the regard and regard-comparison metrics
regard = evaluate.load("regard")
regard_comp = evaluate.load("regard", "compare")

# Compute the regard (polarities) of each group separately
polarity_results_1 = regard.compute(data=group1)
print("Polarity in group 1:\n", polarity_results_1)
polarity_results_2 = regard.compute(data=group2)
print("Polarity in group 2:\n", polarity_results_2)

# Compute the relative regard between the two groups for comparison
polarity_results_comp = regard_comp.compute(data=group1, references=group2)
print("Polarity comparison between groups:\n", polarity_results_comp)
```
Output:
```python 
Polarity in group 1:
{'regard': [[{'label': 'neutral', 'score': 0.9586169719696045}, {'label': 'negative', 'score': 0.020242035388946533}, {'label': 'positive', 'score': 0.014409131370484829}, {'label': 'other', 'score': 0.006731783039867878}], [{'label': 'neutral', 'score': 0.8816481828689575}, {'label': 'positive', 'score': 0.08354275673627853}, {'label': 'negative', 'score': 0.019816529005765915}, {'label': 'other', 'score': 0.014992612414062023}]]}
Polarity in group 2:
{'regard': [[{'label': 'negative', 'score': 0.9745951890945435}, {'label': 'other', 'score': 0.017152629792690277}, {'label': 'neutral', 'score': 0.007746347691863775}, {'label': 'positive', 'score': 0.0005058051901869476}], [{'label': 'neutral', 'score': 0.766608476638794}, {'label': 'negative', 'score': 0.10047465562820435}, {'label': 'positive', 'score': 0.07146857678890228}, {'label': 'other', 'score': 0.061448343098163605}]]}
Polarity comparison between groups:
{'regard_difference': {'neutral': 0.5329551652539521, 'negative': -0.5175056401640177, 'positive': 0.012988753063837066, 'other': -0.02843828871846199}}
```