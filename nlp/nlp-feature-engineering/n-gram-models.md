# Bag of words (BOW)

It allows you to convert a tet in a vector of occurrences. IT's important to do some text preprocessing to avoid unnecessary tokens to be included and improve performance.

```python 
# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create CountVectorizer object
vectorizer = CountVectorizer()

# Generate matrix of word vectors
bow_matrix = vectorizer.fit_transform(corpus)

# Convert bow_matrix into a DataFrame
bow_df = pd.DataFrame(bow_matrix.toarray())

# Map the column names to vocabulary 
bow_df.columns = vectorizer.get_feature_names()
```

Output:

```shell
an  decade  endangered  have  is  ...  lion  lions  of  species  the
0   0       0           0     0   1  ...     1      0   1        0    3
1   0       1           0     1   0  ...     0      1   1        0    0
2   1       0           1     0   1  ...     1      0   0        1    1
    
[3 rows x 13 columns]
```

# NAive Bayes BOW 

```python
# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create a CountVectorizer object
vectorizer = CountVectorizer(lowercase=True, stop_words='english')

# Fit and transform X_train
X_train_bow = vectorizer.fit_transform(X_train)

# Transform X_test
X_test_bow = vectorizer.transform(X_test)


# Create a MultinomialNB object
clf = MultinomialNB()

# Fit the classifier
clf.fit(X_train_bow, y_train)

# Measure the accuracy
accuracy = clf.score(X_test_bow, y_test)
print("The accuracy of the classifier on the test set is %.3f" % accuracy)

# Predict the sentiment of a negative review
review = "The movie was terrible. The music was underwhelming and the acting mediocre."
prediction = clf.predict(vectorizer.transform([review]))[0]
print("The sentiment predicted by the classifier is %i" % (prediction))
```

Output:

```shell 
The accuracy of the classifier on the test set is 0.732
The sentiment predicted by the classifier is 0
```

#N grams

Helps to keep context performing analysis on a sequence of contiguous words.


```python 
# Generate n-grams upto n=1
vectorizer_ng1 = CountVectorizer(ngram_range=(1,1))
ng1 = vectorizer_ng1.fit_transform(corpus)

# Generate n-grams upto n=2
vectorizer_ng2 = CountVectorizer(ngram_range=(1,2))
ng2 = vectorizer_ng2.fit_transform(corpus)

# Generate n-grams upto n=3
vectorizer_ng3 = CountVectorizer(ngram_range=(1, 3))
ng3 = vectorizer_ng3.fit_transform(corpus)

# Print the number of features for each model
print("ng1, ng2 and ng3 have %i, %i and %i features respectively" % (ng1.shape[1], ng2.shape[1], ng3.shape[1]))
```

Output: 

```shell 
ng1, ng2 and ng3 have 6614, 37100 and 76881 features respectively
```

Using high n in your n-grams could lead to increased computational costs and a problem known as the curse of dimensionality.