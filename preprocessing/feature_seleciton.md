# Feature Selection

## Redundant features

exists in another form
Strong correlation


### Methods

#### Manual

^ We create new features and we have duplicate
* We have granular information and we don''t see value of some of them (Locations, Dates)

``` shell 
#   Column                     Non-Null Count  Dtype         
---  ------                     --------------  -----         
 0   vol_requests               617 non-null    int64         
 1   title                      617 non-null    object        
 2   hits                       617 non-null    int64         
 3   category_desc              617 non-null    object        
 4   locality                   552 non-null    object        
 5   region                     617 non-null    object        
 6   postalcode                 612 non-null    float64       
 7   created_date               617 non-null    datetime64[ns]
 8   vol_requests_lognorm       617 non-null    float64       
 9   created_month              617 non-null    int64         
 10  Education                  617 non-null    uint8         
 11  Emergency Preparedness     617 non-null    uint8         
 12  Environment                617 non-null    uint8         
 13  Health                     617 non-null    uint8         
 14  Helping Neighbors in Need  617 non-null    uint8         
 15  Strengthening Communities  617 non-null    uint8
```

```python 
# Create a list of redundant column names to drop
to_drop = ["locality", "region", "category_desc", "vol_requests", "created_date" ]

# Drop those columns from the dataset
volunteer_subset = volunteer.drop(to_drop, axis=1)

# Print out the head of volunteer_subset
print(volunteer_subset.head())
```

#### Correlation

```shell
In [1]: wine.corr()
Out[1]:
                              Flavanoids  Total phenols  Malic acid  OD280/OD315 of diluted wines    Hue
Flavanoids                         1.000          0.865      -0.411                         0.787  0.543
Total phenols                      0.865          1.000      -0.335                         0.700  0.434
Malic acid                        -0.411         -0.335       1.000                        -0.369 -0.561
OD280/OD315 of diluted wines       0.787          0.700      -0.369                         1.000  0.565
Hue                                0.543          0.434      -0.561                         0.565  1.000
```

``` python 
# Drop that column from the DataFrame
wine = wine.drop("Flavanoids", axis= 1)
```

### Reduce text vectors

```python 
# Add in the rest of the arguments
def return_weights(vocab, original_vocab, vector, vector_index, top_n):
    zipped = dict(zip(vector[vector_index].indices, vector[vector_index].data))
    
    # Transform that zipped dict into a series
    zipped_series = pd.Series({vocab[i]:zipped[i] for i in vector[vector_index].indices})
    
    # Sort the series to pull out the top n weighted words
    zipped_index = zipped_series.sort_values(ascending=False)[:top_n].index
    return [original_vocab[i] for i in zipped_index]

def words_to_filter(vocab, original_vocab, vector, top_n):
    filter_list = []
    for i in range(0, vector.shape[0]):
    
        # Call the return_weights function and extend filter_list
        filtered = return_weights(vocab, original_vocab, vector, i, top_n)
        filter_list.extend(filtered)
        
    # Return the list in a set, so we don't get duplicate word indices
    return set(filter_list)

# Call the function to get the list of word indices
filtered_words = words_to_filter(vocab, tfidf_vec.vocabulary_, text_tfidf, top_n=3)

# Filter the columns in text_tfidf to only those in filtered_words
filtered_text = text_tfidf[:, list(filtered_words)]

# Split the dataset according to the class distribution of category_desc
X_train, X_test, y_train, y_test = train_test_split(filtered_text.toarray(), y, stratify=y, random_state=42)

# Fit the model to the training data
nb.fit(X_train, y_train)

# Print out the model's accuracy
print(nb.score(X_test, y_test))
```

#### Dimensionality reduction

This is explained extensively in the dimension-reduction