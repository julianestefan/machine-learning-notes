### Parse and clean data

``` python 
# Change the type of seconds to float
ufo["seconds"] = ufo["seconds"].astype("float")

# Change the date column to type datetime
ufo["date"] = pd.to_datetime(ufo["date"])

# Drop rows where length_of_time, state, or type are missing
ufo_no_missing = ufo.dropna()

def return_minutes(time_string):
    # Search for numbers in time_string
    num = re.search("\d+", time_string)
    if num is not None:
        return int(num.group(0))
        
# Apply the extraction to the length_of_time column
ufo["minutes"] = ufo["length_of_time"].apply( lambda x: return_minutes(x))

# Take a look at the head of both of the columns
print(ufo[["minutes", "length_of_time"]].head())

# Log normalize the seconds column
ufo["seconds_log"] = np.log(ufo["seconds"])
```
### Feature Engineering

```python 
ufo["country_enc"] = ufo["country"].apply( lambda x: 1 if x == "us" else 0 )

# Print the number of unique type values
print(len(ufo["type"].unique()))

# Create a one-hot encoded set of the type values
type_set = pd.get_dummies(ufo["type"])

# Concatenate this set back to the ufo DataFrame
ufo = pd.concat([ufo, type_set], axis=1)

# Extract the month from the date column
ufo["month"] = ufo["date"].dt.month

# Extract the year from the date column
ufo["year"] = ufo["date"].dt.year

# Instantiate the tfidf vectorizer object
vec = TfidfVectorizer()

# Fit and transform desc using vec
desc_tfidf = vec.fit_transform(ufo["desc"])

# Look at the number of columns and rows
print(desc_tfidf.shape)
```

### Selection and modeling

Now to get rid of some of the unnecessary features in the ufo dataset. Because the country column has been encoded as country_enc, you can select it and drop the other columns related to location: city, country, lat, long, and state.

You've engineered the month and year columns, so you no longer need the date or recorded columns. You also standardized the seconds column as seconds_log, so you can drop seconds and minutes.

You vectorized desc, so it can be removed. For now you'll keep type.

You can also get rid of the length_of_time column, which is unnecessary after extracting minutes

``` python 
# Make a list of features to drop
to_drop = ["length_of_time", "desc", "date", "seconds", "minutes", "recorded", "city", "country", "lat", "long", "state" ]

# Drop those features
ufo_dropped = ufo.drop(to_drop, axis=1)

# Let's also filter some words out of the text vector we created
filtered_words = words_to_filter(vocab, vec.vocabulary_, desc_tfidf, 4)

# Take a look at the features in the X set of data
print(X.columns)

# Split the X and y sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Fit knn to the training sets
knn.fit(X_train, y_train)

# Print the score of knn on the test sets
print(knn.score(X_test, y_test))
```