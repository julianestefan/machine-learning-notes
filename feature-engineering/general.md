# Feature engineering

## Understand your data 

``` python 
# Create subset of only the numeric columns
so_numeric_df = so_survey_df.select_dtypes(include=['int', 'float'])

# Print the column names contained in so_survey_df_num
print(so_numeric_df.columns)
```

## Categorical data

### One hot encoding

Create n variables for each category. More explainable but it creates duplicate data.

```python 
# Convert the Country column to a one hot encoded Data Frame
one_hot_encoded = pd.get_dummies(so_survey_df, columns=['Country'], prefix='OH')

# Print the columns names
print(one_hot_encoded.columns)
```

### Dummy encoding

Create `n - 1` variables.It drops first category

```python 
# Create dummy variables for the Country column
dummy = pd.get_dummies(so_survey_df, columns=["Country"], drop_first=True, prefix='DM')

# Print the columns names
print(dummy.columns)
```

### Limit columns

Bot could create too many columns. You may limit the amount of created columns

```python 
# Create a series out of the Country column
countries = so_survey_df["Country"]

# Get the counts of each category
country_counts = countries.value_counts()

# Create a mask for only categories that occur less than 10 times
mask = countries.isin(country_counts[country_counts < 10].index)

# Label all other categories as Other
countries[mask] = 'Other'
```
## Numeric features

### turn into binary

```python 
# Create the Paid_Job column filled with zeros
so_survey_df["Paid_Job"] = 0

# Replace all the Paid_Job values where ConvertedSalary is > 0
so_survey_df.loc[so_survey_df["ConvertedSalary"] > 0, 'Paid_Job'] = 1

# Print the first five rows of the columns
print(so_survey_df[['Paid_Job', 'ConvertedSalary']].head())
```

### turn into bins

```python 
# Import numpy
import numpy as np

# Specify the boundaries of the bins
bins = [-np.inf, 10000, 50000, 100000, 150000, np.inf]

# Bin labels
labels = ['Very low', 'Low', 'Medium', 'High', 'Very high']

# Bin the continuous variable ConvertedSalary using these boundaries
so_survey_df['boundary_binned'] = pd.cut(so_survey_df['ConvertedSalary'], 
                                         bins=bins, labels=labels)

# Print the first 5 rows of the boundary_binned column
print(so_survey_df[['boundary_binned', 'ConvertedSalary']].head())
```
