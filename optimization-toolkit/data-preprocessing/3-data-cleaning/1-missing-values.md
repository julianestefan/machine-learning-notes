# Handling Missing Values

Missing values can significantly impact model performance and lead to biased or incorrect predictions.

## Understanding Missing Data

Missing data can occur for various reasons:
- Data entry errors
- Technical issues during data collection
- Non-responses in surveys
- Merging datasets with different structures
- Intentional omission due to privacy/security

## Types of Missing Data

1. **Missing Completely at Random (MCAR)**: No relationship between missing data and any values
2. **Missing at Random (MAR)**: Missing values related to observed data but not to the missing data itself
3. **Missing Not at Random (MNAR)**: Missing values related to the values that would have been observed

## Detection Strategies

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('dataset.csv')

# Checking for missing values
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

# Display missing values count and percentage
missing_data = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percentage
})
print(missing_data[missing_data['Missing Values'] > 0].sort_values('Missing Values', ascending=False))

# Visualize missing data
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Missing Value Heatmap')
plt.tight_layout()
plt.show()

# Check missing value patterns
import missingno as msno
msno.matrix(df)
plt.title('Missing Value Pattern')
plt.show()

# Correlation of missingness
msno.heatmap(df)
plt.title('Correlation of Missing Values')
plt.show()
```

## Handling Techniques

### 1. Deletion Methods

#### List-wise Deletion (Drop Rows)

Remove rows with any missing values:

```python
# Drop rows with any missing values
no_missing_values_rows = df.dropna()
print(f"Original shape: {df.shape}, After dropping rows: {no_missing_values_rows.shape}")
```

#### Column Deletion (Drop Features)

Remove columns with excessive missing values:

```python
# Drop columns with more than 30% missing values
threshold = len(df) * 0.7  # Keep columns with at least 70% non-missing
no_missing_cols = df.dropna(thresh=threshold, axis=1)
print(f"Original columns: {len(df.columns)}, After dropping columns: {len(no_missing_cols.columns)}")
```

#### Targeted Deletion

Remove rows with missing values in specific columns:

```python
# Drop rows with missing values in the 'Gender' column
no_gender_missing = df.dropna(subset=["Gender"])
print(f"Rows with Gender data: {len(no_gender_missing)}")
```

### 2. Imputation Methods

#### Simple Imputation

For categorical data:

```python
# Fill missing categorical values with 'Not Given'
df['Category'] = df['Category'].fillna(value="Not Given")

# Fill with most frequent value (mode)
mode_value = df['Category'].mode()[0]
df['Category'] = df['Category'].fillna(value=mode_value)
```

For continuous data:

```python
# Fill with mean
df['Numeric_Column'].fillna(df['Numeric_Column'].mean(), inplace=True)

# Fill with median (better for skewed distributions)
df['Skewed_Column'].fillna(df['Skewed_Column'].median(), inplace=True)

# Fill with constant value
df['Optional_Column'].fillna(0, inplace=True)
```

#### Advanced Imputation with Scikit-learn

```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Simple mean imputation
mean_imputer = SimpleImputer(strategy='mean')
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = mean_imputer.fit_transform(df[numeric_cols])

# KNN imputation (for related features)
knn_imputer = KNNImputer(n_neighbors=5)
df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])

# MICE (Multiple Imputation by Chained Equations)
mice_imputer = IterativeImputer(max_iter=10, random_state=0)
df[numeric_cols] = mice_imputer.fit_transform(df[numeric_cols])
```

#### Time Series Imputation

For time-based data:

```python
# Forward fill (use previous values)
df['Time_Series'] = df['Time_Series'].fillna(method='ffill')

# Backward fill (use next values)
df['Time_Series'] = df['Time_Series'].fillna(method='bfill')

# Linear interpolation
df['Time_Series'] = df['Time_Series'].interpolate(method='linear')
```

### 3. Missing Value Indicators

Create binary features to mark where values were missing:

```python
# Create indicator features
for col in ['Income', 'Age', 'Education']:
    if df[col].isnull().sum() > 0:
        df[f'{col}_is_missing'] = df[col].isnull().astype(int)
```

## Imputation Method Comparison

| Method | Pros | Cons | Best Used When |
|--------|------|------|---------------|
| Mean/Median | Simple, fast | Distorts distribution | Missing data is minimal |
| Mode | Works for categorical data | May increase bias | Missing at random |
| Constant/Zero | Transparent | Can introduce bias | Value has contextual meaning |
| KNN | Preserves relationships | Computationally expensive | Data has strong correlations |
| MICE | Accounts for uncertainty | Complex to implement | Statistical analysis is primary goal |
| Interpolation | Works well for time series | Assumes linearity | Data has temporal structure |

## Best Practices

1. **Understand the cause** of missing data before choosing a strategy
2. **Document all handling decisions** for reproducibility
3. **Compare multiple strategies** and their impact on model performance
4. **Be cautious with imputation** for critical variables
5. **Consider domain knowledge** when selecting imputation methods
6. **Test sensitivity** of models to different missing data approaches

## Machine Learning Pipeline Implementation

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Define preprocessing for numeric and categorical columns
numeric_features = ['age', 'income', 'score']
categorical_features = ['gender', 'category']

# Define preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create and evaluate the pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

model.fit(X_train, y_train)
``` 