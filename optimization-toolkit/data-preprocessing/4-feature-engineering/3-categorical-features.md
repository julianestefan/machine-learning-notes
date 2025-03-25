# Categorical Feature Engineering

Categorical feature engineering involves transforming and creating features from non-numeric categorical data to improve model performance.

## Types of Categorical Variables

1. **Nominal**: Categories without any natural order (e.g., colors, countries)
2. **Ordinal**: Categories with a meaningful order (e.g., education levels, satisfaction ratings)
3. **Binary**: Categories with only two values (e.g., yes/no, true/false)
4. **Cyclic**: Categories with a circular relationship (e.g., days of week, months)

## Basic Encoding Techniques

### Label Encoding

Converts categorical values into numeric labels. Best for ordinal categories.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder object
label_encoder = LabelEncoder()

# Apply label encoding
df['education_encoded'] = label_encoder.fit_transform(df['education'])

# View the mapping
mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Mapping:", mapping)

# For ordered categories, ensure the order is preserved
education_order = ['High School', 'Bachelor', 'Master', 'PhD']
df['education_ordered'] = df['education'].map({val: i for i, val in enumerate(education_order)})
```

### One-Hot Encoding

Creates binary columns for each category. Best for nominal categories.

```python
# Method 1: Using pandas get_dummies
one_hot_encoded = pd.get_dummies(df['country'], prefix='country')

# Join with original dataframe
df = pd.concat([df, one_hot_encoded], axis=1)

# Method 2: Using scikit-learn
from sklearn.preprocessing import OneHotEncoder

# Initialize encoder
encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' removes first category to avoid multicollinearity

# Fit and transform
X_encoded = encoder.fit_transform(df[['country']])

# Convert to DataFrame with proper column names
encoded_df = pd.DataFrame(
    X_encoded,
    columns=encoder.get_feature_names_out(['country']),
    index=df.index
)

# Join with original dataframe
df = pd.concat([df, encoded_df], axis=1)
```

### Dummy Encoding

Similar to one-hot but drops one category to avoid multicollinearity:

```python
# Using pandas get_dummies with drop_first=True
dummy_encoded = pd.get_dummies(df['country'], prefix='country', drop_first=True)

# Join with original dataframe
df = pd.concat([df, dummy_encoded], axis=1)
```

### Binary Encoding

Encodes categories using binary digits, useful for high cardinality:

```python
# Install if not available: pip install category_encoders
import category_encoders as ce

# Initialize binary encoder
binary_encoder = ce.BinaryEncoder(cols=['country'])

# Fit and transform
binary_encoded = binary_encoder.fit_transform(df[['country']])

# Join with original dataframe
df = pd.concat([df.drop('country', axis=1), binary_encoded], axis=1)
```

## Advanced Encoding Techniques

### Target Encoding

Replaces categories with the mean of the target variable for each category:

```python
# Group by category and calculate mean of target
category_means = df.groupby('category')['target'].mean()

# Map means to original data
df['category_target_encoded'] = df['category'].map(category_means)

# With smoothing to prevent overfitting
def target_encode_smooth(df, column, target, alpha=10):
    # Calculate global mean
    global_mean = df[target].mean()
    
    # Calculate category metrics
    aggr = df.groupby(column).agg({target: ['mean', 'count']})
    counts = aggr[(target, 'count')]
    means = aggr[(target, 'mean')]
    
    # Apply smoothing
    smooth = (counts * means + alpha * global_mean) / (counts + alpha)
    
    # Map to original data
    return df[column].map(smooth)

df['category_smooth_encoded'] = target_encode_smooth(df, 'category', 'target')
```

### Frequency Encoding

Replaces categories with their frequency:

```python
# Calculate frequency of each category
category_counts = df['category'].value_counts(normalize=True)

# Map frequencies to original data
df['category_frequency'] = df['category'].map(category_counts)
```

### Weight of Evidence (WOE) Encoding

Used primarily in classification tasks:

```python
# Calculate good/bad ratio for binary target
def woe_encoding(df, column, target, positive_class=1):
    # Create crosstab of category vs target
    cross_tab = pd.crosstab(df[column], df[target])
    
    # Calculate good and bad counts
    good = cross_tab[positive_class]
    bad = cross_tab[1 - positive_class]
    
    # Calculate WOE
    woe = np.log((good / sum(good)) / (bad / sum(bad)))
    
    # Map WOE to original data
    return df[column].map(woe)

df['category_woe'] = woe_encoding(df, 'category', 'binary_target')
```

### Mean Encoding with Cross-Validation

Prevents target leakage when using target encoding:

```python
from sklearn.model_selection import KFold

def target_encode_cv(df, column, target, n_folds=5):
    # Create a copy of the column
    df_encoded = df.copy()
    df_encoded[f'{column}_target_enc'] = np.nan
    
    # Set up KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # For each fold, train on out-of-fold data
    for train_idx, test_idx in kf.split(df):
        # Calculate target mean on training data
        target_means = df.iloc[train_idx].groupby(column)[target].mean()
        
        # Apply to test data
        df_encoded.iloc[test_idx, df_encoded.columns.get_loc(f'{column}_target_enc')] = \
            df.iloc[test_idx][column].map(target_means)
    
    # Fill missing values (for categories not seen in some folds)
    global_mean = df[target].mean()
    df_encoded[f'{column}_target_enc'].fillna(global_mean, inplace=True)
    
    return df_encoded[f'{column}_target_enc']

df['category_target_enc_cv'] = target_encode_cv(df, 'category', 'target')
```

## Handling High Cardinality

When a categorical feature has too many unique values:

```python
# For one-hot encoding, limit to top N categories
def limit_categories(series, top_n=10, other_label='Other'):
    """Limit categories to top N by frequency"""
    top_categories = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top_categories), other_label)

# Apply to high cardinality column
df['limited_category'] = limit_categories(df['high_cardinality_column'])

# One-hot encode the limited categories
limited_one_hot = pd.get_dummies(df['limited_category'], prefix='lim_cat')
df = pd.concat([df, limited_one_hot], axis=1)
```

## Encoding for Tree-Based Models

Some models handle categories differently:

```python
# For tree-based models like Random Forest or XGBoost
# Label encoding often works well even for nominal categories

# Apply label encoding to all categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    df[f'{col}_label'] = LabelEncoder().fit_transform(df[col])

# For XGBoost with categorical_columns parameter
# (Available in newer versions)
import xgboost as xgb

# Ensure categories are in integer type (required by XGBoost)
for col in categorical_cols:
    df[col] = df[col].astype('category').cat.codes

# Get categorical feature indices
cat_indices = [i for i, col in enumerate(df.columns) if col in categorical_cols]

# Train model with categorical features
model = xgb.XGBClassifier(
    tree_method='hist',  # Required for categorical features
    enable_categorical=True  # Enable categorical data support
)
model.fit(
    df.drop('target', axis=1), 
    df['target'],
    categorical_feature=cat_indices  # Specify which features are categorical
)
```

## Feature Crossing

Combining multiple categorical features:

```python
# Simple concatenation
df['country_gender'] = df['country'].astype(str) + '_' + df['gender'].astype(str)

# Count encoding of crossed features
crossed_counts = df.groupby(['country', 'gender']).size()
crossed_counts = crossed_counts / len(df)  # Normalize to get frequencies

# Map back to original data
df['country_gender_freq'] = df.apply(lambda row: crossed_counts.get((row['country'], row['gender']), 0), axis=1)

# One-hot encoding of crossed features
crossed_one_hot = pd.get_dummies(df['country_gender'], prefix='country_gender')
df = pd.concat([df, crossed_one_hot], axis=1)
```

## Encoded Feature Visualization

Visualizing the impact of encodings:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize relationship between target and encoded feature
plt.figure(figsize=(12, 6))
sns.boxplot(x='category', y='target', data=df)
plt.title('Target vs Raw Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualize target encoding
plt.figure(figsize=(12, 6))
plt.scatter(df['category_target_encoded'], df['target'])
plt.xlabel('Target Encoded Category')
plt.ylabel('Target')
plt.title('Target vs Target Encoded Category')
plt.tight_layout()
plt.show()

# Visualize WOE encoding for binary classification
if 'category_woe' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df['category'].unique(), y=[df[df['category']==cat]['category_woe'].iloc[0] for cat in df['category'].unique()])
    plt.title('Weight of Evidence by Category')
    plt.xlabel('Category')
    plt.ylabel('WOE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
```

## Encoding Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| Label Encoding | Simple, memory-efficient | Implies ordering | Ordinal categories |
| One-Hot Encoding | No ordering implied | Creates many columns | Nominal categories with few values |
| Dummy Encoding | Avoids multicollinearity | Reference category needed | Most general cases |
| Binary Encoding | Space efficient | Less interpretable | High cardinality with no ordering |
| Target Encoding | Captures relationship with target | Risk of overfitting | High cardinality with strong signal |
| Frequency Encoding | Simple, informative | Loses category identity | High cardinality with frequency patterns |
| WOE | Directly shows predictive power | Only for binary targets | Credit scoring, risk models |

## Handling Missing Values in Categorical Features

Special treatment for missing values:

```python
# Create a missing indicator
df['category_missing'] = df['category'].isna().astype(int)

# Fill missing with a new category
df['category_filled'] = df['category'].fillna('Missing')

# For one-hot encoding, allow NaN to be its own category
one_hot_with_na = pd.get_dummies(df['category'], prefix='cat', dummy_na=True)
df = pd.concat([df, one_hot_with_na], axis=1)
```

## Time-Based Categorical Features

Handling cyclical features like days, months:

```python
# Convert day of week to cyclic features
df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Convert month to cyclic features
df['month'] = df['date'].dt.month
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Visualize circular nature of features
plt.figure(figsize=(8, 8))
plt.scatter(df['month_cos'], df['month_sin'])
plt.title('Circular Encoding of Months')
plt.xlabel('Cosine Component')
plt.ylabel('Sine Component')
plt.axis('equal')
plt.grid(True)
plt.show()
```

## Evaluation of Categorical Encodings

Compare the impact of different encodings on model performance:

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Define test function
def evaluate_encoding(X, y, cv=5):
    model = RandomForestClassifier(random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    return scores.mean(), scores.std()

# Original data with label encoding
X_label = df.drop('target', axis=1)
X_label['category_label'] = label_encoder.fit_transform(df['category'])
label_score, label_std = evaluate_encoding(X_label, df['target'])

# One-hot encoding
X_onehot = df.drop('target', axis=1)
X_onehot = pd.concat([X_onehot, pd.get_dummies(df['category'], prefix='cat')], axis=1)
onehot_score, onehot_std = evaluate_encoding(X_onehot, df['target'])

# Target encoding
X_target = df.drop('target', axis=1)
X_target['category_target'] = target_encode_cv(df, 'category', 'target')
target_score, target_std = evaluate_encoding(X_target, df['target'])

# Compare results
print(f"Label encoding: AUC = {label_score:.4f} (±{label_std:.4f})")
print(f"One-hot encoding: AUC = {onehot_score:.4f} (±{onehot_std:.4f})")
print(f"Target encoding: AUC = {target_score:.4f} (±{target_std:.4f})")

# Visualize results
plt.figure(figsize=(10, 6))
plt.bar(['Label', 'One-hot', 'Target'], [label_score, onehot_score, target_score], yerr=[label_std, onehot_std, target_std])
plt.title('Performance Comparison of Encoding Methods')
plt.ylabel('AUC Score')
plt.ylim(0.5, 1.0)
plt.tight_layout()
plt.show()
```

## Best Practices for Categorical Feature Engineering

1. **Choose encoding based on category type**: Ordinal vs. nominal vs. binary
2. **Handle high cardinality** by grouping, limiting or using advanced encoding techniques
3. **Watch for leakage** when using target-based encodings
4. **Encode missing values** explicitly rather than dropping them
5. **Consider feature crossing** for interacting categorical variables
6. **Use cross-validation** to prevent overfitting with target-based encoding
7. **Document mappings** for interpretability and deployment
8. **For tree-based models**, simpler encodings often work well (label encoding)
9. **For linear models**, one-hot or similar encodings are typically needed
10. **Apply encodings consistently** between training and prediction phases 