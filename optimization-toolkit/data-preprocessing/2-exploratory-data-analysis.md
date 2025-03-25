# Exploratory Data Analysis (EDA)

Exploratory Data Analysis is the critical first step in the data preprocessing pipeline, allowing you to understand your dataset before applying any transformations.

## Goals of EDA

* Understand the structure and characteristics of your data
* Identify patterns, relationships, and anomalies
* Guide feature engineering and preprocessing decisions
* Generate hypotheses for further investigation
* Detect data quality issues early

## Basic Data Exploration

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('dataset.csv')

# Basic information
print("Dataset shape:", df.shape)
print("\nData types and non-null counts:")
print(df.info())

# View first few rows
print("\nFirst 5 rows:")
print(df.head())

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Count unique values in categorical columns
print("\nUnique values in categorical columns:")
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    print(f"{col}: {df[col].nunique()} unique values")
    print(df[col].value_counts().head(5))
    print()
```

## Missing Value Analysis

Identifying and quantifying missing data:

```python
# Check for missing values
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

# Display missing values summary
missing_data = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percentage
})
print("Missing data summary:")
print(missing_data[missing_data['Missing Values'] > 0].sort_values('Missing Values', ascending=False))

# Visualize missing values pattern
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Missing Value Heatmap')
plt.tight_layout()
plt.show()
```

## Numerical Data Analysis

Exploring distributions and relationships of numerical features:

```python
# Select numerical columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Distribution plots for numerical features
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols[:9], 1):  # Limit to first 9 columns
    plt.subplot(3, 3, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
plt.show()

# Box plots for outlier detection
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols[:9], 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=df[col])
    plt.title(f'Box Plot of {col}')
    plt.tight_layout()
plt.show()

# Calculate skewness and kurtosis
skew_data = pd.DataFrame({
    'Skewness': df[numeric_cols].skew(),
    'Kurtosis': df[numeric_cols].kurtosis()
})
print("Skewness and Kurtosis:")
print(skew_data)

# Identify highly skewed features (|skew| > 1)
print("\nHighly skewed features:")
print(skew_data[abs(skew_data['Skewness']) > 1].sort_values('Skewness', ascending=False))
```

## Categorical Data Analysis

Exploring categorical features:

```python
# Analyze categorical features
for col in categorical_cols:
    plt.figure(figsize=(12, 6))
    
    # Handle high cardinality (limit to top 10 categories)
    if df[col].nunique() > 10:
        top_categories = df[col].value_counts().nlargest(10).index
        filtered_col = df[col].copy()
        filtered_col[~filtered_col.isin(top_categories)] = 'Other'
        sns.countplot(y=filtered_col, order=filtered_col.value_counts().index)
        plt.title(f'Top 10 Categories in {col}')
    else:
        sns.countplot(y=df[col], order=df[col].value_counts().index)
        plt.title(f'Distribution of {col}')
    
    plt.tight_layout()
    plt.show()
```

## Relationship Analysis

Exploring relationships between features:

```python
# Correlation matrix for numerical features
correlation_matrix = df[numeric_cols].corr()

# Visualize correlations
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Feature Correlation Matrix')
plt.show()

# Find highly correlated feature pairs (above 0.8)
def find_high_correlations(correlation_matrix, threshold=0.8):
    corr_pairs = []
    # Get the upper triangle of the correlation matrix
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    
    # Find feature pairs with correlation greater than threshold
    for col in upper_tri.columns:
        high_corr = upper_tri[col][abs(upper_tri[col]) > threshold].index.tolist()
        for feature in high_corr:
            corr_pairs.append((col, feature, correlation_matrix.loc[col, feature]))
    
    return corr_pairs

high_correlations = find_high_correlations(correlation_matrix, 0.8)
print("Highly correlated feature pairs:")
for feat1, feat2, corr in high_correlations:
    print(f"{feat1} and {feat2}: {corr:.2f}")

# Scatter plot matrix for selected numerical features
if len(numeric_cols) > 4:
    selected_numeric = numeric_cols[:4]  # Select first 4 numerical features
else:
    selected_numeric = numeric_cols

sns.pairplot(df[selected_numeric])
plt.suptitle('Scatter Plot Matrix', y=1.02)
plt.show()
```

## Target Variable Analysis

For supervised learning problems:

```python
# Assuming 'target' is the target variable
if 'target' in df.columns:
    target_col = 'target'
    
    # Analyze target distribution
    plt.figure(figsize=(10, 6))
    if df[target_col].dtype == 'object' or df[target_col].nunique() < 10:
        # For categorical target
        sns.countplot(y=df[target_col])
        plt.title('Target Distribution')
    else:
        # For numerical target
        sns.histplot(df[target_col], kde=True)
        plt.title('Target Distribution')
    plt.tight_layout()
    plt.show()
    
    # Feature relationship with target
    for col in numeric_cols:
        if col != target_col:
            plt.figure(figsize=(10, 6))
            
            if df[target_col].dtype == 'object' or df[target_col].nunique() < 10:
                # Categorical target
                sns.boxplot(x=df[target_col], y=df[col])
                plt.title(f'Relationship between {target_col} and {col}')
            else:
                # Numerical target
                sns.scatterplot(x=df[col], y=df[target_col])
                plt.title(f'Relationship between {col} and {target_col}')
            
            plt.tight_layout()
            plt.show()
```

## Time Series Analysis

For time-based data:

```python
# Convert to datetime if date column exists
date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]

if date_cols:
    date_col = date_cols[0]  # Use the first date column found
    
    # Convert to datetime if not already
    if df[date_col].dtype != 'datetime64[ns]':
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Sort by date
    df_sorted = df.sort_values(date_col)
    
    # Plot time series for a numeric column
    if len(numeric_cols) > 0:
        plt.figure(figsize=(12, 6))
        plt.plot(df_sorted[date_col], df_sorted[numeric_cols[0]])
        plt.title(f'Time Series of {numeric_cols[0]}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    # Analyze distribution by time components
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    
    # Monthly pattern
    plt.figure(figsize=(12, 6))
    sns.countplot(x='month', data=df, order=range(1, 13))
    plt.title('Monthly Distribution')
    plt.show()
```

## EDA Checklist

- [ ] Basic data overview (shape, types)
- [ ] Summary statistics
- [ ] Missing value analysis
- [ ] Duplicate records check
- [ ] Distribution analysis for numerical features
- [ ] Category analysis for categorical features
- [ ] Correlation analysis
- [ ] Outlier detection
- [ ] Target variable analysis
- [ ] Time series patterns (if applicable)

## EDA Best Practices

1. **Start broad, then drill down**: Begin with an overview, then explore specific areas of interest
2. **Visualize extensively**: Use appropriate visualizations for different data types
3. **Document findings**: Record insights and anomalies for later preprocessing steps
4. **Generate hypotheses**: Use EDA to form hypotheses about relationships in the data
5. **Iterate**: As you discover insights, refine your exploration 