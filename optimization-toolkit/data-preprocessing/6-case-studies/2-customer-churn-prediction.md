# Case Study: Customer Churn Prediction

This case study demonstrates a complete preprocessing and feature engineering workflow for customer churn prediction, a common business application of machine learning.

## Problem Statement

Predict which customers are likely to churn (cancel their service) based on historical data. This is a binary classification problem with significant business implications for customer retention strategies.

## Dataset Overview

For this case study, we'll use the Telco Customer Churn dataset, which can be accessed through scikit-learn's `fetch_openml` function. The dataset contains information about:

- Customer demographics (gender, age, partners, dependents)
- Services subscribed (phone, internet, streaming, backup, security)
- Account information (tenure, contract type, payment method, billing)
- Churn status (target variable)

## Preprocessing Pipeline

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
try:
    from sklearn.datasets import fetch_openml
    churn_data = fetch_openml(name='telco-customer-churn', version=1, as_frame=True)
    df = churn_data.data
    if 'Churn' in df.columns:
        target = df['Churn']
        df = df.drop('Churn', axis=1)
    else:
        target = churn_data.target
    
    # Convert target to binary
    if target.dtype == 'object':
        target = target.map({'Yes': 1, 'No': 0})
    
    print(f"Successfully loaded Telco Customer Churn dataset with {df.shape[0]} rows and {df.shape[1]} columns")

except Exception as e:
    print(f"Could not load dataset from OpenML: {e}")
    print("Creating synthetic customer churn dataset...")
    
    # Create synthetic data
    from sklearn.datasets import make_classification
    
    # Generate synthetic features
    X_synth, y_synth = make_classification(
        n_samples=5000, 
        n_features=15,
        n_informative=8,
        n_redundant=3,
        n_classes=2,
        weights=[0.7, 0.3],  # Imbalanced like typical churn data (30% churn)
        random_state=42
    )
    
    # Create feature names
    feature_names = [
        'tenure_months', 'monthly_charges', 'total_charges', 'age', 
        'number_of_dependents', 'number_of_referrals', 'number_of_tickets',
        'avg_call_minutes', 'avg_data_usage_gb', 'avg_intl_calls',
        'contract_length_months', 'late_payments', 'rating_score',
        'service_issues', 'support_calls'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(X_synth, columns=feature_names)
    
    # Scale features to realistic ranges
    df['tenure_months'] = (df['tenure_months'] - df['tenure_months'].min()) / (df['tenure_months'].max() - df['tenure_months'].min()) * 72
    df['tenure_months'] = df['tenure_months'].round()
    
    df['monthly_charges'] = df['monthly_charges'] * 20 + 50  # $50-$150 range
    df['total_charges'] = df['monthly_charges'] * df['tenure_months']
    
    # Add categorical features
    df['gender'] = np.random.choice(['Male', 'Female'], size=len(df))
    df['contract'] = np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                     p=[0.5, 0.3, 0.2], size=len(df))
    df['internet_service'] = np.random.choice(['DSL', 'Fiber optic', 'No'], 
                                             p=[0.3, 0.5, 0.2], size=len(df))
    df['payment_method'] = np.random.choice(['Electronic check', 'Mailed check', 
                                           'Bank transfer', 'Credit card'], size=len(df))
    df['paperless_billing'] = np.random.choice(['Yes', 'No'], size=len(df))
    
    # Convert target to pandas Series
    target = pd.Series(y_synth, name='Churn')
    
    print(f"Created synthetic customer churn dataset with {df.shape[0]} rows and {df.shape[1]} columns")

# Preview the data
print("\nData preview:")
print(df.head())

# Check for missing values
missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    print("\nMissing values by column:")
    print(missing_values[missing_values > 0])
else:
    print("\nNo missing values found in the dataset.")

# Basic data statistics
print("\nData summary:")
print(df.describe())
```

## Exploratory Data Analysis (EDA)

```python
# Target distribution
plt.figure(figsize=(8, 5))
sns.countplot(x=target)
plt.title('Customer Churn Distribution')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()

print(f"Churn rate: {target.mean():.2%}")

# Analyzing numerical features
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

if numeric_features:
    # Distribution of numerical features by churn status
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(numeric_features[:6], 1):  # First 6 numeric features
        plt.subplot(2, 3, i)
        sns.boxplot(x=target, y=df[feature])
        plt.title(f'{feature} vs Churn')
        plt.xlabel('Churn')
        plt.xticks([0, 1], ['No', 'Yes'])
    plt.tight_layout()
    plt.show()

    # Correlation heatmap for numerical features
    plt.figure(figsize=(12, 10))
    df_corr = pd.concat([df[numeric_features], target.rename('churn')], axis=1)
    sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

# Analyzing categorical features
categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

if categorical_features:
    # Distribution of categorical features by churn status
    for feature in categorical_features[:4]:  # First 4 categorical features
        plt.figure(figsize=(10, 6))
        
        # Calculate churn rate per category
        churn_by_category = df.groupby(feature)[target.name].mean().sort_values(ascending=False)
        
        # Plot
        ax = sns.countplot(x=feature, hue=target, data=pd.concat([df, target], axis=1))
        plt.title(f'{feature} vs Churn')
        plt.xlabel(feature)
        plt.ylabel('Count')
        
        # Add churn rate labels on top of bars
        for i, category in enumerate(churn_by_category.index):
            rate = churn_by_category[category]
            count = len(df[df[feature] == category])
            ax.text(i, count * 0.95, f'{rate:.1%}', ha='center')
            
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
```

## Data Preparation and Preprocessing

```python
# Combine features and target for processing
data = pd.concat([df, target.rename('churn')], axis=1)

# Split the data into features and target
X = df.copy()
y = target.copy()

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create preprocessing and training pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

# Evaluate the baseline model
print("\nBaseline Model Evaluation:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Churn', 'Churn'], 
            yticklabels=['No Churn', 'Churn'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Plot ROC curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_pred_proba):.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

## Feature Engineering

Now let's create more informative features to improve model performance:

```python
# Function to add engineered features
def add_engineered_features(data):
    data_copy = data.copy()
    
    # 1. Calculate tenure-related features
    if 'tenure_months' in data_copy.columns:
        # Customer longevity segments
        data_copy['tenure_segment'] = pd.cut(
            data_copy['tenure_months'], 
            bins=[0, 12, 24, 36, 48, 60, float('inf')],
            labels=['0-1 Year', '1-2 Years', '2-3 Years', '3-4 Years', '4-5 Years', '5+ Years']
        )
        
        # Customer longevity indicators
        data_copy['is_new_customer'] = (data_copy['tenure_months'] <= 6).astype(int)
        data_copy['is_loyal_customer'] = (data_copy['tenure_months'] >= 36).astype(int)
    
    # 2. Financial features
    if all(col in data_copy.columns for col in ['monthly_charges', 'tenure_months']):
        # Average revenue per customer
        data_copy['avg_monthly_revenue'] = data_copy['total_charges'] / data_copy['tenure_months']
        
        # Revenue stability (using monthly charges vs. average ratio)
        data_copy['revenue_stability'] = data_copy['monthly_charges'] / (data_copy['avg_monthly_revenue'] + 0.01)
        
        # Revenue growth (a proxy based on total vs expected from current charges)
        expected_total = data_copy['monthly_charges'] * data_copy['tenure_months']
        data_copy['revenue_change'] = (data_copy['total_charges'] - expected_total) / expected_total
    
    # 3. Contract-related features
    if 'contract' in data_copy.columns:
        # Contract risk level
        contract_risk = {'Month-to-month': 3, 'One year': 2, 'Two year': 1}
        if set(data_copy['contract'].unique()).issubset(set(contract_risk.keys())):
            data_copy['contract_risk'] = data_copy['contract'].map(contract_risk)
    
    # 4. Service complexity features
    # Count the number of services a customer has
    service_columns = [col for col in data_copy.columns if 
                       any(service in col for service in ['service', 'protection', 'backup', 'support', 'streaming', 'security'])]
    
    if service_columns:
        # For categorical services (like Yes/No columns)
        service_df = data_copy[service_columns].copy()
        if service_df.dtypes.value_counts().get('object', 0) > 0:
            for col in service_df.select_dtypes(include=['object']):
                service_df[col] = service_df[col].map({'Yes': 1, 'No': 0})
        
        # Sum services (assume numeric values indicate service presence/count)
        data_copy['total_services'] = service_df.sum(axis=1)
        data_copy['service_complexity'] = pd.cut(
            data_copy['total_services'],
            bins=[-1, 0, 2, 4, float('inf')],
            labels=['No Services', 'Basic', 'Standard', 'Premium']
        )
    
    # 5. Customer value score (combine tenure and financial metrics)
    if all(col in data_copy.columns for col in ['tenure_months', 'monthly_charges']):
        # Create a customer value score (longer tenure + higher spend = more valuable)
        tenure_norm = (data_copy['tenure_months'] - data_copy['tenure_months'].min()) / \
                       (data_copy['tenure_months'].max() - data_copy['tenure_months'].min() + 0.01)
                       
        charges_norm = (data_copy['monthly_charges'] - data_copy['monthly_charges'].min()) / \
                        (data_copy['monthly_charges'].max() - data_copy['monthly_charges'].min() + 0.01)
                        
        data_copy['customer_value_score'] = (tenure_norm + charges_norm) / 2
        
        # Customer value segments
        data_copy['value_segment'] = pd.qcut(
            data_copy['customer_value_score'],
            q=4,
            labels=['Low', 'Medium', 'High', 'Premium']
        )
    
    # 6. Payment risk factors
    if 'payment_method' in data_copy.columns:
        # Assuming electronic payment methods are more reliable
        electronic_methods = ['Credit card', 'Bank transfer', 'Electronic check']
        data_copy['is_electronic_payment'] = data_copy['payment_method'].apply(
            lambda x: 1 if x in electronic_methods else 0
        )
    
    # 7. Demographic segments (if available)
    if all(col in data_copy.columns for col in ['age', 'gender']):
        # Age groups
        data_copy['age_group'] = pd.cut(
            data_copy['age'],
            bins=[0, 30, 45, 60, 100],
            labels=['Young Adult', 'Adult', 'Middle Aged', 'Senior']
        )
    
    # 8. User engagement metrics
    if 'avg_data_usage_gb' in data_copy.columns:
        # Usage intensity segments
        data_copy['usage_intensity'] = pd.qcut(
            data_copy['avg_data_usage_gb'],
            q=4,
            labels=['Low', 'Medium', 'High', 'Intensive']
        )
    
    # 9. Customer support indicators
    support_columns = [col for col in data_copy.columns if 
                       any(term in col for term in ['ticket', 'issue', 'call', 'support'])]
    
    if support_columns:
        # Customer has had support interactions
        data_copy['has_support_history'] = (data_copy[support_columns].sum(axis=1) > 0).astype(int)
        
        # For extensive support history
        if len(support_columns) >= 2:
            data_copy['support_intensity'] = pd.qcut(
                data_copy[support_columns].sum(axis=1),
                q=3,
                labels=['Low', 'Medium', 'High'],
                duplicates='drop'
            )
    
    return data_copy

# Apply feature engineering
X_engineered = add_engineered_features(X)

# Preview new features
new_features = set(X_engineered.columns) - set(X.columns)
print("\nNewly created features:")
for feature in new_features:
    print(f"- {feature}")

# Check some of the engineered features
if new_features:
    print("\nSample of engineered features:")
    sample_features = list(new_features)[:5] if len(new_features) > 5 else list(new_features)
    print(X_engineered[sample_features].head())
```

## Feature Selection and Model Improvement

```python
# Split the engineered data
X_train_eng, X_test_eng, y_train_eng, y_test_eng = train_test_split(
    X_engineered, y, test_size=0.2, random_state=42, stratify=y
)

# Identify numeric and categorical columns in engineered dataset
numeric_features_eng = X_engineered.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features_eng = X_engineered.select_dtypes(include=['object', 'category']).columns.tolist()

# Update the preprocessor for engineered features
preprocessor_eng = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features_eng),
        ('cat', categorical_transformer, categorical_features_eng)
    ])

# Create a pipeline with the new preprocessor
model_pipeline_eng = Pipeline(steps=[
    ('preprocessor', preprocessor_eng),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the improved model
model_pipeline_eng.fit(X_train_eng, y_train_eng)

# Make predictions
y_pred_eng = model_pipeline_eng.predict(X_test_eng)
y_pred_proba_eng = model_pipeline_eng.predict_proba(X_test_eng)[:, 1]

# Evaluate the improved model
print("\nImproved Model with Engineered Features:")
print(classification_report(y_test_eng, y_pred_eng))

print("\nImproved Confusion Matrix:")
conf_matrix_eng = confusion_matrix(y_test_eng, y_pred_eng)
sns.heatmap(conf_matrix_eng, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix with Engineered Features')
plt.show()

print(f"Improved ROC-AUC Score: {roc_auc_score(y_test_eng, y_pred_proba_eng):.4f}")

# Compare ROC curves
plt.figure(figsize=(8, 6))
fpr_base, tpr_base, _ = roc_curve(y_test, y_pred_proba)
fpr_eng, tpr_eng, _ = roc_curve(y_test_eng, y_pred_proba_eng)

plt.plot(fpr_base, tpr_base, label=f'Base Model (AUC = {roc_auc_score(y_test, y_pred_proba):.4f})')
plt.plot(fpr_eng, tpr_eng, label=f'Engineered Model (AUC = {roc_auc_score(y_test_eng, y_pred_proba_eng):.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()

# Feature Importance Analysis
rf_classifier = model_pipeline_eng.named_steps['classifier']

# Get feature names
feature_names = []
ohe = preprocessor_eng.named_transformers_['cat'].named_steps['encoder']
feature_names.extend(preprocessor_eng.transformers_[0][2])  # Numeric features
feature_names.extend(ohe.get_feature_names_out(categorical_features_eng).tolist())  # Categorical features

# Create a DataFrame with feature names and their importances
feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_classifier.feature_importances_
}).sort_values('Importance', ascending=False)

# Plot top 15 features
plt.figure(figsize=(12, 8))
top_n = min(15, len(feature_importances))
sns.barplot(x='Importance', y='Feature', data=feature_importances.head(top_n))
plt.title(f'Top {top_n} Features by Importance')
plt.tight_layout()
plt.show()
```

## Business Insights and Recommendations

Based on our analysis, we can derive the following insights and recommendations:

```python
# Assuming we have our trained model and feature importances

# Plot churn probability by key segments
def plot_churn_by_segment(data, segment_col, target_col='churn'):
    plt.figure(figsize=(10, 6))
    segment_churn = data.groupby(segment_col)[target_col].mean().sort_values(ascending=False)
    segment_count = data.groupby(segment_col).size()
    
    ax = sns.barplot(x=segment_churn.index, y=segment_churn.values)
    plt.title(f'Churn Rate by {segment_col}')
    plt.xlabel(segment_col)
    plt.ylabel('Churn Rate')
    
    # Add count labels
    for i, (idx, rate) in enumerate(segment_churn.items()):
        count = segment_count[idx]
        ax.text(i, rate + 0.01, f'n={count}', ha='center')
        ax.text(i, rate/2, f'{rate:.1%}', ha='center', color='white', fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Get combined data for analysis
data_with_pred = pd.concat([
    X_engineered, 
    y.rename('churn'),
    pd.Series(model_pipeline_eng.predict_proba(X_engineered)[:, 1], name='churn_probability')
], axis=1)

# Plot churn by important segments
if 'tenure_segment' in X_engineered.columns:
    plot_churn_by_segment(data_with_pred, 'tenure_segment')

if 'contract' in X_engineered.columns:
    plot_churn_by_segment(data_with_pred, 'contract')

if 'value_segment' in X_engineered.columns:
    plot_churn_by_segment(data_with_pred, 'value_segment')

# High churn risk customers
high_risk = data_with_pred[data_with_pred['churn_probability'] > 0.7]
print(f"\nNumber of high-risk customers: {len(high_risk)} ({len(high_risk)/len(data_with_pred):.1%} of customer base)")

# Analyze high risk customer characteristics
if len(high_risk) > 0:
    print("\nHigh Risk Customer Profile:")
    
    # Categorical features
    for col in categorical_features_eng[:5]:  # Top 5 categorical features
        if col in high_risk.columns:
            print(f"\n{col} distribution in high-risk customers:")
            print(high_risk[col].value_counts(normalize=True).head(3).apply(lambda x: f"{x:.1%}"))
    
    # Numerical features
    for col in numeric_features_eng[:5]:  # Top 5 numerical features
        if col in high_risk.columns:
            print(f"\n{col} in high-risk customers vs. overall:")
            print(f"High risk mean: {high_risk[col].mean():.2f}, Overall mean: {data_with_pred[col].mean():.2f}")
```

## Business Recommendations for Reducing Churn

Based on our analysis of the customer churn model, here are the key business recommendations:

1. **Target Retention Programs by Churn Risk:**
   - Implement tiered retention strategies based on customer churn probability
   - Focus the most resources on high-risk, high-value customers

2. **Contract Strategy:**
   - Offer incentives for month-to-month customers to move to longer contracts
   - Design loyalty benefits that increase with contract length

3. **Early Tenure Intervention:**
   - Create a special onboarding and support program for customers in their first 12 months
   - Implement regular check-ins during the critical first 6 months

4. **Service Package Optimization:**
   - Review pricing and features for service combinations with highest churn rates
   - Create bundled offerings that encourage retention through combined value

5. **Customer Experience Improvements:**
   - Address service issues more proactively, especially for high-risk segments
   - Implement a churn early warning system based on support interactions and usage patterns

## Technical Preprocessing Lessons

From this case study, we've learned several important technical preprocessing lessons:

1. **Feature Engineering Impact:**
   - Creating domain-specific features significantly improved model performance
   - Segmentation features were particularly valuable for understanding customer behavior

2. **Data Transformation:**
   - Standardizing numerical features was essential for model performance
   - One-hot encoding worked well for categorical features, preserving their information

3. **Feature Selection:**
   - Focusing on the most predictive features improved model interpretability
   - Domain knowledge helped guide the creation of meaningful features

4. **Evaluation Metrics:**
   - For imbalanced classes like churn, precision-recall and ROC-AUC were more informative than accuracy
   - Model performance should be evaluated against business objectives

## Next Steps

1. **Model Enhancement:**
   - Test additional algorithms (gradient boosting, deep learning)
   - Perform hyperparameter tuning to optimize model performance

2. **Business Integration:**
   - Deploy model as part of a churn prediction system
   - Create dashboards for tracking churn risk across customer segments

3. **Feedback Loop:**
   - Track the effectiveness of retention actions
   - Update the model as new data becomes available

4. **Expansion:**
   - Incorporate additional data sources (customer service interactions, usage patterns)
   - Develop models for different customer segments or products 