# Explainable AI

How do we explain decision taken by AI to get the output. Some models like decision tree has a transparent process to do that while neural networks hasn't

```python 
model = DecisionTreeClassifier(random_state=42, max_depth=2)
model.fit(X_train, y_train)

# Extract the rules
rules = export_text(model, feature_names= list(X_train.columns))
print(rules)

y_pred = model.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_pred, y_test)
print(f"Accuracy: {accuracy:.2f}")
```
output:

```shell
|--- education_num <= 12.50
|   |--- age <= 33.50
|   |   |--- class:  <=50K
|   |--- age >  33.50
|   |   |--- class:  <=50K
|--- education_num >  12.50
|   |--- age <= 29.50
|   |   |--- class:  <=50K
|   |--- age >  29.50
|   |   |--- class:  >50K

Accuracy: 0.78
```

```python 
model = MLPClassifier(hidden_layer_sizes=(36, 12), random_state=42)
# Train the MLPClassifier
model.fit(X_train,y_train)

# Derive the predictions on the test set
y_pred = model.predict(X_test)

# Compute the test accuracy
accuracy = accuracy_score(y_pred, y_test)
print(f"Accuracy: {accuracy:.2f}")
```
output:
```shell
Accuracy: 0.80
```

MLPClassifier's accuracy is usually higher than decision tree, but unlike decision trees, it's harder to interpret how the decisions are made.

# Linear models

Use coefficients to get the effect of each feature. Features should be normalized to keep them comparable.

```python 
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Derive coefficients
coefficients = model.coef_[0]
feature_names = X_train.columns

# Plot coefficients
plt.bar(feature_names, coefficients)
plt.show()s
```

# Tree based 

use feature importance instead fo coefficients.

```python 
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Derive feature importances
feature_importances = model.feature_importances_
feature_names = X_train.columns

# Plot the feature importances
plt.barh(feature_names, feature_importances)
plt.show()
```