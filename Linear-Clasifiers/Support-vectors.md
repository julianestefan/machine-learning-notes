# Support vectors 

## Definition
* a training example not in the flat part of  loss diagram
* an example incorrectly classified or close to the boundary

## Properties

If an example is not a support vector removing it has no effect
Having a small number of support vectors makes kernel SVMs really fast 

### Example

Compare the decision boundaries of the two trained models. They should be the same

```python
# Train a linear SVM
svm = SVC(kernel="linear")
svm.fit(X, y)
plot_classifier(X, y, svm, lims=(11,15,0,6))

# Make a new data set keeping only the support vectors
print("Number of original examples", len(X))
print("Number of support vectors", len(svm.support_))
X_small = X[svm.support_]
y_small = y[svm.support_]

# Train a new SVM using only the support vectors
svm_small = SVC(kernel="linear")
svm_small.fit(X_small, y_small)
plot_classifier(X_small, y_small, svm_small, lims=(11,15,0,6))
```
