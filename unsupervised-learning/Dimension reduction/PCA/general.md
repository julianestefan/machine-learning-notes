You learned about Principal Component Analysis (PCA), a fundamental technique for dimension reduction that summarizes a dataset by identifying patterns and re-expressing it in a compressed form. This is crucial for making computations more efficient and for reducing datasets to their essential features, which is especially beneficial in supervised learning tasks. Key points covered include:

PCA Transformation: PCA involves two main steps. The first step, de-correlation, rotates and shifts the samples so they align with the coordinate axes and have a mean of zero, without changing the data's dimension. This process ensures that no information is lost, regardless of the dataset's size.

Principal Components: The principal components, which are the directions along which the data varies the most, are aligned with the coordinate axes by PCA. These components are available in the components_ attribute after fitting a PCA model.

``` python 
# Import PCA
from sklearn.decomposition import PCA

# Create PCA instance: model
model = PCA()

# Apply the fit_transform method of model to grains: pca_features
pca_features = model.fit_transform(grains)

# Assign 0th column of pca_features: xs
xs = pca_features[:,0]

# Assign 1st column of pca_features: ys
ys = pca_features[:,1]

# Scatter plot xs vs ys
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation of xs and ys
correlation, pvalue = pearsonr(xs, ys)

# Display the correlation
print(correlation)
```

Using PCA to perform dimensional reduction 

```python
from sklearn.decomposition import PCA

# Create a PCA model with 2 components: pca
pca = PCA(n_components=2)

# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)

# Transform the scaled samples: pca_features
pca_features = pca.transform(scaled_samples)

# Print the shape of pca_features
print(pca_features.shape)

```

TDF-IDF processing

[WIkipedia](https://es.wikipedia.org/wiki/Tf-idf)

```python
# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TfidfVectorizer: tfidf
tfidf = TfidfVectorizer() 

# Apply fit_transform to document: csr_mat
csr_mat = tfidf.fit_transform(documents)

# Print result of toarray() method
print(csr_mat.toarray())

# Get the words: words
words = tfidf.get_feature_names()

# Print words
print(words)
```

TruncatedSVD is able to perform PCA on sparse arrays in csr_matrix format, such as word-frequency arrays. 

``` python 
# Perform the necessary imports
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
import pandas as pd

# Create a TruncatedSVD instance: svd
svd = TruncatedSVD(n_components=50)

# Create a KMeans instance: kmeans
kmeans = KMeans(n_clusters=6)

# Create a pipeline: pipeline
pipeline = make_pipeline(svd, kmeans)

# Fit the pipeline to articles
pipeline.fit(articles)

# Calculate the cluster labels: labels
labels = pipeline.predict(articles)

# Create a DataFrame aligning labels and titles: df
df = pd.DataFrame({'label': labels, 'article': titles})

# Display df sorted by cluster label
print(df.sort_values(['label']))
```
