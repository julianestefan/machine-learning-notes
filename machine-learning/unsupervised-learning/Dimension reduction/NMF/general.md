When you left 20 hours ago, you worked on Discovering Interpretable Features, chapter 4 of the course Unsupervised Learning in Python. Here is what you covered in your last lesson:

You learned about Non-negative Matrix Factorization (NMF), a technique for dimensionality reduction that is particularly useful for finding interpretable parts in data. Key points covered include:

NMF Basics: NMF works by breaking down datasets into components and features, making it excellent for applications like topic modeling in text or pattern identification in images.

Application on Documents: By applying NMF to a collection of documents, you discovered that its components can represent topics within these documents. For example, using the NMF model, you identified topics based on word frequencies across documents.

import pandas as pd
components_df = pd.DataFrame(model.components_, columns=words)
print(components_df.iloc[3].nlargest())
Application on Images: You explored how NMF can decompose images into commonly occurring patterns. This was demonstrated with a dataset of LED digit images, where NMF identified the parts of the images that make up the digits.

Comparison with PCA: Unlike NMF, Principal Component Analysis (PCA) does not learn interpretable parts of data. This distinction was highlighted by applying both NMF and PCA to the same dataset of LED digit images and observing the components each method produced.

Throughout these exercises, you gained hands-on experience with NMF by applying it to both text and image data, learning how to interpret the components it produces, and contrasting its capabilities with those of PCA.

The goal of the next lesson is to explore how Non-negative Matrix Factorization (NMF) can be utilized for creating recommender systems that suggest articles and music based on user preferences.

```python 
# Import NMF
from sklearn.decomposition import NMF

# Create an NMF instance: model
model = NMF(n_components=6)

# Fit the model to articles
model.fit(articles)

# Transform the articles: nmf_features
nmf_features = model.transform(articles)

# Print the NMF features
```

Nmf features * components allow reconstruction of values
