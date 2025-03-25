# Image Feature Engineering

Feature engineering for image data involves extracting meaningful representations from visual information to improve model performance. Unlike text or tabular data, images require specialized techniques to capture spatial relationships, patterns, and visual characteristics.

## Basic Image Feature Extraction

### Color-Based Features

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import feature, color
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load sample image
image = cv2.imread('sample_image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Function to extract color histograms
def extract_color_histogram(image, bins=32):
    """Extract color histogram features from an image."""
    # Split the image into three channels
    channels = cv2.split(image)
    features = []
    
    # Calculate histogram for each channel
    for channel, color in zip(channels, ['r', 'g', 'b']):
        hist = cv2.calcHist([channel], [0], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    
    return np.array(features)

# Extract color histogram features
color_features = extract_color_histogram(image_rgb)
print(f"Color histogram features shape: {color_features.shape}")

# Visualize color histogram
plt.figure(figsize=(12, 4))
colors = ('r', 'g', 'b')
for i, color in enumerate(colors):
    hist = color_features[i*32:(i+1)*32]
    plt.subplot(1, 3, i+1)
    plt.bar(range(32), hist, color=color)
    plt.title(f'{color.upper()} Channel Histogram')
    plt.xlim([0, 32])
plt.tight_layout()
plt.show()

# Extract color statistics
def extract_color_stats(image):
    """Extract statistical features from color channels."""
    features = []
    
    # Split the image into channels
    channels = cv2.split(image)
    
    # For each channel, calculate statistics
    for channel in channels:
        features.append(np.mean(channel))  # Mean
        features.append(np.std(channel))   # Standard deviation
        features.append(np.percentile(channel, 25))  # 25th percentile
        features.append(np.median(channel))  # Median
        features.append(np.percentile(channel, 75))  # 75th percentile
        features.append(np.max(channel) - np.min(channel))  # Range
    
    return np.array(features)

# Extract color statistics
color_stats = extract_color_stats(image_rgb)
print(f"Color statistics features: {color_stats}")
print(f"Color statistics feature names: ['r_mean', 'r_std', 'r_25th', 'r_median', 'r_75th', 'r_range', "
      f"'g_mean', 'g_std', 'g_25th', 'g_median', 'g_75th', 'g_range', "
      f"'b_mean', 'b_std', 'b_25th', 'b_median', 'b_75th', 'b_range']")
```

### Texture Features

Texture features capture patterns and regularities in an image:

```python
# Convert image to grayscale for texture analysis
gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

# Extract Haralick texture features
def extract_haralick_features(gray_image):
    """Extract Haralick texture features from grayscale image."""
    textures = feature.graycomatrix(
        gray_image, 
        distances=[1, 3, 5], 
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
        levels=256,
        symmetric=True, 
        normed=True
    )
    
    # Calculate properties from the GLCM
    contrast = feature.graycoprops(textures, 'contrast')
    dissimilarity = feature.graycoprops(textures, 'dissimilarity')
    homogeneity = feature.graycoprops(textures, 'homogeneity')
    energy = feature.graycoprops(textures, 'energy')
    correlation = feature.graycoprops(textures, 'correlation')
    ASM = feature.graycoprops(textures, 'ASM')
    
    # Flatten the features
    features = np.hstack([
        contrast.flatten(), dissimilarity.flatten(),
        homogeneity.flatten(), energy.flatten(),
        correlation.flatten(), ASM.flatten()
    ])
    
    return features

# Extract texture features
texture_features = extract_haralick_features(gray_image)
print(f"Texture features shape: {texture_features.shape}")

# Extract Local Binary Patterns (LBP)
def extract_lbp_features(gray_image, num_points=24, radius=3, method='uniform'):
    """Extract Local Binary Pattern features."""
    lbp = feature.local_binary_pattern(gray_image, num_points, radius, method)
    
    # Calculate the histogram of LBP
    n_bins = num_points + 2 if method == 'uniform' else 2**num_points
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    
    # Normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
    return hist

# Extract LBP features
lbp_features = extract_lbp_features(gray_image)
print(f"LBP features shape: {lbp_features.shape}")

# Visualize LBP
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Grayscale Image')
plt.subplot(1, 2, 2)
lbp = feature.local_binary_pattern(gray_image, 24, 3, 'uniform')
plt.imshow(lbp, cmap='gray')
plt.title('LBP Visualization')
plt.tight_layout()
plt.show()
```

### Shape Features

Contour and shape descriptors for objects in images:

```python
# Simple edge detection for shape analysis
def extract_shape_features(gray_image):
    """Extract basic shape features from image."""
    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, 100, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours are found, return zeros
    if len(contours) == 0:
        return np.zeros(5)
    
    features = []
    
    # Area of the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    features.append(area)
    
    # Perimeter of the largest contour
    perimeter = cv2.arcLength(largest_contour, True)
    features.append(perimeter)
    
    # Circularity: 4*pi*area/perimeter^2
    circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
    features.append(circularity)
    
    # Aspect ratio of the bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    features.append(aspect_ratio)
    
    # Extent: ratio of contour area to bounding rectangle area
    rect_area = w * h
    extent = float(area) / rect_area if rect_area > 0 else 0
    features.append(extent)
    
    return np.array(features)

# Extract shape features
shape_features = extract_shape_features(gray_image)
print(f"Shape features: {shape_features}")
print(f"Shape feature names: ['area', 'perimeter', 'circularity', 'aspect_ratio', 'extent']")

# Visualize the edges used for shape analysis
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Grayscale Image')
plt.subplot(1, 2, 2)
edges = cv2.Canny(gray_image, 100, 200)
plt.imshow(edges, cmap='gray')
plt.title('Edges for Shape Analysis')
plt.tight_layout()
plt.show()
```

## Advanced Image Feature Extraction

### HOG (Histogram of Oriented Gradients)

HOG is particularly useful for object detection:

```python
# Extract HOG features
def extract_hog_features(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """Extract Histogram of Oriented Gradients (HOG) features."""
    # Resize image to a standard size for consistent feature extraction
    resized = cv2.resize(gray_image, (128, 128))
    
    # Calculate HOG features
    hog_features = feature.hog(
        resized, 
        orientations=orientations, 
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block, 
        block_norm='L2-Hys', 
        visualize=False
    )
    
    return hog_features

# Extract HOG features
hog_features = extract_hog_features(gray_image)
print(f"HOG features shape: {hog_features.shape}")

# Visualize HOG
_, hog_image = feature.hog(
    cv2.resize(gray_image, (128, 128)), 
    orientations=9, 
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2), 
    block_norm='L2-Hys', 
    visualize=True
)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Grayscale Image')
plt.subplot(1, 2, 2)
plt.imshow(hog_image, cmap='gray')
plt.title('HOG Visualization')
plt.tight_layout()
plt.show()
```

### SIFT and SURF Features

SIFT (Scale-Invariant Feature Transform) and SURF (Speeded-Up Robust Features) are useful for identifying local features:

```python
# Note: OpenCV SIFT/SURF may require additional installation or usage
# Below is a simplified implementation using OpenCV's SIFT

def extract_sift_features(gray_image, n_keypoints=100):
    """Extract SIFT features and return a fixed-length feature vector."""
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    
    # If no keypoints are detected, return zeros
    if descriptors is None or len(keypoints) == 0:
        return np.zeros(128 * min(n_keypoints, 1))
    
    # Limit to n_keypoints
    if len(keypoints) > n_keypoints:
        # Sort keypoints by response (strength)
        keypoints_sorted = sorted(keypoints, key=lambda x: x.response, reverse=True)
        keypoints_top = keypoints_sorted[:n_keypoints]
        
        # Get indices of top keypoints
        indices = [keypoints.index(kp) for kp in keypoints_top]
        descriptors = descriptors[indices]
    
    # Flatten descriptors and pad/truncate to fixed length
    flattened = descriptors.flatten()
    if len(flattened) < 128 * n_keypoints:
        flattened = np.pad(flattened, (0, 128 * n_keypoints - len(flattened)))
    else:
        flattened = flattened[:128 * n_keypoints]
    
    return flattened

# Extract SIFT features (using a small number of keypoints for example)
try:
    sift_features = extract_sift_features(gray_image, n_keypoints=5)
    print(f"SIFT features shape: {sift_features.shape}")
    
    # Visualize SIFT keypoints
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(gray_image, None)
    sift_image = cv2.drawKeypoints(gray_image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Original Grayscale Image')
    plt.subplot(1, 2, 2)
    plt.imshow(sift_image)
    plt.title('SIFT Keypoints')
    plt.tight_layout()
    plt.show()
except:
    print("SIFT not available in this OpenCV installation.")
```

## Deep Learning Based Feature Extraction

Using pre-trained deep neural networks for feature extraction:

```python
# This requires installing TensorFlow/Keras
# pip install tensorflow

def extract_deep_features(image_rgb, model_name='VGG16', target_size=(224, 224)):
    """Extract features using a pre-trained deep learning model."""
    from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
    from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
    from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.models import Model
    
    # Resize image to target size expected by the model
    resized = cv2.resize(image_rgb, target_size)
    img_array = img_to_array(resized)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Select model and preprocessing function
    if model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False)
        preprocessed = preprocess_vgg(img_array)
    elif model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False)
        preprocessed = preprocess_resnet(img_array)
    elif model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False)
        preprocessed = preprocess_mobilenet(img_array)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Create a model that outputs features from a specific layer
    model = Model(inputs=base_model.input, outputs=base_model.output)
    
    # Extract features
    features = model.predict(preprocessed)
    
    # Flatten features
    flattened = features.flatten()
    
    return flattened

# The following code would work with TensorFlow installed
try:
    deep_features = extract_deep_features(image_rgb, model_name='VGG16')
    print(f"Deep learning features shape: {deep_features.shape}")
except:
    print("TensorFlow not installed or model not available.")
    print("To use deep learning features, install TensorFlow: pip install tensorflow")
```

## Feature Reduction and Selection for Image Data

When working with high-dimensional image features:

```python
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# Example: Combine multiple feature types
def combine_features(feature_sets):
    """Combine multiple feature sets into one feature vector."""
    return np.hstack(feature_sets)

# Combine color and texture features
combined_features = combine_features([color_features, texture_features, lbp_features])
print(f"Combined features shape: {combined_features.shape}")

# Example: Dimensionality reduction with PCA
def reduce_features_pca(features, n_components=50):
    """Reduce feature dimensionality using PCA."""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features.reshape(1, -1))
    
    pca = PCA(n_components=min(n_components, features.shape[0]))
    features_reduced = pca.fit_transform(features_scaled)
    
    return features_reduced.flatten(), pca

# Apply PCA to combined features
reduced_features, pca_model = reduce_features_pca(combined_features, n_components=50)
print(f"PCA reduced features shape: {reduced_features.shape}")
print(f"Explained variance ratio: {sum(pca_model.explained_variance_ratio_):.3f}")

# For feature selection with labels, you would use SelectKBest
# This is pseudocode since we don't have labels in this example
'''
def select_best_features(features, labels, k=20):
    """Select top k features based on ANOVA F-value."""
    selector = SelectKBest(f_classif, k=k)
    features_selected = selector.fit_transform(features, labels)
    return features_selected, selector

# Example usage with multiple images and labels
X = np.vstack([extract_all_features(img) for img in images])
y = np.array(labels)
X_selected, selector = select_best_features(X, y, k=20)
'''
```

## Creating an Image Feature Extraction Pipeline

Putting it all together in a reusable pipeline:

```python
# Complete pipeline for image feature extraction
def extract_image_features(image_path, include_deep=False):
    """
    Extract comprehensive image features from an image file.
    
    Parameters:
    -----------
    image_path : str
        Path to the image file
    include_deep : bool
        Whether to include deep learning features
        
    Returns:
    --------
    features : dict
        Dictionary of feature arrays
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to RGB (from BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Extract features
    features = {}
    
    # Color features
    features['color_histogram'] = extract_color_histogram(image_rgb)
    features['color_stats'] = extract_color_stats(image_rgb)
    
    # Texture features
    features['haralick'] = extract_haralick_features(gray_image)
    features['lbp'] = extract_lbp_features(gray_image)
    
    # Shape features
    features['shape'] = extract_shape_features(gray_image)
    
    # HOG features
    features['hog'] = extract_hog_features(gray_image)
    
    # SIFT features (if available)
    try:
        features['sift'] = extract_sift_features(gray_image, n_keypoints=5)
    except:
        features['sift'] = np.array([])
    
    # Deep learning features (if requested)
    if include_deep:
        try:
            features['deep'] = extract_deep_features(image_rgb)
        except:
            features['deep'] = np.array([])
    
    return features

# Example usage of the pipeline (pseudocode)
'''
# For a single image
image_features = extract_image_features('path/to/image.jpg')

# For multiple images
feature_dicts = [extract_image_features(img_path) for img_path in image_paths]

# Convert to DataFrame for analysis
feature_df = pd.DataFrame([
    {
        **{'image_path': path},
        **{f'color_hist_{i}': feat['color_histogram'][i] for i in range(len(feat['color_histogram']))},
        **{f'color_stat_{i}': feat['color_stats'][i] for i in range(len(feat['color_stats']))},
        # Add other features similarly...
    }
    for path, feat in zip(image_paths, feature_dicts)
])

# Add labels if available
feature_df['label'] = labels

# Now you can use this DataFrame for machine learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X = feature_df.drop(['image_path', 'label'], axis=1)
y = feature_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.3f}")
'''
```

## Image Augmentation for Feature Engineering

Data augmentation can be used to enhance model training:

```python
# Image augmentation transforms image data to create variations
def augment_image(image):
    """Apply random transformations to create a new variant of the image."""
    import random
    
    # Make a copy to avoid modifying the original
    img = image.copy()
    
    # Random horizontal flip
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
    
    # Random rotation
    angle = random.uniform(-15, 15)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    
    # Random brightness/contrast adjustment
    alpha = random.uniform(0.8, 1.2)  # Contrast
    beta = random.uniform(-10, 10)    # Brightness
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    # Random noise
    noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    return img

# Visualize augmentations
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.imshow(image_rgb)
plt.title('Original Image')

for i in range(5):
    plt.subplot(2, 3, i+2)
    augmented = augment_image(image_rgb)
    plt.imshow(augmented)
    plt.title(f'Augmentation {i+1}')

plt.tight_layout()
plt.show()

# Generate augmented datasets (pseudocode)
'''
def generate_augmented_dataset(images, labels, n_augmentations=5):
    """Generate an augmented dataset with n_augmentations per original image."""
    augmented_images = []
    augmented_labels = []
    
    for image, label in zip(images, labels):
        # Add the original image
        augmented_images.append(image)
        augmented_labels.append(label)
        
        # Add augmented variants
        for _ in range(n_augmentations):
            augmented_images.append(augment_image(image))
            augmented_labels.append(label)
    
    return np.array(augmented_images), np.array(augmented_labels)
'''
```

## Best Practices for Image Feature Engineering

1. **Start with domain-specific features**: Choose features that make sense for your specific problem (e.g., color for skin lesions, texture for materials, edges for manufactured parts).

2. **Use multiple feature types**: Combine color, texture, shape, and other features for a comprehensive representation.

3. **Apply normalization**: Always normalize features to ensure that no single feature dominates.

4. **Reduce dimensionality**: Use PCA or other techniques to reduce the size of large feature sets.

5. **Consider deep features**: For complex recognition tasks, features from pre-trained CNNs often outperform hand-crafted features.

6. **Test feature relevance**: Evaluate the impact of different feature types on model performance.

7. **Use augmentation**: Increase your training data size and robustness with augmentations.

8. **Balance extraction speed vs. accuracy**: Consider computational requirements, especially for real-time applications.

9. **Validate visually**: Visualize features when possible to ensure they capture the information you expect.

10. **Combine with end-to-end learning**: Consider hybrid approaches where some features are engineered and others are learned. 