# Neural Network Theory

Neural networks form the foundation of modern deep learning. This document provides a comprehensive overview of the key theoretical concepts behind neural networks.

## Neurons: The Building Blocks

A neural network consists of interconnected computational units called neurons that process and transform input data through weighted connections.

### The Perceptron: A Single Neuron

The perceptron is the simplest form of a neural network, consisting of:
- Input features (x₁, x₂, ..., xₙ)
- Weights for each input (w₁, w₂, ..., wₙ)
- A bias term (b)
- An activation function (f)

The output of a perceptron is calculated as:
```
y = f(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
```

In vector notation:
```
y = f(w·x + b)
```

Where:
- w is the weight vector
- x is the input vector
- b is the bias term
- f is the activation function

```python
import numpy as np
import matplotlib.pyplot as plt

def perceptron(inputs, weights, bias, activation_fn):
    """
    Implements a single perceptron.
    
    Parameters:
    -----------
    inputs : array-like
        Input features
    weights : array-like
        Weight values for each input
    bias : float
        Bias term
    activation_fn : function
        Activation function to apply
        
    Returns:
    --------
    float
        Output of the perceptron
    """
    # Calculate the weighted sum of inputs plus bias
    z = np.dot(inputs, weights) + bias
    
    # Apply activation function
    return activation_fn(z)

# Example: Step function activation for binary classification
def step_activation(z):
    """Step activation function: 1 if z >= 0, otherwise 0"""
    return 1 if z >= 0 else 0

# Simple example: AND logic gate
and_weights = np.array([1, 1])
and_bias = -1.5

# Test with different inputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for input_pair in inputs:
    output = perceptron(input_pair, and_weights, and_bias, step_activation)
    print(f"Input: {input_pair}, Output: {output}")
```

## Activation Functions

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns.

### Common Activation Functions

1. **Sigmoid**: Maps input to a value between 0 and 1
   ```
   σ(z) = 1 / (1 + e^(-z))
   ```

2. **Hyperbolic Tangent (tanh)**: Maps input to a value between -1 and 1
   ```
   tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
   ```

3. **Rectified Linear Unit (ReLU)**: Returns the input if positive, otherwise 0
   ```
   ReLU(z) = max(0, z)
   ```

4. **Leaky ReLU**: Modifies ReLU to have a small slope for negative inputs
   ```
   Leaky ReLU(z) = max(αz, z) where α is a small constant (e.g., 0.01)
   ```

5. **Softmax**: Used for multi-class classification, converts values to probabilities
   ```
   softmax(z)ᵢ = e^(zᵢ) / Σ e^(zⱼ)
   ```

```python
import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))

def tanh(z):
    """Hyperbolic tangent activation function"""
    return np.tanh(z)

def relu(z):
    """Rectified Linear Unit activation function"""
    return np.maximum(0, z)

def leaky_relu(z, alpha=0.01):
    """Leaky ReLU activation function"""
    return np.maximum(alpha * z, z)

def softmax(z):
    """Softmax activation function"""
    exp_z = np.exp(z - np.max(z))  # Subtract max for numerical stability
    return exp_z / np.sum(exp_z)

# Visualize activation functions
z = np.linspace(-5, 5, 100)
activations = {
    'Sigmoid': sigmoid(z),
    'Tanh': tanh(z),
    'ReLU': relu(z),
    'Leaky ReLU': leaky_relu(z)
}

plt.figure(figsize=(12, 8))
for name, values in activations.items():
    plt.plot(z, values, label=name)

plt.title('Comparison of Activation Functions')
plt.xlabel('Input (z)')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.show()
```

### Choosing the Right Activation Function

| Activation | Advantages | Disadvantages | Use Cases |
|------------|------------|---------------|-----------|
| Sigmoid | Smooth, bounded output | Vanishing gradient, not zero-centered | Binary classification output layers |
| Tanh | Zero-centered, bounded output | Vanishing gradient | Hidden layers (legacy networks) |
| ReLU | Computationally efficient, reduces vanishing gradient | "Dying ReLU" problem | Default for hidden layers in CNNs |
| Leaky ReLU | Prevents dying neurons | Additional hyperparameter | When ReLU units are frequently inactive |
| Softmax | Outputs sum to 1 (probabilities) | Computationally expensive | Multi-class classification output layers |

## Feedforward Neural Networks

Feedforward neural networks (FFNs) consist of multiple layers of neurons where information flows in one direction, from input to output.

### Network Architecture

A typical feedforward neural network includes:
- **Input Layer**: Receives the raw input data
- **Hidden Layers**: Intermediate layers that transform the input
- **Output Layer**: Produces the final prediction

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple feedforward neural network
class FeedforwardNN(nn.Module):
    """
    A simple feedforward neural network with configurable architecture.
    """
    def __init__(self, input_size, hidden_sizes, output_size, activation=nn.ReLU()):
        """
        Initialize the network.
        
        Parameters:
        -----------
        input_size : int
            Size of the input features
        hidden_sizes : list of int
            Sizes of the hidden layers
        output_size : int
            Size of the output layer
        activation : torch.nn.Module
            Activation function to use between layers
        """
        super(FeedforwardNN, self).__init__()
        
        # Create a list to hold all layers
        layers = []
        
        # Input layer to first hidden layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(activation)
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(activation)
        
        # Last hidden layer to output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Create a sequential model from the layers
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.model(x)

# Example: Create a network with 2 inputs, 2 hidden layers, and 1 output
input_size = 2
hidden_sizes = [4, 3]  # Two hidden layers with 4 and 3 neurons
output_size = 1

model = FeedforwardNN(input_size, hidden_sizes, output_size)
print(model)

# Example input
x = torch.tensor([[0.5, 0.2]])
output = model(x)
print(f"Model output: {output}")
```

## Backpropagation: Training Neural Networks

Backpropagation is the core algorithm for training neural networks. It calculates gradients of the loss function with respect to weights, enabling optimization.

### The Backpropagation Algorithm

Backpropagation works in two phases:

1. **Forward Pass**: Compute outputs and store intermediate values
2. **Backward Pass**: Compute gradients and update weights

The key insight of backpropagation is the chain rule from calculus, which allows us to compute gradients for all layers by working backward from the output.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Create a simple neural network
model = nn.Sequential(
    nn.Linear(2, 3),
    nn.ReLU(),
    nn.Linear(3, 1)
)

# Sample input and target
x = torch.tensor([[0.5, 0.7]], requires_grad=True, dtype=torch.float32)
y_true = torch.tensor([[0.3]], dtype=torch.float32)

# Forward pass with gradient tracking
y_pred = model(x)

# Calculate loss
loss_fn = nn.MSELoss()
loss = loss_fn(y_pred, y_true)
print(f"Prediction: {y_pred.item():.4f}, Target: {y_true.item():.4f}")
print(f"Loss: {loss.item():.4f}")

# Backward pass (compute gradients)
loss.backward()

# Print gradients for first layer weights
print("Gradients for first layer weights:")
print(model[0].weight.grad)

# Update weights using an optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer.step()
```

### Computational Graphs

Backpropagation operates on a computational graph, which represents the flow of data through operations:

- Nodes represent operations (e.g., matrix multiplication, activation functions)
- Edges represent data flowing between operations
- Gradients flow backward through this graph

### Automatic Differentiation

Modern deep learning frameworks like PyTorch and TensorFlow implement automatic differentiation, which:

- Tracks operations in a dynamic computational graph
- Automatically computes gradients for all variables
- Simplifies the implementation of complex models

## Cost Functions

Cost functions (also called loss functions) measure how well the network's predictions match the true values.

### Common Cost Functions

#### Mean Squared Error (MSE)
Used for regression problems:

```
MSE = (1/n) * Σ(y_true - y_pred)²
```

```python
# MSE Loss in PyTorch
criterion = nn.MSELoss()
mse_loss = criterion(predictions, targets)
```

#### Binary Cross-Entropy
Used for binary classification problems:

```
BCE = -(1/n) * Σ[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
```

```python
# Binary Cross-Entropy in PyTorch
criterion = nn.BCELoss()
bce_loss = criterion(predictions, targets)
```

#### Categorical Cross-Entropy
Used for multi-class classification problems:

```
CCE = -(1/n) * Σ Σ y_true_ij * log(y_pred_ij)
```

```python
# Cross-Entropy in PyTorch (combines Softmax and NLL)
criterion = nn.CrossEntropyLoss()
ce_loss = criterion(logits, class_indices)
```

## Vanishing and Exploding Gradients

Two common problems in training deep networks:

1. **Vanishing Gradients**: Gradients become very small as they propagate backward, making early layers learn very slowly
2. **Exploding Gradients**: Gradients become very large, causing unstable updates

### Solutions for Vanishing Gradients

- Use activation functions with better gradient properties (ReLU, Leaky ReLU)
- Implement skip connections (as in ResNets)
- Use batch normalization
- Apply proper weight initialization

### Solutions for Exploding Gradients

- Gradient clipping (limit gradient magnitude)
- Proper weight initialization
- Batch normalization
- Regularization techniques

```python
# Gradient clipping example
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## Weight Initialization Techniques

Proper weight initialization is crucial for effective training:

### Xavier/Glorot Initialization

Designed for layers with tanh or sigmoid activations:

```python
# Xavier initialization in PyTorch
nn.init.xavier_uniform_(layer.weight)
```

### He/Kaiming Initialization

Designed for ReLU activations:

```python
# Kaiming initialization in PyTorch
nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
```

## Regularization Techniques

Regularization helps prevent overfitting by adding constraints to the learning process:

### L1 and L2 Regularization

Add penalties to the loss function based on weight magnitudes:

```python
# L2 regularization in PyTorch (weight decay)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
```

### Dropout

Randomly deactivate neurons during training to prevent co-adaptation:

```python
# Add dropout layers in PyTorch
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),  # 50% dropout probability
    nn.Linear(256, 10)
)
```

### Batch Normalization

Normalize layer inputs to reduce internal covariate shift:

```python
# Add batch normalization in PyTorch
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
```
