# Neural Network From Scratch

A modular and extensible neural network framework built entirely from scratch with **Python and NumPy**.

This project provides a complete implementation of neural networks from first principles, focusing on understanding how networks function internally without relying on high-level deep learning frameworks. The entire system follows **Object-Oriented Programming** principles, delivering a flexible, configurable, and maintainable codebase.

---

## 💠Features

- **Fully configurable architecture** — Add layers, neurons, and activation functions dynamically
- **Object-oriented and modular design** — Each component is independent and extensible
- **Pure NumPy implementation** — No high-level deep learning dependencies
- **Multiple optimization algorithms** — SGD, Momentum, RMSProp, Adam
- **Comprehensive activation functions** — ReLU, Sigmoid, Tanh, Softmax
- **Advanced initialization strategies** — He and Xavier initialization
- **Multiple loss functions** — MSE, Binary Cross Entropy, Multiclass Cross Entropy
- **Thorough evaluation** — Classification and regression metrics
- **Experimental framework** — Hyperparameter analysis with visualization
- **Real-world applications** — MNIST classification example
- **Mathematical verification** — Numerical gradient checking and validation tests

---

## 💠Project Structure

```
neural-network-from-scratch/
│
├── Core/
│   ├── main_structure.py
│   ├── evaluator.py
│   ├── losses.py
│   ├── optimizers.py
│   │
│   └── layer/
│       ├── activation.py
│       ├── initialization.py
│       └── layers.py
│
├── Experiments/
│   ├── learning_rate/
│   ├── architecture/
│   ├── batch_size/
│   ├── batch_normalization/
│   ├── optimizers/
│   ├── initializations/
│   │
│   └── test_results/
│
├── Applications/
│   └── MNIST/
│
├── Verifications/
│   ├── activation/
│   ├── backpropagation/
│   ├── forward_pass/
│   ├── initialization/
│   └── loss/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Architecture Overview

The framework is organized into four complementary layers:

### 🧠Core

The neural network implementation, built around a small set of focused classes:

**NeuralNetwork**  
The primary interface for building, training, and evaluating networks. Supports:
- `add_layer()` — Dynamically add layers to the network
- `set_optimizer()` — Configure the optimization algorithm
- `fit()` — Train the network on data
- `predict()` — Generate predictions
- `evaluate()` — Assess model performance

Because layers are added dynamically, different architectures can be tested without modifying the framework:

```python
model = NeuralNetwork(input_size=784)

model.add_layer(64, "ReLU", hidden_init, bias=True)
model.add_layer(32, "ReLU", hidden_init, bias=True)
model.add_layer(16, "ReLU", hidden_init, bias=True)
model.add_layer(10, "Softmax", output_init, bias=True)

model.set_optimizer(Adam(learning_rate=0.001))
model.fit(X_train, y_train, epochs=100, batch_size=32)

predictions = model.predict(X_test)
accuracy, loss = model.evaluate(X_test, y_test)
```

**Layer**  
Individual neural network layers that encapsulate parameters and computation logic for forward and backward propagation. The independence of layers enables flexible architecture design.

**Optimizer**  
Manages parameter updates during training. Implementations include:
- Stochastic Gradient Descent (SGD)
- Momentum
- RMSProp
- Adam

Optimizers are decoupled from the network structure, allowing seamless algorithm switching.

**Evaluator**  
Computes performance metrics independently of the model. Supports:
- **Classification** — Accuracy, loss
- **Regression** — MAPE, loss

### Activation Functions

Fully implemented with forward pass and derivative computation:

- **ReLU** — Linear rectification for hidden layers
- **Sigmoid** — Smooth squashing to (0, 1)
- **Tanh** — Symmetric squashing to (-1, 1)
- **Softmax** — Multiclass probability distribution

### Weight Initialization

Two proven strategies for stable training:

- **He Initialization** — Suited for ReLU networks
- **Xavier Initialization** — General-purpose initialization

### Loss Functions

Three loss functions with gradient computation:

- **Mean Squared Error (MSE)** — Regression
- **Binary Cross Entropy** — Binary classification
- **Multiclass Cross Entropy** — Multiclass classification

---

## 🧪Experiments

Systematic investigation of architectural choices and hyperparameters through controlled experiments with quantitative results and visualizations.

**Learning Rate**  
Analyzes convergence speed, training stability, and final model performance across a range of learning rates.

**Architecture**  
Examines the effects of:
- Network depth (number of layers)
- Layer width (neurons per layer)
- Overall capacity and representation

**Batch Size**  
Studies the relationship between batch size and training dynamics, convergence, and generalization.

**Batch Normalization**  
Evaluates the impact of batch normalization on training speed and stability.

**Optimizers**  
Compares SGD, Momentum, RMSProp, and Adam across multiple tasks and settings.

**Initializations**  
Benchmarks He and Xavier initialization strategies on the same architectures.

All experiments include test results with plots and analytical conclusions in the `test_results/` directory.

---

## 🔨Applications

### MNIST Handwritten Digit Classification

Demonstrates the framework applied to real-world data. The complete workflow includes:

1. **Dataset** — Load and partition MNIST
2. **Preprocessing** — Normalize and shape data
3. **Network Construction** — Define architecture
4. **Training** — Optimize parameters
5. **Prediction** — Classify test samples
6. **Evaluation** — Measure performance

This end-to-end example proves the framework's applicability beyond isolated experiments.

---

## 📐Verification

Mathematical correctness is validated through systematic testing:

**Activation Functions**  
Verifies forward passes and derivative computation against expected behavior.

**Forward Propagation**  
Confirms data transformation through the network layers.

**Backpropagation**  
Compares analytically computed gradients with numerically estimated gradients to verify implementation correctness.

**Weight Initialization**  
Validates He and Xavier strategies meet statistical requirements.

**Loss Functions**  
Tests loss computation and gradient derivation.

---

## 🏛️Design Philosophy

The framework prioritizes modularity and extensibility through separation of concerns:

```
NeuralNetwork
      │
  ┌───┼───┐
  │   │   │
Layer Optimizer Evaluator
  │
  ├── Activation
  └── Initialization
```

This architecture enables independent evolution of components:

- Add optimizers without rewriting the network
- Introduce activation functions without modifying layers
- Extend loss functions without restructuring training
- Create architectures without changing core systems

The framework is designed as a small but complete neural network library rather than a monolithic model.

---

## ⚙️Technologies

**Python**  
The primary language for implementation and experimentation.

**NumPy**  
The sole numerical dependency. Used for:
- Linear algebra operations
- Array computations
- Gradient calculations

No high-level deep learning frameworks are used. All components—layers, propagation, activations, losses, initialization, optimizers, training, and evaluation—are implemented directly.

---

## ✅Learning Objectives

This project explores neural networks as complete systems by implementing core mechanisms from first principles:

- Neural network architecture and design patterns
- Dense layer computation and parameter management
- Forward propagation and data transformation
- Backpropagation and gradient flow
- Optimization algorithms and convergence
- Activation functions and nonlinearity
- Loss functions and training objectives
- Weight initialization and training stability
- Batch normalization and training dynamics
- Model evaluation and metrics
- Hyperparameter tuning through experimentation
- Numerical verification and validation
- Object-oriented software architecture

The goal is to understand neural networks as sophisticated systems rather than treating them as black boxes.

---

## 💠Future Development

The framework is designed for incremental expansion:

**Regularization**
- L1 regularization (Lasso)
- L2 regularization (Ridge)
- Additional regularization techniques

**Advanced Techniques**
- Dropout
- Early stopping
- Learning rate scheduling
- Custom training loops

**Additional Components**
- More activation functions (Leaky ReLU, ELU, GELU)
- More optimizers (AdaGrad, Adadelta)
- Additional loss functions for specialized tasks

**Enhanced Verification**
- Complete backpropagation pipeline testing
- Extended gradient checking across all components
- Performance benchmarking against reference implementations

---

## 🔄️Project Status
**Current State**  
The core framework is fully functional and production-ready for educational use. It supports:
- Configurable architectures
- Multiple activation functions
- Various initialization methods
- Multiple loss functions and optimizers
- Comprehensive hyperparameter tuning
- Full experimental framework

**Development**  
The project is actively maintained and expanded with new neural network concepts, techniques, experiments, and verification methods.

---

##  ✍️Author

**Sadra Mohamady**

Built from scratch with Python and NumPy as an exploration of neural networks, machine learning, mathematical implementation, experimentation, and modular software engineering.
