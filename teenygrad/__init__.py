"""
TeenyGrad: A Minimal Deep Learning Framework

TeenyGrad is a tiny, educational deep learning framework designed to demonstrate
the core concepts of automatic differentiation and neural network training.

Key Features:
-------------
- Automatic Differentiation: Reverse-mode autodiff (backpropagation)
- Lazy Evaluation: Operations build a computation graph for optimization
- Tensor Operations: Full suite of mathematical and neural network operations
- Optimizers: SGD, Adam, AdamW, LAMB
- Neural Network Layers: Convolutions, activations, normalization
- Pure Python: Minimal dependencies (numpy only for CPU backend)

Main Components:
----------------
- Tensor: The core N-dimensional array class with autograd support
- LazyBuffer: Deferred execution engine for operation optimization
- mlops: Differentiable operations (add, mul, conv2d, etc.)
- ops: Low-level operation primitives
- nn.optim: Optimization algorithms

Quick Start:
------------
    from teenygrad import Tensor

    # Create tensors
    x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    y = Tensor([[5, 6], [7, 8]], requires_grad=True)

    # Forward pass
    z = (x * y).sum()

    # Backward pass (compute gradients)
    z.backward()

    # Access gradients
    print(x.grad)  # Gradients w.r.t. x

Educational Purpose:
--------------------
TeenyGrad is inspired by micrograd and tinygrad, focusing on clarity and
simplicity to help understand how modern deep learning frameworks work internally.

See mnist.py for a complete training example on the MNIST dataset.
"""
from teenygrad.tensor import Tensor  # noqa: F401
