# About teenygrad
[George Hotz | Programming | teenygrad: because tinygrad is 4317 lines now! | github.com/tinygrad - YouTube](https://www.youtube.com/watch?v=yyHU5SJ-BPA)

teenygrad is <1000 line MNIST trainer

It shares 90% of its code with tinygrad,
so understanding teenygrad is a good step to understanding tinygrad.

While it supports almost all of tinygrad's functionality,
the extra 4k lines in tinygrad get you speed and diverse backend support.

tensor.py and mlops.py are both tinygrad's and teenygrad's frontend.
lazy.py is a replacement for all of tinygrad's backend.

The only dependency of teenygrad is numpy (well, and tqdm)

Usage:
pip install numpy tqdm
PYTHONPATH="." python mnist.py


# Project walk through

Here are some key components:

1. **Tensor Operations ([`teenygrad/tensor.py`](command:_github.copilot.openRelativePath?%5B%22teenygrad%2Ftensor.py%22%5D "teenygrad/tensor.py"))**: This file likely contains the `Tensor` class, which is the fundamental data structure in [`teenygrad`](command:_github.copilot.openRelativePath?%5B%22teenygrad%22%5D "teenygrad"). Tensors are multi-dimensional arrays, and they are the primary objects that neural networks operate on.

2. **Operations ([`teenygrad/ops.py`](command:_github.copilot.openRelativePath?%5B%22teenygrad%2Fops.py%22%5D "teenygrad/ops.py"))**: This file likely contains the definitions of various operations that can be performed on tensors, such as addition, multiplication, and various unary and binary operations.

3. **Optimizers ([`teenygrad/nn/optim.py`](command:_github.copilot.openRelativePath?%5B%22teenygrad%2Fnn%2Foptim.py%22%5D "teenygrad/nn/optim.py"))**: This file contains implementations of various optimization algorithms, such as Stochastic Gradient Descent (SGD), Adam, and AdamW. These algorithms are used to adjust the parameters of the network in response to the gradients computed during backpropagation.

4. **Network Definitions ([`test/test_optim.py`](command:_github.copilot.openRelativePath?%5B%22test%2Ftest_optim.py%22%5D "test/test_optim.py"))**: This file contains the definition of [`TinyNet`](command:_github.copilot.openSymbolInFile?%5B%22test%2Ftest_optim.py%22%2C%22TinyNet%22%5D "test/test_optim.py"), a simple neural network. The network is defined as a class with a [`forward`](command:_github.copilot.openSymbolInFile?%5B%22mnist.py%22%2C%22forward%22%5D "mnist.py") method, which computes the output of the network given some input.

5. **Tests ([`test/test_optim.py`](command:_github.copilot.openRelativePath?%5B%22test%2Ftest_optim.py%22%5D "test/test_optim.py"))**: This file contains various tests for the optimization algorithms. It appears to compare the output of the [`teenygrad`](command:_github.copilot.openRelativePath?%5B%22teenygrad%22%5D "teenygrad") optimizers with those from the `torch` library to ensure they are working correctly.

6. **Machine Learning Operations ([`teenygrad/mlops.py`](command:_github.copilot.openRelativePath?%5B%22teenygrad%2Fmlops.py%22%5D "teenygrad/mlops.py"))**: This file contains the implementation of various machine learning operations as classes that inherit from the `Function` class. Each class implements a [`forward`](command:_github.copilot.openSymbolInFile?%5B%22mnist.py%22%2C%22forward%22%5D "mnist.py") method for computing the output of the operation, and a [`backward`](command:_github.copilot.openSymbolInFile?%5B%22teenygrad%2Fmlops.py%22%2C%22backward%22%5D "teenygrad/mlops.py") method for computing the gradient during backpropagation.
