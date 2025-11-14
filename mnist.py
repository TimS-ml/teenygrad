#!/usr/bin/env python3
"""
MNIST Digit Classification Example for TeenyGrad

This script demonstrates how to train a convolutional neural network
to classify handwritten digits from the MNIST dataset using TeenyGrad.

MNIST Dataset:
--------------
- 60,000 training images of handwritten digits (0-9)
- 10,000 test images
- Each image is 28x28 grayscale pixels
- Classic benchmark for evaluating ML frameworks

Architecture (TinyConvNet):
---------------------------
- Input: 28x28 grayscale image (flattened to 784 features)
- Conv Layer 1: 8 filters, 3x3 kernel, ReLU activation, 2x2 max pooling
- Conv Layer 2: 16 filters, 3x3 kernel, ReLU activation, 2x2 max pooling
- Fully Connected: 400 -> 10 classes
- Output: Log softmax probabilities over 10 digit classes

Training:
---------
- Optimizer: Adam with learning rate 0.001
- Batch size: 128
- Loss: Sparse categorical cross-entropy
- Expected test accuracy: >93% after 100 steps

Usage:
------
    python mnist.py

This trains the model and evaluates it on the test set.
"""
import numpy as np
from teenygrad import Tensor
from tqdm import trange
import gzip, os

from teenygrad.nn import optim
from teenygrad.helpers import getenv


def train(model,
          X_train,
          Y_train,
          optim,
          steps,
          BS=128,
          lossfn=lambda out, y: out.sparse_categorical_crossentropy(y),
          transform=lambda x: x,
          target_transform=lambda x: x,
          noloss=False):
    """
    Train a model using mini-batch gradient descent.

    Args:
        model: The neural network model to train (must have forward() method or be callable)
        X_train: Training data features (numpy array)
        Y_train: Training data labels (numpy array)
        optim: Optimizer instance (e.g., SGD, Adam)
        steps: Number of training iterations
        BS: Batch size (default: 128)
        lossfn: Loss function (default: sparse categorical cross-entropy)
        transform: Function to transform input features (default: identity)
        target_transform: Function to transform labels (default: identity)
        noloss: If True, skip loss/accuracy computation for faster training (default: False)

    Returns:
        List containing [losses, accuracies] over training steps
    """
    Tensor.training = True
    losses, accuracies = [], []
    for i in (t := trange(steps, disable=getenv('CI', False))):
        samp = np.random.randint(0, X_train.shape[0], size=(BS))
        x = Tensor(transform(X_train[samp]), requires_grad=False)
        y = Tensor(target_transform(Y_train[samp]))

        # network
        out = model.forward(x) if hasattr(model, 'forward') else model(x)

        loss = lossfn(out, y)
        optim.zero_grad()
        loss.backward()
        if noloss: del loss
        optim.step()

        # printing
        if not noloss:
            cat = np.argmax(out.numpy(), axis=-1)
            accuracy = (cat == y.numpy()).mean()

            loss = loss.detach().numpy()
            losses.append(loss)
            accuracies.append(accuracy)
            t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))
    return [losses, accuracies]


def evaluate(model,
             X_test,
             Y_test,
             num_classes=None,
             BS=128,
             return_predict=False,
             transform=lambda x: x,
             target_transform=lambda y: y):
    """
    Evaluate model performance on test data.

    Args:
        model: The trained model to evaluate
        X_test: Test data features (numpy array)
        Y_test: Test data labels (numpy array)
        num_classes: Number of output classes (auto-detected if None)
        BS: Batch size for evaluation (default: 128)
        return_predict: If True, return predictions along with accuracy (default: False)
        transform: Function to transform input features (default: identity)
        target_transform: Function to transform labels (default: identity)

    Returns:
        Test accuracy (float), or (accuracy, predictions) if return_predict=True
    """
    Tensor.training = False

    def numpy_eval(Y_test, num_classes):
        Y_test_preds_out = np.zeros(list(Y_test.shape) + [num_classes])
        for i in trange((len(Y_test) - 1) // BS + 1,
                        disable=getenv('CI', False)):
            x = Tensor(transform(X_test[i * BS:(i + 1) * BS]))
            out = model.forward(x) if hasattr(model, 'forward') else model(x)
            Y_test_preds_out[i * BS:(i + 1) * BS] = out.numpy()
        Y_test_preds = np.argmax(Y_test_preds_out, axis=-1)
        Y_test = target_transform(Y_test)
        return (Y_test == Y_test_preds).mean(), Y_test_preds

    if num_classes is None: num_classes = Y_test.max().astype(int) + 1
    acc, Y_test_pred = numpy_eval(Y_test, num_classes)
    print("test set accuracy is %f" % acc)
    return (acc, Y_test_pred) if return_predict else acc


def fetch_mnist():
    """
    Load MNIST dataset from gzipped files.

    Returns:
        Tuple of (X_train, Y_train, X_test, Y_test)
        - X_train: Training images (60000, 784) float32 array
        - Y_train: Training labels (60000,) uint8 array
        - X_test: Test images (10000, 784) float32 array
        - Y_test: Test labels (10000,) uint8 array

    Note:
        Files are expected in extra/datasets/mnist/ directory.
        Images are flattened from 28x28 to 784-dimensional vectors.
    """
    parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8
                                       ).copy()
    BASE = os.path.dirname(__file__) + "/extra/datasets"
    # Parse training data (skip 16-byte header with [0x10:])
    X_train = parse(BASE + "/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape(
        (-1, 28 * 28)).astype(np.float32)
    Y_train = parse(BASE + "/mnist/train-labels-idx1-ubyte.gz")[8:]
    # Parse test data
    X_test = parse(BASE + "/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape(
        (-1, 28 * 28)).astype(np.float32)
    Y_test = parse(BASE + "/mnist/t10k-labels-idx1-ubyte.gz")[8:]
    return X_train, Y_train, X_test, Y_test


X_train, Y_train, X_test, Y_test = fetch_mnist()


# Create a model with convolutional layers
class TinyConvNet:
    """
    Tiny Convolutional Neural Network for MNIST classification.

    Architecture:
    -------------
    Input (batch_size, 784)
    -> Reshape (batch_size, 1, 28, 28)
    -> Conv2D(8 filters, 3x3) + ReLU + MaxPool(2x2)  # 8x14x14
    -> Conv2D(16 filters, 3x3) + ReLU + MaxPool(2x2) # 16x5x5
    -> Flatten (400)
    -> Linear(400 -> 10)
    -> LogSoftmax
    Output (batch_size, 10)

    Parameters are initialized with scaled uniform distribution for stable training.
    Based on: https://keras.io/examples/vision/mnist_convnet/
    """

    def __init__(self):
        """Initialize model parameters with scaled uniform initialization."""
        conv = 3  # Kernel size
        #inter_chan, out_chan = 32, 64  # Original size
        inter_chan, out_chan = 8, 16  # Reduced for speed

        # Conv layer 1: 8 filters, 1 input channel, 3x3 kernel
        self.c1 = Tensor.scaled_uniform(inter_chan, 1, conv, conv)
        # Conv layer 2: 16 filters, 8 input channels, 3x3 kernel
        self.c2 = Tensor.scaled_uniform(out_chan, inter_chan, conv, conv)
        # Fully connected: 16*5*5=400 -> 10 classes
        self.l1 = Tensor.scaled_uniform(out_chan * 5 * 5, 10)

    def forward(self, x: Tensor):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 784)

        Returns:
            Log probabilities for each class, shape (batch_size, 10)
        """
        # Reshape flattened input to 2D image
        x = x.reshape(shape=(-1, 1, 28, 28))  # (batch, channels, height, width)

        # First conv block: 28x28 -> 14x14
        x = x.conv2d(self.c1).relu().max_pool2d()

        # Second conv block: 14x14 -> 5x5
        x = x.conv2d(self.c2).relu().max_pool2d()

        # Flatten for fully connected layer
        x = x.reshape(shape=[x.shape[0], -1])  # (batch, 400)

        # Classification head
        return x.dot(self.l1).log_softmax()


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(1337)

    # Create model
    model = TinyConvNet()

    # Create optimizer with model parameters
    optimizer = optim.Adam([model.c1, model.c2, model.l1], lr=0.001)

    # Train for 100 steps
    print("Training TinyConvNet on MNIST...")
    train(model, X_train, Y_train, optimizer, steps=100)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    accuracy = evaluate(model, X_test, Y_test)

    # Verify minimum accuracy threshold
    assert accuracy > 0.93, f"Test accuracy {accuracy:.4f} is below threshold 0.93"
    print(f"✓ Achieved target accuracy: {accuracy:.4f} > 0.93")
