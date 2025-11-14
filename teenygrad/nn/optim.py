"""
Optimizers for Training Neural Networks in TeenyGrad

This module implements various optimization algorithms used for training neural networks.
Optimizers update model parameters based on computed gradients to minimize the loss function.

Implemented Optimizers:
-----------------------
1. SGD (Stochastic Gradient Descent):
   - Basic gradient descent with optional momentum
   - Momentum accelerates convergence by accumulating gradient history
   - Nesterov momentum looks ahead before computing gradients
   - Weight decay adds L2 regularization

2. Adam (Adaptive Moment Estimation):
   - Maintains adaptive learning rates for each parameter
   - First moment (mean) and second moment (variance) of gradients
   - Bias correction for moment estimates
   - Very popular for deep learning

3. AdamW (Adam with Decoupled Weight Decay):
   - Fixes weight decay implementation in Adam
   - Weight decay applied directly to parameters, not gradients

4. LAMB (Layer-wise Adaptive Moments for Batch training):
   - Designed for large batch training
   - Adds trust ratio based on layer-wise norm ratios
   - Can behave as Adam when trust ratio disabled

Typical Usage:
--------------
    # Create model and define parameters
    model = MyModel()
    params = [layer.weight, layer.bias for layer in model.layers]

    # Initialize optimizer
    opt = SGD(params, lr=0.01, momentum=0.9)

    # Training loop
    for batch in data:
        opt.zero_grad()          # Clear previous gradients
        loss = compute_loss()     # Forward pass
        loss.backward()           # Compute gradients
        opt.step()                # Update parameters

# sorted in order of increasing complexity
"""
from typing import List
from teenygrad.helpers import dedup
from teenygrad.tensor import Tensor


class Optimizer:
    """
    Base optimizer class.

    Attributes:
    - params: List of tensors to optimize
    - device: Device where the optimization is performed
    - buffers: List of non-trainable tensors
    - lr: Learning rate tensor

    Methods:
    - zero_grad(): Zeros out the gradients of all parameters
    - realize(): Realizes the parameters and buffers
    """

    def __init__(self, params: List[Tensor], lr: float):
        """
        Initialize the optimizer.

        Parameters:
        - params: List of tensors to optimize
        - lr: Learning rate
        """
        # if it's None, but being put into an optimizer, set it to True
        for x in params:
            if x.requires_grad is None: x.requires_grad = True

        self.params: List[Tensor] = dedup(
            [x for x in params if x.requires_grad])
        assert len(self.params) != 0, "optimizer must have at least one param"
        self.device = self.params[0].device
        self.buffers: List[Tensor] = dedup([
            x for x in params if not x.requires_grad
        ])  # buffers are still realized
        self.lr = Tensor([lr], requires_grad=False,
                         device=self.device).contiguous()

    def zero_grad(self):
        """
        Zero out all parameter gradients.

        This should be called before each backward pass to clear accumulated gradients
        from the previous iteration. Gradients accumulate by default in TeenyGrad.
        """
        for param in self.params:
            param.grad = None

    def realize(self, extra=None):
        """
        Force realization of all parameters, buffers, and optional extra tensors.

        This ensures all lazy computations are executed and values are materialized
        in memory. Called automatically by step() in optimizers to ensure updates
        are applied before the next iteration.

        Args:
            extra: Optional list of additional tensors to realize
        """
        # NOTE: in extra is too late for most of the params due to issues with assign
        Tensor.corealize(extra + self.params +
                         self.buffers if extra is not None else self.params +
                         self.buffers)


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer with momentum.

    Attributes:
    - momentum: Momentum factor
    - wd: Weight decay
    - nesterov: Whether to use Nesterov momentum
    - b: Momentum buffers

    Methods:
    - step(): Performs a single optimization step
    """

    def __init__(self,
                 params: List[Tensor],
                 lr=0.001,
                 momentum=0,
                 weight_decay=0.0,
                 nesterov=False):
        """
        Initialize the SGD optimizer.

        Parameters:
        - params: List of tensors to optimize
        - lr: Learning rate (default: 0.001)
        - momentum: Momentum factor (default: 0)
        - weight_decay: Weight decay factor (default: 0.0)
        - nesterov: Whether to use Nesterov momentum (default: False)
        """
        super().__init__(params, lr)
        self.momentum, self.wd, self.nesterov = momentum, weight_decay, nesterov
        self.b = [
            Tensor.zeros(*t.shape, device=t.device, requires_grad=False)
            for t in self.params
        ] if self.momentum else []

    # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    def step(self) -> None:
        """
        Perform a single optimization step (parameter update).

        SGD Update Rules:
        -----------------
        Without momentum:
            θ = θ - lr * (∇L + wd * θ)

        With momentum:
            v = momentum * v + (∇L + wd * θ)
            θ = θ - lr * v

        With Nesterov momentum:
            v = momentum * v + (∇L + wd * θ)
            θ = θ - lr * (∇L + momentum * v)

        Where:
        - θ: parameter
        - ∇L: gradient of loss w.r.t. parameter
        - lr: learning rate
        - wd: weight decay
        - v: velocity (momentum buffer)
        """
        for i, t in enumerate(self.params):
            assert t.grad is not None
            # Gradient with weight decay (L2 regularization)
            g = t.grad.realize() + self.wd * t.detach()
            if self.momentum:
                # Update momentum buffer: v = momentum * v + g
                self.b[i].assign(self.momentum * self.b[i] + g).realize(
                )  # NOTE: self.b[i] is zero on the first run, no if required
                # Nesterov: use (g + momentum * v), else: use v
                g = (g +
                     self.momentum * self.b[i]) if self.nesterov else self.b[i]
            # Update parameter: θ = θ - lr * g
            t.assign(t.detach() - g * self.lr)
        self.realize(self.b)


# LAMB is essentially just the trust ratio part of LARS applied to Adam/W so if we just set the trust ratio to 1.0 its just Adam/W.
def AdamW(params: List[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
    return LAMB(params, lr, b1, b2, eps, wd, adam=True)


def Adam(params: List[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    return LAMB(params, lr, b1, b2, eps, 0.0, adam=True)


class LAMB(Optimizer):
    """
    LAMB (Layer-wise Adaptive Moments optimizer for Batch training) optimizer.
    Can also behave as Adam/AdamW when adam=True.

    Attributes:
    - b1: First moment decay rate
    - b2: Second moment decay rate
    - eps: Small constant for numerical stability
    - wd: Weight decay
    - adam: Whether to run in Adam mode
    - t: Step counter
    - m: First moment buffers
    - v: Second moment buffers

    Methods:
    - step(): Performs a single optimization step
    """

    def __init__(self,
                 params: List[Tensor],
                 lr=0.001,
                 b1=0.9,
                 b2=0.999,
                 eps=1e-6,
                 wd=0.0,
                 adam=False):
        """
        Initialize the LAMB optimizer.

        Parameters:
        - params: List of tensors to optimize
        - lr: Learning rate (default: 0.001)
        - b1: First moment decay rate (default: 0.9)
        - b2: Second moment decay rate (default: 0.999)
        - eps: Small constant for numerical stability (default: 1e-6)
        - wd: Weight decay (default: 0.0)
        - adam: Whether to run in Adam mode (default: False)
        """
        super().__init__(params, lr)
        self.b1, self.b2, self.eps, self.wd, self.adam, self.t = b1, b2, eps, wd, adam, Tensor(
            [0], requires_grad=False).realize()
        self.m = [
            Tensor.zeros(*t.shape, device=t.device, requires_grad=False)
            for t in self.params
        ]
        self.v = [
            Tensor.zeros(*t.shape, device=t.device, requires_grad=False)
            for t in self.params
        ]

    def step(self) -> None:
        """
        Perform a single optimization step (parameter update).

        LAMB/Adam Update Rules:
        -----------------------
        1. Update biased moment estimates:
           m = β₁ * m + (1 - β₁) * g        (first moment, mean)
           v = β₂ * v + (1 - β₂) * g²       (second moment, variance)

        2. Bias correction:
           m_hat = m / (1 - β₁^t)
           v_hat = v / (1 - β₂^t)

        3. Compute update:
           up = m_hat / (√v_hat + ε) + wd * θ

        4. LAMB trust ratio (skip if adam=True):
           r = ||θ|| / ||up||  (layer-wise scaling)

        5. Update parameter:
           θ = θ - lr * r * up

        Where:
        - g: gradient
        - m, v: moment estimates
        - β₁, β₂: moment decay rates
        - t: timestep
        - ε: numerical stability constant
        - wd: weight decay
        - r: trust ratio (LAMB only)
        """
        # Increment timestep
        self.t.assign(self.t + 1).realize()
        for i, t in enumerate(self.params):
            assert t.grad is not None
            g = t.grad.realize()

            # Update biased first moment estimate: m = β₁ * m + (1 - β₁) * g
            self.m[i].assign(self.b1 * self.m[i] +
                             (1.0 - self.b1) * g).realize()
            # Update biased second moment estimate: v = β₂ * v + (1 - β₂) * g²
            self.v[i].assign(self.b2 * self.v[i] + (1.0 - self.b2) *
                             (g * g)).realize()

            # Bias-corrected moment estimates
            m_hat = self.m[i] / (1.0 - self.b1**self.t)
            v_hat = self.v[i] / (1.0 - self.b2**self.t)

            # Compute adaptive update direction
            up = (m_hat / (v_hat.sqrt() + self.eps)) + self.wd * t.detach()

            # LAMB: compute trust ratio for layer-wise adaptation
            if not self.adam:
                r1 = t.detach().square().sum().sqrt()  # ||θ||
                r2 = up.square().sum().sqrt()          # ||up||
                r = Tensor.where(r1 > 0, Tensor.where(r2 > 0, r1 / r2, 1.0),
                                 1.0)
            else:
                r = 1.0  # Adam: no trust ratio

            # Update parameter: θ = θ - lr * r * up
            t.assign(t.detach() - self.lr * r * up)
        self.realize([self.t] + self.m + self.v)
