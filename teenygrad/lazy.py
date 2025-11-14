"""
Lazy Buffer Implementation for TeenyGrad

This module implements lazy evaluation for tensor operations in TeenyGrad.
Lazy evaluation means that operations are not executed immediately when called,
but instead build up a computation graph that can be optimized and executed later.

Key Concepts:
-------------
1. LazyBuffer: The core abstraction that wraps tensor data and operations.
   - Stores the computation graph instead of immediate results
   - Operations return new LazyBuffers that reference the graph
   - Actual computation happens only when .schedule() and realize() are called

2. RawCPUBuffer: A simple wrapper around numpy arrays for realized (computed) data.

Benefits of Lazy Evaluation:
-----------------------------
- Operation Fusion: Multiple operations can be combined into single kernels
- Memory Optimization: Intermediate results can be avoided
- Graph Optimization: The computation graph can be optimized before execution
- Device Flexibility: Operations can be scheduled optimally across devices

Example Flow:
-------------
1. User creates tensors and performs operations: z = (x + y) * 2
2. Each operation creates a LazyBuffer node in the graph
3. When z.realize() is called:
   - schedule() walks the graph and creates an execution plan
   - run_schedule() executes the plan
   - The result is materialized into actual memory

Note: In this minimal implementation, operations are executed eagerly using numpy,
but the structure supports true lazy evaluation in a full implementation.
"""
from __future__ import annotations
from teenygrad.helpers import DType, dtypes, DEBUG
from teenygrad.ops import UnaryOps, BinaryOps, ReduceOps, TernaryOps, LoadOps
import numpy as np


class RawCPUBuffer:
    """
    Represents a raw CPU buffer.

    Attributes:
    - x: The raw CPU buffer.

    Methods:
    - toCPU(): Returns the raw CPU buffer.
    """

    def __init__(self, x):
        self.x = x

    def toCPU(self):
        """
        Returns the raw CPU buffer.

        Returns:
        - The raw CPU buffer.
        """
        return self.x


class LazyBuffer:
    """
    Represents a lazy buffer.

    Attributes:
    - device: The device of the buffer.
    - _np: The underlying numpy array.

    Properties:
    - base: Returns the base buffer.
    - dtype: Returns the data type of the buffer.
    - realized: Returns the realized buffer.
    - shape: Returns the shape of the buffer.

    Methods:
    - __repr__(): Returns a string representation of the buffer.
    - schedule(seen=None): Returns the schedule of the buffer.
    - is_unrealized_contiguous_const(): Checks if the buffer is an unrealized contiguous constant.
    - copy_to_device(device: str) -> LazyBuffer: Copies the buffer to the specified device.
    - fromCPU(x): Creates a lazy buffer from a CPU buffer.
    - loadop(op, shape, dtype, device, arg=None, src=None) -> LazyBuffer: Creates a lazy buffer using a load operation.
    - contiguous(x): Returns the contiguous buffer.
    - const(x) -> LazyBuffer: Creates a constant buffer.
    - cast(dtype: DType, bitcast: bool = False): Casts the buffer to the specified data type.
    - e(op, *srcs: LazyBuffer): Performs an element-wise operation on the buffer.
    - r(op, new_shape): Performs a reduction operation on the buffer.
    - reshape(arg): Reshapes the buffer.
    - expand(arg): Expands the buffer.
    - shrink(arg): Shrinks the buffer.
    - permute(arg): Permutes the buffer.
    - pad(arg): Pads the buffer.
    - stride(arg): Applies strides to the buffer.
    """

    device = "CPU"

    def __init__(self, buf: np.ndarray):
        self._np = buf

    @property
    def base(self):
        """
        Returns the base buffer.

        Returns:
        - The base buffer.
        """
        return self

    @property
    def dtype(self):
        """
        Returns the data type of the buffer.

        Returns:
        - The data type of the buffer.
        """
        return dtypes.from_np(self._np.dtype)

    @property
    def realized(self):
        """
        Returns the realized buffer.

        Returns:
        - The realized buffer.
        """
        return RawCPUBuffer(self._np)

    @property
    def shape(self):
        """
        Returns the shape of the buffer.

        Returns:
        - The shape of the buffer.
        """
        return self._np.shape

    def __repr__(self):
        """
        Returns a string representation of the buffer.

        Returns:
        - A string representation of the buffer.
        """
        return f"<LB {self.shape} {self.dtype}>"

    def schedule(self, seen=None):
        """
        Returns the schedule of the buffer.

        Parameters:
        - seen: A set of already seen buffers (default: None).

        Returns:
        - The schedule of the buffer.
        """
        return []

    def is_unrealized_contiguous_const(self):
        """
        Checks if the buffer is an unrealized contiguous constant.

        Returns:
        - True if the buffer is an unrealized contiguous constant, False otherwise.
        """
        return False

    def copy_to_device(self, device: str) -> LazyBuffer:
        """
        Copies the buffer to the specified device.

        Parameters:
        - device: The device to copy the buffer to.

        Returns:
        - The copied buffer.
        """
        return self

    @staticmethod
    def fromCPU(x):
        """
        Creates a lazy buffer from a CPU buffer.

        Parameters:
        - x: The CPU buffer.

        Returns:
        - The lazy buffer.
        """
        return LazyBuffer(x)

    @staticmethod
    def loadop(op, shape, dtype, device, arg=None, src=None) -> LazyBuffer:
        """
        Creates a lazy buffer using a load operation.

        Parameters:
        - op: The load operation.
        - shape: The shape of the buffer.
        - dtype: The data type of the buffer.
        - device: The device of the buffer.
        - arg: The argument for the load operation (default: None).
        - src: The source buffer for the load operation (default: None).

        Returns:
        - The lazy buffer.
        """
        if op == LoadOps.RAND:
            return LazyBuffer(
                np.random.default_rng(arg).random(size=shape, dtype=dtype.np))
        elif op == LoadOps.CONST:
            return LazyBuffer(np.full(shape, arg, dtype=dtype.np))
        elif op == LoadOps.EMPTY:
            return LazyBuffer(np.empty(shape, dtype=dtype.np))
        else:
            raise NotImplementedError(op)

    def contiguous(x):
        """
        Returns the contiguous buffer.

        Parameters:
        - x: The buffer.

        Returns:
        - The contiguous buffer.
        """
        return x

    def const(self, x) -> LazyBuffer:
        """
        Creates a constant buffer.

        Parameters:
        - x: The constant value.

        Returns:
        - The constant buffer.
        """
        return LazyBuffer(np.full_like(self._np, x))

    def cast(self, dtype: DType, bitcast: bool = False):
        """
        Casts the buffer to the specified data type.

        Parameters:
        - dtype: The data type to cast to.
        - bitcast: Whether to perform a bitcast (default: False).

        Returns:
        - The casted buffer.
        """
        return LazyBuffer(
            self._np.view(dtype.np) if bitcast else self._np.astype(dtype.np))

    def e(self, op, *srcs: LazyBuffer):
        """
        Performs an element-wise operation on the buffer.

        Parameters:
        - op: The element-wise operation.
        - srcs: The source buffers.

        Returns:
        - The result buffer.
        """
        if DEBUG >= 1: print(op, self, srcs)
        if op == UnaryOps.NEG: ret = -self._np
        elif op == UnaryOps.EXP2: ret = np.exp2(self._np)
        elif op == UnaryOps.LOG2: ret = np.log2(self._np)
        elif op == UnaryOps.SIN: ret = np.sin(self._np)
        elif op == UnaryOps.SQRT: ret = np.sqrt(self._np)
        elif op == BinaryOps.ADD: ret = self._np + srcs[0]._np
        elif op == BinaryOps.SUB: ret = self._np - srcs[0]._np
        elif op == BinaryOps.MUL: ret = self._np * srcs[0]._np
        elif op == BinaryOps.DIV: ret = self._np / srcs[0]._np
        elif op == BinaryOps.MAX: ret = np.maximum(self._np, srcs[0]._np)
        elif op == BinaryOps.CMPLT: ret = self._np < srcs[0]._np
        elif op == TernaryOps.WHERE:
            ret = np.where(self._np, srcs[0]._np, srcs[1]._np)
        else:
            raise NotImplementedError(op)
        return LazyBuffer(
            ret.astype(self.dtype.np if len(srcs) == 0 else max(
                self.dtype, *[x.dtype for x in srcs]).np,
                       copy=False))

    def r(self, op, new_shape):
        """
        Performs a reduction operation on the buffer.

        Parameters:
        - op: The reduction operation.
        - new_shape: The new shape of the buffer.

        Returns:
        - The result buffer.
        """
        if DEBUG >= 1: print(op, self, new_shape)
        assert len(self.shape) == len(
            new_shape), "reduce shapes must have same dimensions"
        axis = tuple(i for i, (a, b) in enumerate(zip(self.shape, new_shape))
                     if a != b)
        if op == ReduceOps.SUM:
            return LazyBuffer(
                self._np.sum(axis, dtype=self._np.dtype, keepdims=True))
        elif op == ReduceOps.MAX:
            return LazyBuffer(self._np.max(axis, keepdims=True))
        else:
            raise NotImplementedError(op)

    # MovementOps
    def reshape(self, arg):
        """
        Reshapes the buffer.

        Parameters:
        - arg: The new shape.

        Returns:
        - The reshaped buffer.
        """
        return LazyBuffer(self._np.reshape(arg))

    def expand(self, arg):
        """
        Expands the buffer.

        Parameters:
        - arg: The new shape.

        Returns:
        - The expanded buffer.
        """
        return LazyBuffer(np.broadcast_to(self._np, arg))

    def shrink(self, arg):
        """
        Shrinks the buffer.

        Parameters:
        - arg: The new shape.

        Returns:
        - The shrunk buffer.
        """
        return LazyBuffer(self._np[tuple(slice(p[0], p[1], None)
                                         for p in arg)])

    def permute(self, arg):
        """
        Permutes the buffer.

        Parameters:
        - arg: The permutation order.

        Returns:
        - The permuted buffer.
        """
        return LazyBuffer(self._np.transpose(arg))

    def pad(self, arg):
        """
        Pads the buffer.

        Parameters:
        - arg: The padding configuration.

        Returns:
        - The padded buffer.
        """
        return LazyBuffer(np.pad(self._np, arg))

    def stride(self, arg):
        """
        Applies strides to the buffer.

        Parameters:
        - arg: The stride configuration.

        Returns:
        - The strided buffer.
        """
        return LazyBuffer(self._np[tuple(slice(None, None, i) for i in arg)])
