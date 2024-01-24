import math
from typing import Tuple, Optional, cast
from teenygrad.helpers import argsort, DType
from teenygrad.ops import UnaryOps, BinaryOps, TernaryOps, ReduceOps
from teenygrad.tensor import Function
from teenygrad.lazy import LazyBuffer
from teenygrad.shape.symbolic import sint


class Contiguous(Function):
    """
    A function that ensures the input tensor is contiguous.

    Args:
        x (LazyBuffer): The input tensor.

    Returns:
        LazyBuffer: The contiguous tensor.
    """

    def forward(self, x: LazyBuffer) -> LazyBuffer:
        return x.contiguous()

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output


class ContiguousBackward(Function):
    """
    A class representing the backward operation for contiguous tensors.

    This class inherits from the Function class and provides the implementation
    for the backward operation. The backward operation takes the gradient of the
    output tensor with respect to the input tensor and returns the gradient of the
    input tensor.

    Args:
        x (LazyBuffer): The input tensor.

    Returns:
        LazyBuffer: The gradient of the input tensor.
    """

    def forward(self, x: LazyBuffer) -> LazyBuffer:
        return x

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.contiguous()


class Cast(Function):
    """
    This class represents a cast operation in the computation graph.

    Args:
        x (LazyBuffer): The input buffer to be casted.
        dtype (DType): The target data type for the cast operation.
        bitcast (bool, optional): Whether to perform a bitcast operation. Defaults to False.

    Returns:
        LazyBuffer: The casted buffer.

    """

    def forward(self,
                x: LazyBuffer,
                dtype: DType,
                bitcast: bool = False) -> LazyBuffer:
        self.input_dtype, self.bitcast = x.dtype, bitcast
        return x.cast(dtype, bitcast)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.cast(self.input_dtype, self.bitcast)


# ************* unary ops *************


class Zero(Function):

    def forward(self, x: LazyBuffer) -> LazyBuffer:
        return x.const(0)

    def backward(self, grad: LazyBuffer) -> LazyBuffer:
        return grad.const(0)


class Neg(Function):

    def forward(self, x: LazyBuffer) -> LazyBuffer:
        return x.e(UnaryOps.NEG)

    def backward(self, grad: LazyBuffer) -> LazyBuffer:
        return grad.e(UnaryOps.NEG)


class Sin(Function):

    def forward(self, x: LazyBuffer) -> LazyBuffer:
        self.x = x
        return x.e(UnaryOps.SIN)

    def backward(self, grad: LazyBuffer) -> LazyBuffer:
        return self.x.const(math.pi / 2).e(BinaryOps.SUB,
                                           self.x).e(UnaryOps.SIN).e(
                                               BinaryOps.MUL, grad)


# NOTE: maximum(x, 0) behaves differently where x=0
class Relu(Function):

    def forward(self, x: LazyBuffer) -> LazyBuffer:
        self.ret = x.e(BinaryOps.MAX, x.const(0))
        return self.ret

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return self.ret.const(0).e(BinaryOps.CMPLT,
                                   self.ret).e(BinaryOps.MUL, grad_output)


class Log(Function):

    def forward(self, x: LazyBuffer) -> LazyBuffer:
        self.x = x
        return x.e(UnaryOps.LOG2).e(BinaryOps.MUL, x.const(math.log(2)))

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.e(BinaryOps.DIV, self.x)


class Exp(Function):

    def forward(self, x: LazyBuffer) -> LazyBuffer:
        self.ret = x.e(BinaryOps.MUL,
                       x.const(1 / math.log(2))).e(UnaryOps.EXP2)
        return self.ret

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return self.ret.e(BinaryOps.MUL, grad_output)


class Sqrt(Function):

    def forward(self, x: LazyBuffer) -> LazyBuffer:
        self.ret = x.e(UnaryOps.SQRT)
        return self.ret

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.e(BinaryOps.DIV,
                             self.ret.e(BinaryOps.MUL, self.ret.const(2)))


# NOTE: the implicit derivative of sigmoid is not stable
# https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
# TODO: have the backend automatically find this
class Sigmoid(Function):

    def forward(self, x: LazyBuffer) -> LazyBuffer:
        self.ret = x.const(1).e(
            BinaryOps.DIV,
            x.const(1).e(
                BinaryOps.ADD,
                x.e(BinaryOps.MUL,
                    x.const(-1 / math.log(2))).e(UnaryOps.EXP2)))
        return self.ret

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return self.ret.e(BinaryOps.MUL,
                          self.ret.const(1).e(BinaryOps.SUB, self.ret)).e(
                              BinaryOps.MUL, grad_output)


# ************* binary ops *************


class Less(Function):
    """
    Performs element-wise less-than comparison between two tensors.

    Args:
        x (LazyBuffer): The first input tensor.
        y (LazyBuffer): The second input tensor.

    Returns:
        LazyBuffer: A tensor containing the result of the element-wise less-than comparison.
    """

    def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
        return x.e(BinaryOps.CMPLT, y)


class Add(Function):
    """
    A function that performs element-wise addition of two tensors.

    Args:
        x (LazyBuffer): The first input tensor.
        y (LazyBuffer): The second input tensor.

    Returns:
        LazyBuffer: The result of element-wise addition of x and y.
    """

    def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
        return x.e(BinaryOps.ADD, y)

    def backward(
        self, grad_output: LazyBuffer
    ) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
        return grad_output if self.needs_input_grad[0] else None, \
               grad_output if self.needs_input_grad[1] else None


class Sub(Function):
    """
    Subtraction function.

    This function performs element-wise subtraction between two input tensors.

    Args:
        x (LazyBuffer): The first input tensor.
        y (LazyBuffer): The second input tensor.

    Returns:
        LazyBuffer: The result of element-wise subtraction between x and y.

    """

    def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
        return x.e(BinaryOps.SUB, y)

    def backward(
        self, grad_output: LazyBuffer
    ) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
        return grad_output if self.needs_input_grad[0] else None, \
               grad_output.e(UnaryOps.NEG) if self.needs_input_grad[1] else None


class Mul(Function):
    """
    This class represents the multiplication operation in the computation graph.

    Attributes:
        x (LazyBuffer): The first input tensor.
        y (LazyBuffer): The second input tensor.
    """

    def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
        self.x, self.y = x, y
        return x.e(BinaryOps.MUL, y)

    def backward(
        self, grad_output: LazyBuffer
    ) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
        return self.y.e(BinaryOps.MUL, grad_output) if self.needs_input_grad[0] else None, \
               self.x.e(BinaryOps.MUL, grad_output) if self.needs_input_grad[1] else None


class Div(Function):
    """
    A class representing the division operation.

    Attributes:
        x (LazyBuffer): The first input tensor.
        y (LazyBuffer): The second input tensor.
    """

    def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
        self.x, self.y = x, y
        return x.e(BinaryOps.DIV, y)

    def backward(
        self, grad_output: LazyBuffer
    ) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
        return (
            grad_output.e(BinaryOps.DIV, self.y) if self.needs_input_grad[0] else None,
            grad_output.e(UnaryOps.NEG)
            .e(BinaryOps.MUL, self.x)
            .e(BinaryOps.DIV, self.y.e(BinaryOps.MUL, self.y))
            if self.needs_input_grad[1]
            else None,
        )


# ************* ternary ops *************


class Where(Function):
    """
    This class represents the Where function, which performs element-wise conditional selection.
    """

    def forward(self, x: LazyBuffer, y: LazyBuffer,
                z: LazyBuffer) -> LazyBuffer:
        """
        Forward pass of the Where function.

        Args:
            x (LazyBuffer): The input tensor.
            y (LazyBuffer): The tensor to select from when the condition is True.
            z (LazyBuffer): The tensor to select from when the condition is False.

        Returns:
            LazyBuffer: The result of the element-wise conditional selection.
        """
        self.x = x
        return x.e(TernaryOps.WHERE, y, z)

    def backward(
        self, grad_output: LazyBuffer
    ) -> Tuple[None, Optional[LazyBuffer], Optional[LazyBuffer]]:
        """
        Backward pass of the Where function.

        Args:
            grad_output (LazyBuffer): The gradient of the output.

        Returns:
            Tuple[None, Optional[LazyBuffer], Optional[LazyBuffer]]: The gradients of the input tensors.
        """
        return None, \
               self.x.e(TernaryOps.WHERE, grad_output, grad_output.const(0)) if self.needs_input_grad[1] else None, \
               self.x.e(TernaryOps.WHERE, grad_output.const(0), grad_output) if self.needs_input_grad[2] else None


# ************* reduce ops *************


class Sum(Function):

    def forward(self, x: LazyBuffer, new_shape: Tuple[int, ...]) -> LazyBuffer:
        self.input_shape = x.shape
        return x.r(ReduceOps.SUM, new_shape)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.expand(self.input_shape)


class Max(Function):
    """
    Computes the maximum value along a specified dimension of a LazyBuffer.

    Args:
        x (LazyBuffer): The input LazyBuffer.
        new_shape (Tuple[int, ...]): The new shape of the LazyBuffer after the maximum operation.

    Returns:
        LazyBuffer: The LazyBuffer containing the maximum values along the specified dimension.

    Examples:
        >>> x = LazyBuffer([1, 2, 3, 4, 5])
        >>> new_shape = (2, 2)
        >>> max_op = Max()
        >>> result = max_op.forward(x, new_shape)
        >>> print(result)
        LazyBuffer([3, 5])
    """

    def forward(self, x: LazyBuffer, new_shape: Tuple[int, ...]) -> LazyBuffer:
        self.x, self.ret = x, x.r(ReduceOps.MAX, new_shape)
        return self.ret

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        # 1s in locations where the max was chosen (can be two locations)
        max_is_1s = self.x.const(1.0).e(
            BinaryOps.SUB,
            self.x.e(BinaryOps.CMPLT, self.ret.expand(self.x.shape)))
        div = max_is_1s.r(ReduceOps.SUM,
                          grad_output.shape).expand(self.x.shape)
        return max_is_1s.e(BinaryOps.DIV,
                           div).e(BinaryOps.MUL,
                                  grad_output.expand(self.x.shape))


# ************* movement ops *************


# NOTE: this is sum in reverse
class Expand(Function):

    def forward(self, x: LazyBuffer, shape: Tuple[int, ...]) -> LazyBuffer:
        self.input_shape = x.shape
        return x.expand(shape)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.r(ReduceOps.SUM, self.input_shape)


class Reshape(Function):

    def forward(self, x: LazyBuffer, shape: Tuple[int, ...]) -> LazyBuffer:
        self.input_shape = x.shape
        return x.reshape(shape)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.reshape(self.input_shape)


class Permute(Function):

    def forward(self, x: LazyBuffer, order: Tuple[int, ...]) -> LazyBuffer:
        self.input_order = order
        return x.permute(order)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.permute(argsort(self.input_order))


class Pad(Function):

    def forward(self, x: LazyBuffer, arg: Tuple[Tuple[int, int],
                                                ...]) -> LazyBuffer:
        self.narg = tuple([(p[0], s + p[0]) for s, p in zip(x.shape, arg)])
        return x.pad(arg)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.shrink(self.narg)


class Shrink(Function):

    def forward(self, x: LazyBuffer, arg: Tuple[Tuple[sint, sint],
                                                ...]) -> LazyBuffer:
        self.narg = tuple([(p[0], s - p[1]) for s, p in zip(x.shape, arg)])
        return x.shrink(arg)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        assert all(
            isinstance(x[0], int) and isinstance(x[1], int)
            for x in self.narg), "symbolic shrink does not support backward"
        # need this cast because mypy cannot narrow the type even with assert
        return grad_output.pad(cast(Tuple[Tuple[int, int], ...], self.narg))


class Flip(Function):

    def forward(self, x: LazyBuffer, axis: Tuple[int, ...]) -> LazyBuffer:
        self.arg = tuple(
            [-1 if i in set(axis) else 1 for i in range(len(x.shape))])
        return x.stride(self.arg)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.stride(self.arg)
