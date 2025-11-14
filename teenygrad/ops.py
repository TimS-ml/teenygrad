# https://www.geeksforgeeks.org/enum-in-python/
from enum import Enum, auto
from typing import Optional


class UnaryOps(Enum):
    """
    Enum class representing unary operations.

    Attributes:
        NOOP: No operation.
        EXP2: Exponential of 2.
        LOG2: Logarithm base 2.
        CAST: Type casting.
        SIN: Sine function.
        SQRT: Square root.
        RECIP: Reciprocal.
        NEG: Negation.
    """
    NOOP = auto()
    EXP2 = auto()
    LOG2 = auto()
    CAST = auto()
    SIN = auto()
    SQRT = auto()
    RECIP = auto()
    NEG = auto()  # noqa: E702


class BinaryOps(Enum):
    """
    Enum class representing binary operations.

    Attributes:
        ADD: Addition operation.
        SUB: Subtraction operation.
        MUL: Multiplication operation.
        DIV: Division operation.
        MAX: Maximum operation.
        MOD: Modulo operation.
        CMPLT: Comparison operation (less than).
    """
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MAX = auto()
    MOD = auto()
    CMPLT = auto()  # noqa: E702


class ReduceOps(Enum):
    """
    Enum class representing reduction operations.

    Reduction operations collapse one or more dimensions of a tensor by applying
    an associative binary operation along those dimensions.

    Attributes:
        SUM: Sum reduction - adds all elements along the specified axis/axes.
             Example: [1, 2, 3] -> 6
        MAX: Maximum reduction - finds the maximum element along the specified axis/axes.
             Example: [1, 5, 3] -> 5
    """
    SUM = auto()
    MAX = auto()  # noqa: E702


class TernaryOps(Enum):
    """
    Enum class representing ternary operations (operations with three inputs).

    Attributes:
        MULACC: Multiply-accumulate operation. Computes a * b + c efficiently in one operation.
                This is a fundamental operation for matrix multiplication and convolutions.
                Common in hardware accelerators (FMA - Fused Multiply-Add).
        WHERE: Conditional selection operation. Given condition c, returns a if c else b.
               Enables element-wise conditional logic: where(c, a, b) = c ? a : b
               Example: where([True, False], [1, 2], [3, 4]) -> [1, 4]
    """
    MULACC = auto()
    WHERE = auto()  # noqa: E702


class MovementOps(Enum):
    """
    Enum class representing movement/view operations.

    These operations change how tensor data is viewed or accessed without modifying
    the underlying data. They manipulate the tensor's shape, stride, and indexing.

    Attributes:
        RESHAPE: Changes the shape of the tensor while preserving total element count.
                 Example: (2, 6) -> (3, 4) or (12,)
        PERMUTE: Rearranges the dimensions (axes) of the tensor.
                 Example: (batch, height, width, channels) -> (batch, channels, height, width)
                 Also known as transpose when swapping two dimensions.
        EXPAND: Broadcasts a tensor along dimensions of size 1 to a larger size.
                Example: (1, 3) -> (5, 3) by repeating the data.
                This is memory-efficient as no data is copied.
        PAD: Adds padding (usually zeros) around the tensor borders.
             Example: Used in convolutions to control output size.
        SHRINK: Extracts a slice/subset of the tensor along one or more dimensions.
                Example: tensor[2:5, 3:7] extracts a rectangular region.
        STRIDE: Changes the step size when iterating through dimensions.
                Example: tensor[::2] takes every other element (stride=2).
                Enables operations like downsampling without copying data.
    """
    RESHAPE = auto()
    PERMUTE = auto()
    EXPAND = auto()
    PAD = auto()
    SHRINK = auto()
    STRIDE = auto()  # noqa: E702


class LoadOps(Enum):
    """
    Enum class representing different load operations.
    
    Attributes:
        EMPTY: Represents an empty load operation.
        RAND: Represents a random load operation.
        CONST: Represents a constant load operation.
        FROM: Represents a load operation from a specific source.
        CONTIGUOUS: Represents a load operation with contiguous data.
        CUSTOM: Represents a custom load operation.
    """
    EMPTY = auto()
    RAND = auto()
    CONST = auto()
    FROM = auto()
    CONTIGUOUS = auto()
    CUSTOM = auto()  # noqa: E702


class Device:
    """
    Represents a device for computation.

    Attributes:
        DEFAULT (str): The default device.
        _buffers (List[str]): The list of available devices.

    Methods:
        canonicalize(device: Optional[str]) -> str: Returns the canonicalized device name.
    """
    DEFAULT = "CPU"
    _buffers = ["CPU"]

    @staticmethod
    def canonicalize(device: Optional[str]) -> str:
        """
        Convert a device specification to its canonical form.

        This method normalizes device names to a standard format. In this minimal
        implementation, all devices are mapped to "CPU".

        Args:
            device: Device name string (e.g., "CPU", "GPU", "cuda:0") or None

        Returns:
            str: Canonicalized device name. Currently always returns "CPU".

        Note:
            In a full implementation, this would:
            - Normalize device names (e.g., "CUDA" -> "GPU:0")
            - Handle device indices (e.g., "GPU:2")
            - Validate device availability
            - Apply default device if None is passed
        """
        return "CPU"
