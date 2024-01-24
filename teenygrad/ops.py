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
    SUM = auto()
    MAX = auto()  # noqa: E702


class TernaryOps(Enum):
    MULACC = auto()
    WHERE = auto()  # noqa: E702


class MovementOps(Enum):
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
        return "CPU"
