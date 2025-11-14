"""
Helper Utilities and Data Type System for TeenyGrad

This module provides essential utility functions and the data type system used
throughout TeenyGrad. It includes:

1. Utility Functions:
   - String/list manipulation: dedup, flatten
   - Argument processing: argfix, make_pair
   - Numeric utilities: argsort, all_int, round_up
   - Environment variables: getenv, DEBUG, CI

2. Data Type System (DType and dtypes):
   - Unified data type representation across backends
   - Support for standard types: float16/32/64, int8/16/32/64, uint8/16/32/64, bool
   - Type conversion between numpy and internal representations
   - Type checking utilities: is_int, is_float, is_unsigned

3. Special Data Types:
   - ImageDType: For image-specific tensor representations
   - PtrDType: For pointer/memory address types

The data type system is crucial for:
- Cross-backend compatibility (CPU, GPU, etc.)
- Type promotion rules in mixed-type operations
- Memory layout and size calculations
- Kernel generation and optimization

Environment Variables:
- DEBUG: Controls debug output level (0=none, higher=more verbose)
- CI: Set in continuous integration environments
"""
from typing import Union, Tuple, Iterator, Optional, Final, Any
import os, functools, platform
import numpy as np
from math import prod  # noqa: F401 # pylint:disable=unused-import
from dataclasses import dataclass

# Platform detection for OS-specific behavior
OSX = platform.system() == "Darwin"


def dedup(x):
    """
    Remove duplicate elements from a list while preserving the order.

    Args:
        x: The input list.

    Returns:
        A new list with duplicate elements removed.
    """
    return list(dict.fromkeys(x))


def argfix(*x):
    """
    Fix the argument format.

    If the first argument is a tuple or list, it is returned as is.
    Otherwise, the arguments are returned as a tuple.

    Args:
        *x: Variable number of arguments.

    Returns:
        A tuple if the first argument is not a tuple or list, otherwise the first argument itself.
    """
    return tuple(x[0]) if x and x[0].__class__ in (tuple, list) else x


def make_pair(x: Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]:
    """
    Create a pair or tuple of the given value.

    If the input is an integer, it is repeated `cnt` times to create a tuple.
    If the input is already a tuple, it is returned as is.

    Args:
        x: The input value.
        cnt: The number of times to repeat the input value. Default is 2.

    Returns:
        A tuple containing the input value repeated `cnt` times.
    """
    return (x, ) * cnt if isinstance(x, int) else x


def flatten(l: Iterator):
    """
    Flatten a nested list.

    Args:
        l: The nested list.

    Returns:
        A flattened list.
    """
    return [item for sublist in l for item in sublist]


def argsort(x):
    """
    Sort the indices of a list based on the corresponding values.

    Args:
        x: The input list.

    Returns:
        A new list containing the sorted indices.
    """
    return type(x)(
        sorted(range(len(x)), key=x.__getitem__)
    )


def all_int(t: Tuple[Any, ...]) -> bool:
    """
    Check if all elements in a tuple are integers.

    Args:
        t: The input tuple.

    Returns:
        True if all elements are integers, False otherwise.
    """
    return all(isinstance(s, int) for s in t)


def round_up(num, amt: int):
    """
    Round up a number to the nearest multiple of a given amount.

    Args:
        num: The input number.
        amt: The amount to round up to.

    Returns:
        The rounded up number.
    """
    return (num + amt - 1) // amt * amt


@functools.lru_cache(maxsize=None)
def getenv(key, default=0):
    """
    Get the value of an environment variable.

    Args:
        key: The name of the environment variable.
        default: The default value to return if the environment variable is not set. Default is 0.

    Returns:
        The value of the environment variable, or the default value if it is not set.
    """
    return type(default)(os.getenv(key, default))


DEBUG = getenv("DEBUG")
CI = os.getenv("CI", "") != ""


@dataclass(frozen=True, order=True)
class DType:
    """
    Represents a data type.

    Attributes:
        priority: The priority of the data type.
        itemsize: The size of each item in the data type.
        name: The name of the data type.
        np: The corresponding NumPy data type. (Optional)
        sz: The size of the data type. Default is 1.
    """
    priority: int
    itemsize: int
    name: str
    np: Optional[type]
    sz: int = 1

    def __repr__(self):
        return f"dtypes.{self.name}"


class dtypes:
    """
    Contains predefined data types.

    Attributes:
        bool: The boolean data type.
        float16: The 16-bit floating-point data type.
        half: Alias for float16.
        float32: The 32-bit floating-point data type.
        float: Alias for float32.
        float64: The 64-bit floating-point data type.
        double: Alias for float64.
        int8: The 8-bit integer data type.
        int16: The 16-bit integer data type.
        int32: The 32-bit integer data type.
        int64: The 64-bit integer data type.
        uint8: The 8-bit unsigned integer data type.
        uint16: The 16-bit unsigned integer data type.
        uint32: The 32-bit unsigned integer data type.
        uint64: The 64-bit unsigned integer data type.
        bfloat16: The 16-bit brain floating-point data type.
    """

    @staticmethod
    def is_int(x: DType) -> bool:
        """
        Check if a data type is an integer.

        Args:
            x: The data type.

        Returns:
            True if the data type is an integer, False otherwise.
        """
        return x in (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64,
                     dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)

    @staticmethod
    def is_float(x: DType) -> bool:
        """
        Check if a data type is a floating-point number.

        Args:
            x: The data type.

        Returns:
            True if the data type is a floating-point number, False otherwise.
        """
        return x in (dtypes.float16, dtypes.float32, dtypes.float64)

    @staticmethod
    def is_unsigned(x: DType) -> bool:
        """
        Check if a data type is unsigned.

        Args:
            x: The data type.

        Returns:
            True if the data type is unsigned, False otherwise.
        """
        return x in (dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)

    @staticmethod
    def from_np(x) -> DType:
        """
        Get the corresponding DType object for a given NumPy data type.

        Args:
            x: The NumPy data type.

        Returns:
            The corresponding DType object.
        """
        return DTYPES_DICT[np.dtype(x).name]

    bool: Final[DType] = DType(0, 1, "bool", np.bool_)
    float16: Final[DType] = DType(9, 2, "half", np.float16)
    half = float16
    float32: Final[DType] = DType(10, 4, "float", np.float32)
    float = float32
    float64: Final[DType] = DType(11, 8, "double", np.float64)
    double = float64
    int8: Final[DType] = DType(1, 1, "char", np.int8)
    int16: Final[DType] = DType(3, 2, "short", np.int16)
    int32: Final[DType] = DType(5, 4, "int", np.int32)
    int64: Final[DType] = DType(7, 8, "long", np.int64)
    uint8: Final[DType] = DType(2, 1, "unsigned char", np.uint8)
    uint16: Final[DType] = DType(4, 2, "unsigned short", np.uint16)
    uint32: Final[DType] = DType(6, 4, "unsigned int", np.uint32)
    uint64: Final[DType] = DType(8, 8, "unsigned long", np.uint64)

    # NOTE: bfloat16 isn't supported in numpy
    bfloat16: Final[DType] = DType(9, 2, "__bf16", None)


DTYPES_DICT = {
    k: v
    for k, v in dtypes.__dict__.items() if not k.startswith('__')
    and not callable(v) and not v.__class__ == staticmethod
}

PtrDType, ImageDType, IMAGE = None, None, 0  # junk to remove
