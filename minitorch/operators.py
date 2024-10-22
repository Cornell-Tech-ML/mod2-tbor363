"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    """Multiples two numbers.

    Args:
    ----
      x: first input value.
      y: second input value.

    Returns:
    -------
      x*y.

    """
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged.

    Args:
    ----
      x: input value.

    Returns:
    -------
      x unchanged.

    """
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers together.

    Args:
    ----
      x: first input value.
      y: second input value.

    Returns:
    -------
      x+y.

    """
    return x + y


def neg(x: float) -> float:
    """Negates input number.

    Args:
    ----
      x: input value.

    Returns:
    -------
      -x.

    """
    return -x


def lt(x: float, y: float) -> float:
    """Checks if x is less than y.

    Args:
    ----
      x: first input value.
      y: second input value.

    Returns:
    -------
      True if x < y, False otherwise.

    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Checks if two numbers are equal.

    Args:
    ----
      x: first input value.
      y: second input value.

    Returns:
    -------
      True if x == y, False otherwise.

    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Returns largest of two numbers.

    Args:
    ----
      x: first input value.
      y: second input value.

    Returns:
    -------
      x if x > y, y otherwise.

    """
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close to each other.

    Args:
    ----
      x: first input value.
      y: second input value.

    Returns:
    -------
      True if |x - y| < 1e-2, False otherwise.

    """
    # $f(x) = |x - y| < 1e-2$
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function.

    Args:
    ----
      x: input value.

    Returns:
    -------
      output to the sigmoid function.

    """
    # $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return (math.exp(x)) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLu activation function.

    Args:
    ----
      x: input value
    Returns:
      output of the ReLU function. max(0, x).

    """
    return x if x > 0.0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Calculats the natural logarithm.

    Args:
    ----
      x: input value
    Returns:
      ln(x).

    """
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Calculates the exponential function.

    Args:
    ----
      x: input value.

    Returns:
    -------
      e^x.

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal.

    Args:
    ----
      x: input value.

    Returns:
    -------
      1/x.

    """
    return 1.0 / x if x != 0.0 else 0.0


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log times a second arg.

    Args:
    ----
      x: first argument.
      y: second argument.

    Returns:
    -------
      d(log(x)*y)/dx.

    """
    return y / (x + EPS)


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of reciprocal times a second arg.

    Args:
    ----
      x: first argument.
      y: second argument.

    Return:
    ------
      d(1/x*y)/dx.

    """
    return -(1.0 / x**2) * y


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of ReLU times a second arg.

    Args:
    ----
      x: first argument.
      y: second argument.

    Return:
    ------
      d(ReLU(x)*y)/dx.

    """
    return y if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


# def map(a: Iterable[float], f: Callable[[float], float]) -> Iterable[float]:
#     """Higher-order function that applies a given function to each element of an iterable.

#     Args:
#     ----
#       a: input iterable.
#       f: function that will be applied to each element.

#     Returns:
#     -------
#       new iterable with f applied on all elements of ls.


#     """
#     out = []
#     for x in a:
#         out.append(f(x))
#     return out
def map(f: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order function that applies a given function to each element of an iterable.

    Args:
    ----
      f: function from one value to one value.

    Returns:
    -------
      A function that takes a list, applies 'fn' to each element, and returns a new list.

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(f(x))
        return ret

    return _map


# def zipWidth(
#     a: Iterable[float], b: Iterable[float], f: Callable[[float, float], float]
# ) -> Iterable[float]:
#     """Higher-order function that combines elements from two iterables using a given function.

#     Args:
#     ----
#       a: first input iterable.
#       b: second input iterable.
#       f: function that takes in to arguments to be applied on a, b.

#     Returns:
#     -------
#       new iterable from f(a, b).


#     """
#     out = []
#     for x, y in zip(a, b):
#         out.append(f(x, y))
#     return out
def zipWidth(
    f: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order function that combines elements from two iterables using a given function.

    Args:
    ----
      f: combine two vbalues.

    Returns:
    -------
      Function that takes two equally sized lists 'ls1' and 'ls2', produce a new list
      applying f(x, y) on each pair of elements.

    """

    def _zipWidth(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(f(x, y))
        return ret

    return _zipWidth


# def reduce(a: Iterable[float], f: Callable[[float, float], float], s: float) -> float:
#     """Higher-order function that reduces an iterable to a single value using a given function.

#     Args:
#     ----
#       a: input iterable.
#       f: function to reduce iterable.
#       s: starting point.

#     Returns:
#     -------
#       output value from f(a).


#     """
#     for x in a:
#         s = f(s, x)
#     return s
def reduce(
    f: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Higher-order function that reduces an iterable to a single value using a given function.

    #     Args:
    #     ----
    #       f: combine two values.
    #       start: start value x_0.

    #     Returns:
    #     -------
    #       Function that takes a list 'ls' of elements and computes the reduction
    #
    """

    def _reduce(ls: Iterable[float]) -> float:
        ret = start
        for x in ls:
            ret = f(ret, x)
        return ret

    return _reduce


def negList(a: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list.

    Args:
    ----
      a: input list.

    Returns:
    -------
      list a with all values negated.

    """
    return map(neg)(a)


def addLists(a: Iterable[float], b: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists.

    Args:
    ----
      a: first input list.
      b: second input list.

    Returns:
    -------
      list made from adding elements in a and b together.

    """
    return zipWidth(add)(a, b)


def sum(a: Iterable[float]) -> float:
    """Sum all elements in a list.

    Args:
    ----
      a: input list
    Returns:
      sum of all elements of a

    """
    # reduce only gets functoon and starting poitn
    return reduce(add, 0.0)(a)


def prod(a: Iterable[float]) -> float:
    """Calculate the product of all elements in a list
    Args:
      a: input list
    Returns:
      product of all elements of a
    """
    return reduce(mul, 1.0)(a)
