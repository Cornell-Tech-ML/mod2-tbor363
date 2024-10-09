from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.
class Mul(ScalarFunction):
    """Multiplication Function"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        ctx.save_for_backward(a, b)
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        a, b = ctx.saved_values
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    """Inverse Function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation Function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = sigmoid(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Performs the forward pass of the sigmoid function.

        Args:
        ----
            ctx: context obeject to store any values for the backward pass.
            a: the input value to the sigmoid function.

        Returns:
        -------
            The result of the sigmoid function $sigmoid(a)$.

        """
        out = operators.sigmoid(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Performs the backward pass of the sigmoid function.

        Args:
        ----
            ctx: context object containing the saved values from the forward pass.
            d_output: the gradient of the output with respect to the final objective.

        Returns:
        -------
            A tuple containing the gradients with respect to the input.

        """
        sigma: float = ctx.saved_values[0]
        return sigma * (1.0 - sigma) * d_output


class ReLU(ScalarFunction):
    """Relu function $f(x) = Relu(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Performs the forward pass of the relu function.

        Args:
        ----
            ctx: context object to store any values for the backward pass.
            a: the input value to the relu function.

        Returns:
        -------
            The result of the relu function $relu(a)$.

        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Performs the backward pass of the relu function.

        Args:
        ----
            ctx: context object containing the saved values from the forward pass.
            d_output: the gradient of the output with respect to the final objective.

        Returns:
        -------
            A tuple containg the gradients with respect to the input.

        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """exponential function $f(x) = e^x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Performs the forward pass of the exponential function.

        Args:
        ----
            ctx: context object to store any values for the backward pass.
            a: the input value to the exponential function.

        Returns:
        -------
            The result of the exponential function $exp(a)$.

        """
        out = operators.exp(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Performs the backward pass of the exponential function.

        Args:
        ----
            ctx: context object containg the saved values from the forward pass.
            d_output: the gradient of the output with respect to the final objective.

        Returns:
        -------
            A tuple containg the gradient with respect to the input.

        """
        out: float = ctx.saved_values[0]
        return d_output * out


class LT(ScalarFunction):
    """Less than function $f(x, y) = x < y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Performs the forward pass of the less than function.

        Args:
        ----
            ctx: context object to store any values for the backward pass.
            a: the first input value to the less than function.
            b: the second input value to the less than function.

        Returns:
        -------
            The result of the less than function $a < b$.

        """
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Performs the backward pass of the less than function.

        Args:
        ----
            ctx: context object containing the saved values from the forward pass.
            d_output: the gradient of the output with respect to the final objective.

        Returns:
        -------
            A tuple containing the gradient with respect to the inputs.

        """
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function $f(x) = x == y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Performs the forward pass of the equal function.

        Args:
        ----
            ctx: context object to store any values for the backward pass.
            a: the first input value to the equal function.
            b: the second input value to the equal function.

        Returns:
        -------
            The result of the equal function $a == b$.

        """
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Performs the backward pass of the equal function.

        Args:
        ----
            ctx: context object containing the saved values from the forward pass.
            d_output: the gradient of the output with respect to the final objective.

        Returns:
        -------
            A tuple containing the gradient with respect to the inputs.

        """
        return 0.0, 0.0
