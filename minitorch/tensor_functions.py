"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    # from typing import Any, List, Tuple
    from typing import Any, List, Tuple, Optional

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Performs the forward pass of the negation function.

        Args:
        ----
            ctx: context object to store any values for the backward pass.
            t1: the input tensor to the negation function.

        Returns:
        -------
            The result of the negation function $-a$.

        """
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Performs the backward pass of the negation function.

        Args:
        ----
            ctx: context object containing saved values from the forward pass.
            grad_output: the gradient of the output with respect to the final objective.

        Returns:
        -------
            The gradient with respect to the input.

        """
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Performs the forward pass of the inverse function.

        Args:
        ----
            ctx: context object to store any values for the backward pass.
            t1: the input tensor to the inverse function.

        Returns:
        -------
            The result of the inverse function $1/a$.

        """
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Performs the backward pass of the inverse function.

        Args:
        ----
            ctx: context object containing saved values from the forward pass.
            grad_output: the gradient of the output with respect to the final objective.

        Returns:
        -------
            The gradient with respect to both inputs.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Performs the forward pass of the addition function.

        Args:
        ----
            ctx: context object to store any values for backward pass.
            t1: the first input tensor to the addition function.
            t2: the second input tensor to the addition function.

        Returns:
        -------
            The result of the addition $a + b$.

        """
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Performs the backward pass of the addition function.

        Args:
        ----
            ctx: the context object containing saved values from the forward pass.
            grad_output: the gradient of the output with respect to the final objective.

        Returns:
        -------
            A tuple containing the gradients with respect to both inputs.

        """
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Optional[Tensor] = None) -> Tensor:
        """Return 1 if all are true"""
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


# TODO: Implement for Task 2.3.
class Mul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Performs the forward pass of the multiplication function.

        Args:
        ----
            ctx: context object to store any values for backward pass.
            t1: the first input tensor to the multiplication function.
            t2: the second input tensor to the multiplication function.

        Returns:
        -------
            The result of the multiplication $a * b$.

        """
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Performs the backward pass of the multiplication function.

        Args:
        ----
            ctx: context object containing saved values fromm the forward pass.
            grad_output: the gradient of the output with respect to the final objective.

        Returns:
        -------
            A tuple containing the gradients with respect to both inputs.

        """
        (t1, t2) = ctx.saved_values
        return t2 * grad_output, t1 * grad_output
        # grad_t1 = grad_output.f.mul_zip(t2, grad_output)
        # grad_t2 = grad_output.f.mul_zip(t1, grad_output)
        # return grad_t1, grad_t2


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Performs the forward pass of the sigmoid function.

        Args:
        ----
            ctx: context obeject to store any values for the backward pass.
            t1: the input tensor to the sigmoid function.

        Returns:
        -------
            The result of the sigmoid function $sigmoid(a)$.

        """
        out = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(out)
        return out
        # ctx.save_for_backward(t1)
        # return t1.f.sigmoid_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Performs the backward pass of the sigmoid function.

        Args:
        ----
            ctx: context object containing the saved values from the forward pass.
            grad_output: the gradient of the output with respect to the final objective.

        Returns:
        -------
            The gradient with respect to the input.

        """
        sigma: Tensor = ctx.saved_values[0]
        # one: Tensor = Tensor()

        data = [1.0] * int(operators.prod(sigma.shape))
        ones = minitorch.Tensor.make(data, sigma.shape, backend=sigma.backend)
        # sigma * (ones - sigma) * grad_output
        return sigma * (ones - sigma) * grad_output


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Performs the forward pass of the relu function.

        Args:
        ----
            ctx: context object to store any values for the backward pass.
            t1: the input tensor to the relu function.

        Returns:
        -------
            The result of the relu function $relu(a)$.

        """
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Performs the backward pass of the relu function.

        Args:
        ----
            ctx: context object containing the saved values from the forward pass.
            grad_output: the gradient of the output with respect to the final objective.

        Returns:
        -------
            The gradient with respect to the input.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.relu_back_zip(t1, grad_output)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Performs the forward pass of the log function.

        Args:
        ----
            ctx: context object to store any values for backward pass.
            t1: the input tensor to the log function.

        Returns:
        -------
            The result of the log function $log(a)$.

        """
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Performs the backward pass of the log function.

        Args:
        ----
            ctx: context object containing saved values from the forward pass.
            grad_output: the gradient of the output with respect to the final objective.

        Returns:
        -------
            The gradient with respect to both inputs.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.log_back_zip(t1, grad_output)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Performs the forward pass of the exponential function.

        Args:
        ----
            ctx: context object to store any values for the backward pass.
            t1: the input tensor to the exponential function.

        Returns:
        -------
            The result of the exponential function $exp(a)$.

        """
        out = t1.f.exp_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Performs the backward pass of the exponential function.

        Args:
        ----
            ctx: context object containg the saved values from the forward pass.
            grad_output: the gradient of the output with respect to the final objective.

        Returns:
        -------
            THe gradient with respect to the input.

        """
        out: Tensor = ctx.saved_values[0]
        return out * grad_output
        # return grad_output.f.mul_zip(grad_output, out)
        # return grad_output.f.mul_zip(out.f.exp_map(out), grad_output)


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor) -> Tensor:
        """Sum along a specified dimension.

        Args:
        ----
            ctx: context object to store any values for the backward pass.
            t1: the input tensor.
            dim: the dimension to sum about.

        Returns:
        -------
            The result of the exponential function $exp(a)$.

        """
        reduce_dim = int(dim.item())
        ctx.save_for_backward(t1, reduce_dim)
        return t1.f.add_reduce(t1, reduce_dim)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for Sum.

        Args:
        ----
            ctx: context object containg the saved values from the forward pass.
            grad_output: the gradient of the output with respect to the final objective.

        Returns:
        -------
            A tuple containg the gradient with respect to the input.

        """
        input, reduce_dim = ctx.saved_values
        # Reshape grad_output to insert singleton dimension at reduce_dim
        grad_output_reshaped = grad_output.view(
            *grad_output.shape[:reduce_dim], 1, *grad_output.shape[reduce_dim:]
        )
        # Create a ones tensor with the original shape
        data = [1.0] * int(operators.prod(input.shape))
        one = minitorch.Tensor.make(data, input.shape, backend=input.backend)
        # Multiply ones by grad_output_reshaped to broadcast the gradient
        # grad_input = grad_output_reshaped.f.mul_zip(grad_output_reshaped, one)
        return grad_output_reshaped * one, grad_output.zeros()
        # return grad_input, grad_output.zeros()


class LT(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Performs the forward pass of the less than function.

        Args:
        ----
            ctx: context object to store any values for the backward pass.
            t1: the first input tensor to the less than function.
            t2: the second input tensor to the less than function.

        Returns:
        -------
            The result of the less than function $a < b$.

        """
        ctx.save_for_backward(t1, t2)
        return t1.f.lt_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Performs the backward pass of the less than function.

        Args:
        ----
            ctx: context object containing the saved values from the forward pass.
            grad_output: the gradient of the output with respect to the final objective.

        Returns:
        -------
            A tuple containing the gradient with respect to the inputs.

        """
        t1, t2 = ctx.saved_values
        out1 = zeros(t1.shape, t1.backend)
        out2 = zeros(t2.shape, t2.backend)
        return out1, out2


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Performs the forward pass of the equal function.

        Args:
        ----
            ctx: context object to store any values for the backward pass.
            t1: the first input tensor to the equal function.
            t2: the second input tensor to the equal function.

        Returns:
        -------
            The result of the equal function $a == b$.

        """
        ctx.save_for_backward(t1, t2)
        return t1.f.eq_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Performs the backward pass of the equal function.

        Args:
        ----
            ctx: context object containing the saved values from the forward pass.
            grad_output: the gradient of the output with respect to the final objective.

        Returns:
        -------
            A tuple containing the gradient with respect to the inputs.

        """
        t1, t2 = ctx.saved_values
        out1 = zeros(t1.shape, t1.backend)
        out2 = zeros(t2.shape, t2.backend)
        return out1, out2


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Performs the forward pass of the isClose function.

        Args:
        ----
            ctx: context object to store any values for the backward pass.
            t1: the first input tensor to the isClose function.
            t2: the second input tensor to the isClose function.

        Returns:
        -------
            The result of the isClose function.

        """
        return t1.f.is_close_zip(t1, t2)


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dims: Tensor) -> Tensor:
        """Performs the forward pass of the permute function.

        Args:
        ----
            ctx: context object to store any values for the backward pass.
            t1: the input tensor to the permute function.
            dims: the new dimensions to pernute the tensor by.

        Returns:
        -------
            The result of the permute function.

        """
        perm = tuple(int(dims[i]) for i in range(dims.size))
        ctx.save_for_backward(perm, t1)

        t1._tensor = t1._tensor.permute(*perm)
        return t1

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Performs the backward pass of the permute function.

        Args:
        ----
            ctx: context object containing the saved values from the forward pass.
            grad_output: the gradient of the output with respect to the final objective.

        Returns:
        -------
            A tuple containing the gradient with respect to the inputs.

        """
        perm, input = ctx.saved_values
        # get original dims
        inv_dims = [0] * len(perm)
        for i, dim in enumerate(perm):
            inv_dims[dim] = i

        # inv_dims = tuple(perm.index(i) for i in range(len(dims)))
        grad_output._tensor = grad_output._tensor.permute(*inv_dims)
        # return Tensor(grad_output._tensor.permute(*inverse_dims))

        return grad_output, 0.0


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Performs the forward pass of the view function.

        Args:
        ----
            ctx: context object to store any values for the backward pass.
            a: the input tensor to the permute function.
            shape: the new shape of the tensor.

        Returns:
        -------
            The result of the view function.

        """
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Matrix Multiply backward (module 3)"""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Computes the gradient of a function using central difference approximation.

    This method perturbs the specified argument by a small amount (epsilon)
    in both the positive and negative directions and calculates the
    approximate derivative.

    Args:
    ----
        f: The function for which to compute the gradient.
        *vals: The input tensors to the function.
        arg: The index of the argument to perturb. Defaults to 0.
        epsilon : The small perturbation value. Defaults to 1e-6.
        ind: The specific index in the argument tensor to perturb.

    Returns:
    -------
        float: The estimated gradient of the function with respect to the specified argument.

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
