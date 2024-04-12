# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Helper functions for calibration directions.
"""

from jax.tree_util import tree_flatten
from jax.flatten_util import ravel_pytree
from jax import vmap, jacrev
import jax.numpy as jnp
from jax import tree_map


class SVDCalibrationDirections:
    """Class for storing and working with SVD directions."""

    def __init__(self, *args, f, real_output=False):
        
        _, (U, S, Vh) = _pytree_jac_svd(*args, f=f, real_output=real_output)
        self.U = U
        self.S = S
        self.Vh = Vh
        self.args = args
    
    def right_SV_linear_combo(self, coeffs, translate_args=False):
        """Take a linear combination of right singular vectors. If ``translate_args==True``, the
        original args at which the SVD was computed are added.
        """
        linear_combo = tree_map(
            lambda x: jnp.tensordot(coeffs, x[:len(coeffs)], axes=(0, 0)),
            self.Vh
        )
        if not translate_args:
            return linear_combo
        
        flat_args, tree_def = tree_flatten(*self.args)
        flat_combo, _ = tree_flatten(linear_combo)

        flat_output = [a + b for a, b in zip(flat_args, flat_combo)]
        return tree_def.unflatten(flat_output)

    def left_SV_linear_combo(self, coeffs):
        return tree_map(lambda x: jnp.tensordot(coeffs, x[..., :len(coeffs)], axes=(0, -1)), self.U)


def _pytree_jac_svd(*args, f, real_output=False):
    """Given a function ``f`` with real arguments, compute ``f(*args)`` as well as the SVD of
    ``Df(*args)`` (where ``Df`` if the Jacobian of ``f``).

    Note that this function works for ``f`` having arbitrary PyTree inputs and outputs (with the
    inputs being real). In the simplest case, when ``f`` has a single 1d array argument ``a``, and
    returns a single 1d array, the SVD portion of the output is the same as
    ``jax.numpy.linalg.svd(jax.jacrev(f)(a))``.

    In the case of a arbitrary PyTree inputs and outputs, the SVD portion behaves as expected when
    interpreting the PyTree structure (of either the input or output) as an inner-product space
    where the inner product is defined by "raveling" all arrays in the pytree into a single vector
    and taking the standard inner-product. The returned ``Vh`` is single PyTree with same container
    structure as ``args``, with the right singular vectors stacked in the 0th index of all of the
    arrays, and ``U`` is a single PyTree with the same container structure as ``f(*args)``, with
    the left singular vectors stacked in the last index.
    
    E.g. to take a linear combination of all right singular vectors: 
    ``tree_map(lambda x: jnp.tensordot(c, x, axis=(0, 0)))(Vh)``, where ``c`` is the 1d coefficient
    array.
    
    Args:
        *args: Positional arguments to ``f``.
        f: The function.
        real_output: Whether or not the output of ``f`` can be assumed to be real. If ``True``, this 
            saves the cost of storing a complex part to the output.
    Returns:
        f(*args), (U, S, Vh): Where (U, S, Vh) is the PyTree SVD of ``Df(*args)``.
    """

    output = f(*args)
    
    raveled_args, unravel_args = ravel_pytree(args)
    _, unravel_output = ravel_pytree(output)
    
    def flat_func(flat_args):
        unraveled_args = unravel_args(flat_args)
        output, _ = ravel_pytree(f(*unraveled_args))
        
        if not real_output:
            return jnp.append(output.real, output.imag)
        return output
    
    Df = jacrev(flat_func)(raveled_args)
    
    U, S, Vh = jnp.linalg.svd(Df, full_matrices=False)

    if not real_output:
        cut = int(len(U) / 2)
        U = U[0:cut] + 1j * U[cut:]
    
    Vh_out = vmap(unravel_args)(Vh)

    # if len(args) == 1, unwrap the tuple
    if len(args) == 1:
        Vh_out = Vh_out[0]

    return output, (vmap(unravel_output, in_axes=1, out_axes=-1)(U), S, Vh_out)