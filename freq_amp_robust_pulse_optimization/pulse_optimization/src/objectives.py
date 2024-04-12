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

from qiskit_dynamics import DYNAMICS_NUMPY as unp


def fidelity(U, V):
    """Computes fidelity between U and V. Note that this works for non-square arrays (of different)
    shapes, by first finding the largest rectangle in both dimensions."""
    d0 = min(U.shape[0], V.shape[0])
    d1 = min(U.shape[1], V.shape[1])
    U = U[0:d0, 0:d1]
    V = V[0:d0, 0:d1]
    return unp.abs((U.conj() * V).sum())**2 / (d0 * d1)


def traceless_hs_norm(A):
    """Computes the Hilbert-Schmidt norm of A - (Tr(A)/d)I, where I is the identity, and d is the
    dimension.
    
    I.e. This computes the norm of the projection of A onto the subspace of traceless matrices
    (hence "traceless"). Note that this generalizes to non-square matrices, with d being the minimum
    of the two dimensions, and I being the identity of that size with padded rows or columns to
    match the shape of A.
    """
    d = min(A.shape)
    return unp.linalg.norm(A - A.trace() * unp.eye(*A.shape, dtype=complex) / d)