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

import jax.numpy as jnp
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from scipy.signal import convolve
from qiskit import pulse
from qiskit_dynamics import DiscreteSignal
from jax.scipy.signal import convolve

from qiskit import QiskitError


def chebyshev_parameterization(
    I_params=None,
    Q_params=None,
    dt=None,
    n_steps=45,
    n_zeros=24,
    cutoff_freq=0.3,
    carrier_freq=0.,
    phase=0.,
    return_waveform=False,
):
    """A commonly used parameterization for a class of smooth and bounded pulse shapes.

    This parameterization consists of the following steps (for both I and Q components):
        - Taking a linear combination of discrete chebyshev polynomials (with no constraints).
        - Passing the samples through a diffeomorphism mapping R -> [-1., 1.]. (Bounded.)
        - Filtering the resulting samples through a low pass filter. (Smooth.)

    For each I/Q component, the number of discretized Chebyshev polynomials used is determined by
    the length of the parameter array. A 

    Args:
        I_params: I quadrature parameters.
        Q_params: Q quadrature parameters.
        dt: Sample width, defaults to 1/4.5.
        n_steps: Number of time steps to use for the discrete Chebyshev polynomials.
        n_zeros: Length of zeros padded before and after envelope before convolution.
        cutoff: Frequency cutoff for low pass filter.
        carrier_freq: Carrier frequency of returned signal.
        phase: Phase between I/Q parts of the signal.
        return_waveform: Whether or not to return as a Qiskit Pulse waveform. If returned as a
            Qiskit Pulse waveform, the carrier_freq and phase arguments have no effect.
    """

    if I_params is None and Q_params is None:
        raise QiskitError("At least one of I_params or Q_params must be specified.")

    if dt is None:
        dt = 1 / 4.5

    samples = jnp.zeros(n_steps + 2 * n_zeros, dtype=complex)

    if I_params is not None:
        samples = _single_quad_chebyshev_parameterization(
            I_params, 
            dt=dt, 
            n_steps=n_steps,
            n_zeros=n_zeros,
            cutoff_freq=cutoff_freq
        )

    if Q_params is not None:
        samples = samples + 1j * _single_quad_chebyshev_parameterization(
            Q_params, 
            dt=dt, 
            n_steps=n_steps,
            n_zeros=n_zeros,
            cutoff_freq=cutoff_freq
        )

    if return_waveform:
        pad_length = int(np.ceil(len(samples) / 16) * 16 - len(samples))
        padded_signal = np.array(samples.tolist() + [0.0] * pad_length)
        return pulse.Waveform(padded_signal, limit_amplitude=False, epsilon=1e-7)

    return DiscreteSignal(dt=dt, samples=samples, carrier_freq=carrier_freq, phase=phase)


def _single_quad_chebyshev_parameterization(
    params,
    dt=None,
    n_steps=45,
    n_zeros=24,
    cutoff_freq=0.3,
):
    if dt is None:
        dt = 1/4.5

    # linear combo of chebyshev
    cheb_basis = _discretized_chebyshev_basis(len(params) - 1, n_steps, dt * n_steps)
    samples = jnp.tensordot(params, cheb_basis, axes=(0, 0))
    
    # bound the signal
    bounded_samples = jnp.arctan(samples) /(np.pi / 2)
    
    # pad with zeros
    zeropad = jnp.zeros(n_zeros)
    padded_samples = jnp.concatenate([zeropad, bounded_samples, zeropad])
    
    # filter
    return _low_pass_filter(padded_samples, cutoff_freq, 1/dt, window_length=int(1.9 * n_zeros))


def _discretized_chebyshev(degree, n, T):
    dt = T / n
    coeffs = np.zeros(degree + 1)
    coeffs[-1] = 1.

    return jnp.array(Chebyshev(coeffs, domain=[0, T], window=[-1,+1])(np.linspace(0, T-dt, n) + dt/2 ))


def _discretized_chebyshev_basis(max_degree, n, T):  
    vals = list(range(max_degree + 1))
    disc_leg_map = map(lambda deg: _discretized_chebyshev(deg, n, T), vals)
    
    return jnp.array(list(disc_leg_map))


def _low_pass_filter(samples, cutoff_freq, sample_rate, window_length):
    # Calculate the Nyquist frequency (half of the sample rate)
    nyquist_freq = 0.5 * sample_rate

    # Normalize the cutoff frequency with respect to the Nyquist frequency
    normalized_cutoff = cutoff_freq / nyquist_freq

    # Design the low-pass filter using a simple windowed sinc function
    window_length = window_length  # Adjust the window length as needed for the filter's characteristics
    t = jnp.linspace(-1, 1, window_length)
    window = jnp.sinc(2 * jnp.pi * normalized_cutoff * t)
    window = jnp.where(jnp.isnan(window), 1.0, window)  # Avoid division by zero

    # Normalize the window to have unity gain
    window /= jnp.sum(window)

    # Convolve the signal with the filter window
    filtered_signal = convolve(samples, window, mode='same')
    return filtered_signal