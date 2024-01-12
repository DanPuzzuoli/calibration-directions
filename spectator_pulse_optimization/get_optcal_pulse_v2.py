#generate opt-cal pulse
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from scipy.signal import convolve
from qiskit import pulse
from qiskit_dynamics import DiscreteSignal
from jax.scipy.signal import convolve
import pickle

def optcal_pulse(params,amp=1,return_opt=0,filter='new'):
    def discretized_chebyshev(degree, n, T):
        dt = T / n
        coeffs = np.zeros(degree + 1)
        coeffs[-1] = 1.

        return jnp.array(Chebyshev(coeffs, domain=[0, T], window=[-1,+1])(np.linspace(0, T-dt, n) + dt/2 ))

    def discretized_chebyshev_basis(max_degree, n, T):  
        vals = list(range(max_degree + 1))
        disc_leg_map = map(lambda deg: discretized_chebyshev(deg, n, T), vals)
        
        return np.array(list(disc_leg_map))
    

    num_params=len(params)-1    ##Parameter space dimension
    max_deg=int((num_params)/2 -1)    ##max degree of chebyshev basis
    dt = 1/4.5

    n = 48 #n_steps from the job_config.yaml file
    T = dt * n

    if filter == 'old':
        T_gate=np.ceil(3*T-2*T/n)   ##Convolution adds some samples. Final T_gate is equal to this.


        ##creating Chebyshev basis and adding zero pad in the begining
        basis=np.concatenate((np.zeros((int(num_params/2),1)),discretized_chebyshev_basis(max_deg, n-1, T)),1) 

        #Gaussian convolution filters
        def gaus_left(t):
            sigma = 8
            delt = 0.015
            return 2.*delt/jnp.sqrt(2.*np.pi*sigma**2)*jnp.exp(-t**2/(2*sigma**2))
        
        def gaus_right(t):
            sigma = 8
            delt = 0.015
            return 2.*delt/jnp.sqrt(2.*np.pi*sigma**2)*jnp.exp(-(t-T)**2/(2*sigma**2))

        convolution_left =np.array([gaus_left(x) for x in (np.linspace(0, T, n) +T/(2*n))])
        convolution_right=np.array([gaus_right(x) for x in (np.linspace(0, T, n) +T/(2*n))])

        def construct_smooth_sig(params):
            sig=jnp.sum(jnp.multiply(basis, params[0:,None]),0)

            #convolve signal with gaus_left and gaus_right
            smooth_sig=convolve(convolve(sig,convolution_left),convolution_right)

            #bound signal from left and right sides
            sig_bounded=jnp.arctan(smooth_sig) /(np.pi / 2)
            return sig_bounded

        def get_parameterized_signal(params):
            real_sig=construct_smooth_sig(params[0:(max_deg+1)])
            comp_sig=construct_smooth_sig(params[max_deg+1:(2*(max_deg+1))])
            smooth_signal=DiscreteSignal(T/n, real_sig+(1j * comp_sig), carrier_freq=0., phase=0.)
            return smooth_signal

        def get_pulse(params,amp=1):
            samples = get_parameterized_signal(params).samples

            #ensure pulse length is a multiple of 16dt
            pad_length = int(np.ceil(len(samples)/16)*16-len(samples))
            padded_signal = amp * np.array(samples.tolist()+[0.0]*pad_length)
            return pulse.Waveform(padded_signal, name='optcal_X90p', limit_amplitude=False, epsilon = 1e-7)
        
        if return_opt == 0:
            return get_pulse(params,amp)
        if return_opt == 1:
            return get_parameterized_signal(params)

    if filter == 'new':
        nzeros = 24

        zeropad = jnp.zeros((int(num_params/2),nzeros))
        #put zero pad at the beginning and end of the pulse
        basis=jnp.concatenate((zeropad,discretized_chebyshev_basis(max_deg, n, T),zeropad),1)

        def low_pass_filter(signal,cutoff_freq,sample_rate,window_length):
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
            filtered_signal = convolve(signal, window, mode='same')
            return filtered_signal 

        def construct_smooth_sig(params,cutoff_freq):
            sig = jnp.sum(jnp.multiply(basis,params),0)
            sig_filtered = low_pass_filter(sig,cutoff_freq,1/dt,window_length=int(1.9*nzeros))
            sig_bounded=jnp.arctan(sig_filtered) /(np.pi / 2)
            return sig_bounded

        def get_parameterized_signal(params):
            phase = params[-1]
            params = params[:-1]
            cutoff = 0.3 #cutoff frequency in GHz units
            real_sig=construct_smooth_sig(params[0:int(num_params/2),None],cutoff)
            comp_sig=construct_smooth_sig(params[int(num_params/2):,None],cutoff)
            smooth_signal=DiscreteSignal(T/n, real_sig+(1j * comp_sig), carrier_freq=0, phase=phase)
            return smooth_signal

        def get_pulse(params,amp=1):
            samples = get_parameterized_signal(params).samples
            pad_length = int(np.ceil(len(samples)/16)*16-len(samples))
            padded_signal = amp * np.array(samples.tolist()+[0.0]*pad_length)
            return pulse.Waveform(padded_signal, name='optcal_X90p', limit_amplitude=False, epsilon = 1e-7)
        
        if return_opt == 0:
            return get_pulse(params,amp)
        if return_opt == 1:
            return get_parameterized_signal(params)
