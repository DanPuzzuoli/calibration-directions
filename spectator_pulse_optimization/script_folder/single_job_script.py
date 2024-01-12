###############################################################
# python script for running and saving results of optimizations
###############################################################

import argparse

# this needs to happen even before the opt_state gets imported
# as this can contain JAX arrays
import os
os.environ["JAX_JIT_PJIT_API_MERGE"] = "0"

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

from qiskit_dynamics.array import Array
Array.set_default_backend("jax")

def run_optimization(x0, config):
    """If x0 is None, x0 will be generated randomly.
    """
    import jax.numpy as jnp
    import numpy as np
    from numpy.polynomial.chebyshev import Chebyshev

    def discretized_chebyshev(degree, n, T):
        dt = T / n
        coeffs = jnp.zeros(degree + 1)
        coeffs = coeffs.at[-1].set(1.)

        return jnp.array(Chebyshev(coeffs, domain=[0, T], window=[-1,+1])(jnp.linspace(0, T-dt, n) + dt/2 ))

    def discretized_chebyshev_basis(max_degree, n, T):
        vals = list(range(max_degree + 1))
        disc_leg_map = map(lambda deg: discretized_chebyshev(deg, n, T), vals)
        
        return jnp.array(list(disc_leg_map))

    dim = config["dim"]

    v = config["v"]
    anharm = config["anharm"]
    r = config["r"]

    v_spec = config["v_spec"]

    a = np.diag(np.sqrt(np.arange(1, dim)), 1)
    adag = np.diag(np.sqrt(np.arange(1, dim)), -1)
    N = np.diag(np.arange(dim))
    X = a + adag

    # static part
    static_hamiltonian = 2 * np.pi * v * N + np.pi * anharm * N * (N - np.eye(dim))
    # drive term
    drive_hamiltonian = 2 * np.pi * r * X

    from qiskit_dynamics.models import RotatingFrame

    rotating_frame = RotatingFrame(static_hamiltonian)  

    num_params = config["num_params"]    ##Parameter space dimension
    max_deg=int((num_params)/2 -1)    ##max degree of chebyshev basis
    dt = 1/4.5

    #T=5.27*0.25/r       ##Domain of basis function
    n = config["n_steps"]         ##number of samples, affects the Total duration of the pulse
    T = dt * n

    basis=jnp.concatenate((jnp.zeros((int(num_params/2),1)),discretized_chebyshev_basis(int(num_params/2)-1, n-1, T)),1) 

    def gaus_left(t):
        sigma = 8
        delt = 0.015
        return 2.*delt/np.sqrt(2.*np.pi*sigma**2)*np.exp(-t**2/(2*sigma**2))
    def gaus_right(t):
        sigma = 8
        delt = 0.015
        return 2.*delt/np.sqrt(2.*np.pi*sigma**2)*np.exp(-(t-T)**2/(2*sigma**2))

    convolution_left =jnp.array([gaus_left(x) for x in (np.linspace(0, T, n) +T/(2*n))])
    convolution_right=jnp.array([gaus_right(x) for x in (np.linspace(0, T, n) +T/(2*n))])

    from qiskit_dynamics import DiscreteSignal
    from jax.scipy.signal import convolve
    def construct_smooth_sig(params):
        sig=jnp.sum(jnp.multiply(basis, params[0:,None]),0)
        smooth_sig=convolve(convolve(sig,convolution_left),convolution_right)
        sig_bounded=jnp.arctan(smooth_sig) /(np.pi / 2)
        return sig_bounded

    def get_parameterized_signal(params):
        phase = params[-1]
        params = params[:-1]
        real_sig=construct_smooth_sig(params[0:(max_deg+1)])
        comp_sig=construct_smooth_sig(params[max_deg+1:(2*(max_deg+1))])
        smooth_signal=DiscreteSignal(T/n, real_sig+(1j * comp_sig), carrier_freq=v, phase=phase)
        return smooth_signal
    
    test_sig = get_parameterized_signal(np.random.uniform(-1e4,1e4,num_params + 1))
    T_gate = len(test_sig.samples) * dt

    iX = -1j * 2 * np.pi * r * Array(X)
    iN = -1j * 2 * np.pi * v * Array(N)

    w_spec = 1j * 2 * np.pi * Array(v_spec)
    ident = Array(np.eye(dim, dtype=complex))

    def construct_generator_parts(params):
        
        sig = get_parameterized_signal(params)
        #sig.phase=phase
        
        def unperturbed_generator(t):
            return (sig(t) * rotating_frame.operator_into_frame(t, iX)).data
        
        # frequency perturbations
        def perturbZ(t):
            return (rotating_frame.operator_into_frame(t, iN)).data
        
        # drive strength perturbations
        def perturbsX(t):                     
            return (sig(t) * rotating_frame.operator_into_frame(t, iX)).data
        
        # dressing perturbations
        def perturbsZ(t):
            return (np.exp(w_spec * t) * sig(t) * rotating_frame.operator_into_frame(t, iN)).data
        
        def perturbmsZ(t):
            return (np.exp(-w_spec * t) * sig(t) * rotating_frame.operator_into_frame(t, iN)).data
        
        def perturbs1(t):
            return (np.exp(w_spec * t) * sig(t) * ident).data
        
        def perturb3(t):
            return static_hamiltonian + 0j
        
        return (unperturbed_generator, perturbZ, perturbsX, perturbsZ, 
                perturbmsZ, perturbs1, perturb3)

    from qiskit_dynamics import solve_lmde
    from qiskit_dynamics.perturbation import solve_lmde_perturbation

    def solve(model_params, control_params):
        generator, perturb0, perturb1, _, _, _, _ = construct_generator_parts(control_params)
        
        full_generator = lambda t: generator(t) + model_params[0] * perturb0(t) + model_params[1] * perturb1(t)
        
        results = solve_lmde(
            generator=full_generator,
            t_span=[0, T_gate],
            y0=np.eye(dim, dtype=complex),
            method='jax_odeint',
            atol=1e-10, rtol=1e-10
        )
        return results.y[-1]

    def solve_w_perturbation(params):
        (generator, perturbZ, perturbsX, perturbsZ, 
                perturbmsZ, perturbs1, perturb3) = construct_generator_parts(params)
        
        results = solve_lmde_perturbation(
            perturbations=[
                perturbZ, perturbsX, perturbsZ, 
                perturbmsZ, perturbs1, perturb3
            ],
            t_span=[0, T_gate],
            expansion_method='dyson',
            expansion_order=1,
            generator=generator,
            integration_method='jax_odeint',
            atol=float(config["tol"]), rtol=float(config["tol"])
        )

        return (
            results.y[-1], 
            results.perturbation_data.get_item([0])[-1], 
            results.perturbation_data.get_item([1])[-1],
            results.perturbation_data.get_item([2])[-1],
            results.perturbation_data.get_item([3])[-1],
            results.perturbation_data.get_item([4])[-1],
            results.perturbation_data.get_item([5])[-1]
        )

    from scipy.linalg import expm

    x = np.array([[0., 1.], [1., 0.]])
    sx = expm(-1j * 0.25 * np.pi * np.array([[0., 1.], [1., 0.]]))

    def fidelity_pi(U):
        U = Array(U[0:2,0:2])
        return np.abs((U.conj() * x).sum())**2 / 4.


    def fidelity_pihalf(U):
        U=Array(U[0:2,0:2])
        return np.abs((U.conj() * sx).sum())**2 / 4.


    def traceless_hs_norm(A):
        A = Array(A[:, 0:2])
        return np.linalg.norm(A - A.trace() * np.eye(dim, 2, dtype=complex) / 2.)

    def pop_leakage(X):
        return np.real(X[0, 0] + X[1, 1])
    
    def target_as_array(params):
        (U, dZ, dsX, dsZ, dmsZ, ds1, dleak) = solve_w_perturbation(params)
        fid = fidelity_pihalf(U).data                     ##change here to optimize pi/2
        oZ = traceless_hs_norm(dZ).data / (2 * np.pi * v * T_gate)
        osX = traceless_hs_norm(dsX).data / (2 * np.pi * r * T_gate)
        osZ = traceless_hs_norm(dsZ).data / (2 * np.pi * r * T_gate)
        omsZ = traceless_hs_norm(dmsZ).data / (2 * np.pi * r * T_gate)
        os1 = np.linalg.norm(Array(ds1)).data / (2 * np.pi * r * T_gate) / dim
        # lowest value of this is 0
        ave_leakage= pop_leakage(dleak) / ((static_hamiltonian[0, 0] + static_hamiltonian[1, 1]) * T_gate) - 1.
        final_leakage = jnp.linalg.norm(U[2:, 0:2])
        
        target=jnp.array([1-fid, oZ, osX, osZ, omsZ, os1, ave_leakage, final_leakage])
        
        return target

    weights = jnp.array(config["weights"])

    def objective(params):
        target=target_as_array(params)
        return jnp.dot(weights, target)
    
    from jax import jit, value_and_grad
    import scipy as sp

    jit_grad_obj = jit(value_and_grad(lambda p: objective(p)))
    
    if x0 is None:
        x0 = np.random.uniform(-1e2,1e2,num_params + 1)

    result = sp.optimize.minimize(
        jit_grad_obj,
        x0,
        method='BFGS',
        jac=True,
        tol=10e-8,
        options={'disp': False, 'maxiter': config["max_iter"]}
    )

    result_dict = {
        "params": result.x,
        "objective_values": target_as_array(result.x),
        "config": config,
        "opt_result": result
    }

    return result_dict




if __name__ == "__main__":
    import pickle
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default=None)
    parser.add_argument("--input_file", default=None)
    parser.add_argument("--result_file", default=None)
    args = parser.parse_args()

    if args.config_file is None:
        raise Exception("No config file specified.")
    
    if args.input_file is None:
        raise Exception("No input file specified.")
    
    if args.result_file is None:
        raise Exception("No result file specified.")

    # load configuration
    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

    # get initial guess
    with open(args.input_file, "rb") as file:
        x0 = pickle.load(file)["params"]

    results = run_optimization(x0, config)

    # save results
    with open(args.result_file, "wb") as file:
        pickle.dump(results, file)




    
