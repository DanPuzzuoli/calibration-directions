import os
os.environ["JAX_JIT_PJIT_API_MERGE"] = "0"

# configure jax to use 64 bit mode
import jax
jax.config.update("jax_enable_x64", True)

# tell JAX we are using CPU
jax.config.update('jax_platform_name', 'cpu')

# import Array and set default backend
from qiskit_dynamics.array import Array
Array.set_default_backend('jax')
from jax import jacrev
import jax.numpy as jnp
import numpy as np
from qiskit_dynamics.models import RotatingFrame
from qiskit_dynamics import solve_lmde
from get_optcal_pulse_v2 import optcal_pulse
import pickle


def SVD_direcs(folder,pulse_index,filter='new'):

    file = 'results/opt_state' + str(pulse_index) + '.pkl'
    fileObj = open(folder+file,'rb')
    contents = pickle.load(fileObj)
    fileObj.close()

    dim = contents['config']['dim']
    v = contents['config']['v']
    r = contents['config']['r']
    v_spec = contents['config']['v_spec']
    anharm = contents['config']['anharm']
    dt = 1/4.5
    n = contents['config']['n_steps']
    c0 = np.array(contents['params'])

    a = np.diag(np.sqrt(np.arange(1, dim)), 1)
    adag = np.diag(np.sqrt(np.arange(1, dim)), -1)
    N = np.diag(np.arange(dim))
    X = a + adag

    # static part
    static_hamiltonian = 2 * np.pi * v * N + np.pi * anharm * N * (N - np.eye(dim))

    rotating_frame = RotatingFrame(static_hamiltonian)

    T = dt * n
    T_gate=np.ceil(3*T-2*T/n)

    iX = -1j * 2 * np.pi * r * Array(X)
    iN = -1j * 2 * np.pi * v * Array(N)

    w_spec = 1j * 2 * np.pi * Array(v_spec)
    ident = Array(np.eye(dim, dtype=complex))

    def construct_generator_parts(params):
        
        sig = optcal_pulse(params,filter=filter,return_opt = 1)
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


    points=jnp.vstack((np.linspace(-0.00001,0.00001,4),np.linspace(-0.03,0.03,4)))
        
    def real_solve(control_params):
        U= jnp.array([solve(x, jnp.append(control_params, c0[-1])) for x in points.T])
        U = U[:,0:2,0:2]
        return jnp.array([U.real, U.imag]).flatten()

    jacobian = jacrev(real_solve)(c0[:-1])
    u, s, vh = jnp.linalg.svd(jacobian)
    cal_num = 4
    cal_direc = vh[:cal_num]

    return cal_direc