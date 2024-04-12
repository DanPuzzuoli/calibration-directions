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
Pulse optimization workflow functions for optimizations using perturbative robustness.
"""

from qiskit_dynamics import Signal
from qiskit_dynamics.models import RotatingFrame
from qiskit_dynamics.perturbation import solve_lmde_perturbation

from multiset import Multiset


def perturbative_objective_function(
    opt_parameters,
    signal_parameterizations,
    model,
    rotating_frame,
    perturbations,
    t_span,
    objective_functions,
    objective_weights,
    expansion_method=None,
    expansion_order=None,
    expansion_labels=None,
    y0=None,
    dyson_in_frame=True,
    integration_method="jax_odeint",
    **kwargs
):
    """Compute an objective involving perturbative terms.
    
    Args:
        opt_parameters: Dictionary of named parameters to be optimized (arrays).
        signal_parmeterizations: Dictionary with format {signal_name: parameterization_func}. The
            parameterization function is assumed take a dictionary of parameters and return a Signal
            object. Note that the name "drift" is reserved.
        model: Dictionary with format {signal_name: operator}. The key "drift" is specially
            reserved for the Hamiltonian component with no signal coefficient.
        rotating_frame: Frame to perform the simulation in.
        perturbations: A dictionary with format {perturbation_label: model_dict}, where each
            model_dict has the same format as the model in the previous argument.
        objective_functions: A dictionary of the form {objective_name: Callable} giving a labelled
            collection of objective functions. The Callable dictionary values are assumed to be
            functions of the results object returned by ``solve_lmde_perturbation`` (which can
            now take string labels for perturbations).
        objective_weights: A dictionary of the form {objective_name: float} giving a labelled
            collection of weights for the objective functions. Note that the keys of
            objective_weights must exactly match those of objective_functions.
        
        The remaining arguments are the same as with solve_lmde_perturbation
        
        t_span: 
        expansion_method: Note this only works for "dyson" or "magnus"
        expansion_order:
        expansion_labels:
        y0:
        dyson_in_frame:
        integration_method: Numerical integration method.
        kwargs: kwargs to pass to the solver (e.g. tolerances).
    Returns:
        OdeResult: Standard solve_lmde_perturbation results object.
    """

    sim_results = optimization_perturbation_sim(
        opt_parameters=opt_parameters,
        signal_parameterizations=signal_parameterizations,
        model=model,
        rotating_frame=rotating_frame,
        perturbations=perturbations,
        t_span=t_span,
        expansion_method=expansion_method,
        expansion_order=expansion_order,
        expansion_labels=expansion_labels,
        y0=y0,
        dyson_in_frame=dyson_in_frame,
        integration_method=integration_method,
        **kwargs
    )

    objective_values = {name: func(sim_results) for name, func in objective_functions.items()}

    objective_value = 0.
    for name, obj_val in objective_values.items():
        objective_value = objective_value + objective_weights[name] * obj_val

    return objective_value


def optimization_perturbation_sim(
    opt_parameters,
    signal_parameterizations,
    model,
    rotating_frame,
    perturbations,
    t_span,
    expansion_method=None,
    expansion_order=None,
    expansion_labels=None,
    y0=None,
    dyson_in_frame=True,
    integration_method="jax_odeint",
    **kwargs
):
    """This is a higher level interface for working with ``solve_lmde_perturbation``.

    Args:
        opt_parameters: Dictionary of named parameters (arrays).
        signal_parmeterizations: Dictionary with format {signal_name: parameterization_func}. The
            parameterization function is assumed take a dictionary of parameters and return a Signal
            object. Note that the name "drift" is reserved.
        model: Dictionary with format {signal_name: operator}. The key "drift" is specially
            reserved for the Hamiltonian component with no signal coefficient.
        rotating_frame: Frame to perform the simulation in.
        perturbations: A dictionary with format {perturbation_label: model_dict}, where each
            model_dict has the same format as the model in the previous argument.
        
        The remaining arguments are the same as with solve_lmde_perturbation
        
        t_span: 
        expansion_method: Note this only works for "dyson" or "magnus"
        expansion_order:
        expansion_labels:
        y0:
        dyson_in_frame:
        integration_method: Numerical integration method.
        kwargs: kwargs to pass to the solver (e.g. tolerances).
    Returns:
        OdeResult: Standard solve_lmde_perturbation results object.
    """

    # build signals
    signals = {name: func(opt_parameters) for name, func in signal_parameterizations.items()}
    signals["drift"] = Signal(1.)

    # build model and perturbation functions

    rotating_frame = RotatingFrame(rotating_frame)

    generator = _get_model_function(model, signals, rotating_frame, generator=True)

    perturbation_labels = []
    perturbation_functions = []
    for label, p_dict in perturbations.items():
        perturbation_labels.append(label)
        perturbation_functions.append(_get_model_function(p_dict, signals, rotating_frame))
    
    base_label_int_map = _generic_labels_to_int_map(perturbation_labels)
    int_labels = _convert_multiset_labels_to_int(perturbation_labels, base_label_int_map)

    results = solve_lmde_perturbation(
        perturbations=perturbation_functions,
        t_span=t_span,
        expansion_method=expansion_method,
        expansion_order=expansion_order,
        expansion_labels=expansion_labels,
        perturbation_labels=int_labels,
        generator=generator,
        y0=y0,
        dyson_in_frame=dyson_in_frame,
        integration_method=integration_method,
        **kwargs
    )

    reverse_base_label_int_map = {value: key for key, value in base_label_int_map.items()}
    results.perturbation_data.labels = _convert_multiset_labels_to_int(results.perturbation_data.labels, reverse_base_label_int_map)

    return results


def _get_model_function(model_dict, signal_dict, rotating_frame, generator=False):

    if generator:
        frame_func = rotating_frame.generator_into_frame
    else:
        frame_func = rotating_frame.operator_into_frame

    def func(t):
        op = -1j * sum(signal_dict[sig_name](t) * op for sig_name, op in model_dict.items())
        return frame_func(t, op)
    
    return func


def _generic_labels_to_int_map(labels):
    """Labels are assumed to denote multisets. Returns a dictionary mapping the original
    base labels to integers, as well as the integer version of the original labels.
    """
    multiset_labels = [Multiset(x) for x in labels]

    # convert labels into numbers
    unique_base_labels = set()
    for label in multiset_labels:
        unique_base_labels = unique_base_labels.union(set(label))
    
    base_label_int_map = {base_label: idx for idx, base_label in enumerate(unique_base_labels)}

    return base_label_int_map


def _convert_multiset_labels_to_int(labels, base_label_int_map):
    multiset_labels = [Multiset(x) for x in labels]
    new_labels = []

    for label in multiset_labels:
        new_labels.append(Multiset({base_label_int_map[base_label]: count for base_label, count in label.items()}))
    
    return new_labels