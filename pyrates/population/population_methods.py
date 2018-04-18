"""Includes functions used to dynamically build population methods during initialization of a population.
"""

# external packages
from typing import Callable, Union
import numpy as np
from scipy.integrate import odeint

# meta infos
__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


######################################
# function_from_snippet constructors #
######################################


def construct_state_update_function(spike_frequency_adaptation: bool = False,
                                    synapse_efficacy_adaptation: bool = False,
                                    leaky_capacitor: bool = False
                                    ) -> str:
    """Builds a character string representation of the state-update method of a population.
    The functional form depends on 3 population features.

    Parameters
    ----------
    spike_frequency_adaptation
        If true, the axonal plasticity will be incorporated into the state update.
    synapse_efficacy_adaptation
        If true, the synaptic plasticity will be incorporated into the state update.
    leaky_capacitor
        If true, the sate update of the population's membrane potential will be described by the leaky-capacitor
        formalism. If false, it will be a direct function of the synaptic convolutions.

    Returns
    -------
    str
        String representation of the state_update method.

    Notes
    -----
    To turn the returned string into a method of a population instance, do the following:
        1.) use `exec(str)` to make the string executable
        2.) use `MethodType(locals()['state_update'], self)` to bind the executable to the instance.

    """

    # define the argument-dependent string snippets
    ###############################################

    # leaky capacitor snippet
    if leaky_capacitor:
        membrane_potential_update_snippet = """self.membrane_potential = self.membrane_potential_solver.\
        solve(self.membrane_potential, self.step_size)"""
    else:
        membrane_potential_update_snippet = """self.membrane_potential = self.get_delta_membrane_potential\
        (membrane_potential=self.membrane_potential)"""

    # axonal plasticity snippet
    if spike_frequency_adaptation:
        spike_frequency_adaption_snippet = """self.axonal_adaptation = self.axonal_adaptation_solver.solve(\
        self.axonal_adaptation, self.step_size)
        self.axon.transfer_function_args['adaptation'] = self.axonal_adaptation"""
    else:
        spike_frequency_adaption_snippet = ""

    # synaptic plasticity snippet
    if synapse_efficacy_adaptation:
        synaptic_efficacy_adaptation_snippet = """self.synaptic_depression = self.synaptic_depression_solver.solve(\
        self.synaptic_depression, self.step_size)
        for i, syn in enumerate(self.synapse.values()):
            syn.depression = self.synaptic_depression[i]"""
    else:
        synaptic_efficacy_adaptation_snippet = ""

    # build state_update method (as a string)
    #########################################

    func_string = f"""
def state_update(self) -> None: 

    # compute average membrane potential
    ####################################
    
    {membrane_potential_update_snippet}

    # compute average firing rate
    #############################

    self.firing_rate = self.get_firing_rate()

    # plasticity mechanisms
    #######################

    {spike_frequency_adaption_snippet}

    {synaptic_efficacy_adaptation_snippet}
    
    # update synaptic input buffer
    ##############################
    
    for _, syn in self.synapses.items():
        syn.rotate_input()
"""

    return func_string


def construct_get_delta_membrane_potential_function(leaky_capacitor: bool = False,
                                                    integro_differential: bool = False,
                                                    enable_modulation: bool = False
                                                    ) -> str:
    """Builds a character string representation of the get_delta_membrane_potential method of a population.
    The functional form depends on 3 population features.

    Parameters
    ----------
    leaky_capacitor
        If true, it will be assumed that the sate update of the population's membrane potential is described by the
        leaky-capacitor formalism. If false, it will be assumed that it's a direct function of the synaptic
        convolutions.
    integro_differential
        If true, the synaptic convolutions will be performed numerically. Else, they will be solved analytically using
        a second order differential equation system.
    enable_modulation
        If true, extrinsic modulation of the synaptic efficacies will be enabled.

    Returns
    -------
    str
        String-based representation of the get_delta_membrane_potential method of a population

    Notes
    -----
    To turn the returned string into a method of a population instance, do the following:
        1.) use `exec(str)` to make the string executable
        2.) use `MethodType(locals()['state_update'], self)` to bind the executable to the instance.

    """

    # define argument-based string snippets
    #######################################

    # define how to get synaptic current from each synapse
    if integro_differential:
        input_arg_snippet = """
            t: float,"""
        synapse_update_snippet = ""
        if leaky_capacitor:
            synaptic_current_snippet = """self.synaptic_currents = self.get_synaptic_responses(self.membrane_potential)
            """
        else:
            synaptic_current_snippet = """self.PSPs = self.get_synaptic_responses(self.membrane_potential)"""
    else:
        input_arg_snippet = ""
        synapse_update_snippet = """self.synaptic_currents[:], \
        self.PSPs[:] = self.synaptic_currents_new[:], \
        self.PSPs_new[:]"""
        synaptic_current_snippet = """self.PSPs_new = self.psp_solver.solve(self.PSPs, self.step_size)
    self.synaptic_currents_new = self.synaptic_current_solver.solve(self.synaptic_currents, self.step_size)"""

    # define how to combine synaptic currents into single value
    if enable_modulation:
        if leaky_capacitor:
            if integro_differential:
                synaptic_modulation_snippet = "syn_current = self.synaptic_currents.dot" \
                                              "(self.extrinsic_synaptic_modulation)"
            else:
                synaptic_modulation_snippet = "syn_current = self.PSPs.dot"\
                    "(self.extrinsic_synaptic_modulation)"
        else:
            synaptic_modulation_snippet = "syn_current = self.PSPs.dot" \
                "(self.extrinsic_synaptic_modulation)"
    else:
        if leaky_capacitor:
            if integro_differential:
                synaptic_modulation_snippet = "syn_current = self.synaptic_currents.sum()"
            else:
                synaptic_modulation_snippet = "syn_current = self.PSPs.sum()"
        else:
            synaptic_modulation_snippet = "syn_current = self.PSPs.sum()"

    # define how to calculate leak current
    if leaky_capacitor:
        leak_current_snippet = """leak_current = (self.resting_potential - membrane_potential) * \
                   self.membrane_capacitance / self.tau_leak"""
        net_current_snippet = "(syn_current + leak_current + self.extrinsic_current) " \
                              "/ self.membrane_capacitance"
    else:
        leak_current_snippet = ""
        net_current_snippet = "syn_current + self.extrinsic_current"

    # build function from snippets
    ##############################

    func_string = f"""def get_delta_membrane_potential(self, {input_arg_snippet}
                                 membrane_potential: Union[float, np.float64]
                                 ) -> Union[float, np.float64]:
    # calculate synaptic currents for each additive synapse
    {synaptic_current_snippet}
    {synapse_update_snippet}
    
    {synaptic_modulation_snippet}

    # calculate leak current
    {leak_current_snippet}
    
    return {net_current_snippet}"""

    return func_string


def construct_get_synaptic_responses(integro_differential: bool = False) -> str:
    """Builds a character string representation of the get_synaptic_responses method of a population.
    The functional form depends on 1 population features.

    Parameters
    ----------
    integro_differential
        If true, the synaptic convolutions will be performed numerically. Else, they will be solved analytically using
        a second order differential equation system.

    Returns
    -------
    str
        String-based representation of the get_synaptic_responses method of a population

    Notes
    -----
    To turn the returned string into a method of a population instance, do the following:
        1.) use `exec(str)` to make the string executable
        2.) use `MethodType(locals()['state_update'], self)` to bind the executable to the instance.

    """

    # define functions
    ##################

    func1 = """def get_synaptic_responses(self,
                          t: float,
                          old_synaptic_responses: np.ndarray
                          ) -> np.ndarray:

    return np.array([syn.get_synaptic_response(old_synaptic_responses[i], self.PSPs[i])
                     for i, syn in enumerate(self.synapses.values())])"""

    func2 = """def get_synaptic_responses(self,
                          membrane_potential: float,
                          ) -> np.ndarray:

    return np.array([syn.get_synaptic_response(membrane_potential)
                     for syn in self.synapses.values()])"""

    return func2 if integro_differential else func1


##########################
# alternative DE solvers #
##########################


def take_step_euler(instance: object,
                    f: Callable,
                    y_old: Union[float, np.float64, np.ndarray],
                    **kwargs
                    ) -> Union[float, np.float64, np.ndarray]:
    """Takes a step of an ODE with right-hand-side f using Euler formalism.

    Parameters
    ----------
    instance
        Instance of either :class:`pyrates.population.Population` or :class:`pyrates.observer.fMRIObserver`
    f
        Function that represents right-hand-side of ODE and takes `t` plus `y_old` as an argument.
    y_old
        Old value of y that needs to be updated according to dy(t)/dt = f(t, y)
    **kwargs
        Name-value pairs to be passed to f.

    Returns
    -------
    float
        Updated value of left-hand-side (y).

    """

    return y_old + instance.step_size * f(y_old, **kwargs)


def take_step_odeint(instance: object,
                     f: Callable,
                     y_old: Union[float, np.float64, np.ndarray],
                     **kwargs
                     ) -> Union[float, np.float64, np.ndarray]:
    """Takes a step of an ODE with right-hand-side f using scipy's odeint method.

    Parameters
    ----------
    instance
        Instance of either :class:`pyrates.population.Population` or :class:`pyrates.observer.fMRIObserver`
    f
        Function that represents right-hand-side of ODE and takes `t` plus `y_old` as an argument.
    y_old
        Old value of y that needs to be updated according to dy(t)/dt = f(t, y)
    **args
        Name-value pairs to be passed to f.

    Returns
    -------
    float
        Updated value of left-hand-side (y).

    """

    return odeint(func=f, y0=y_old, t=(0., instance.step_size), args=tuple(kwargs.values()))[-1]
