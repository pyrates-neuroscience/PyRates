"""Includes functions used to dynamically build population methods during initialization of a population.
"""

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
        membrane_potential_update_snippet = """membrane_potential = self.take_step(f=self.get_delta_membrane_potential,\
                                        y_old=membrane_potential)"""
    else:
        membrane_potential_update_snippet = "membrane_potential = self.get_delta_membrane_potential(membrane_potential)"

    # axonal plasticity snippet
    if spike_frequency_adaptation:
        spike_frequency_adaption_snippet = """self.axon_update()
    self.axonal_adaptation = self.axon.transfer_function_args['adaptation']"""
    else:
        spike_frequency_adaption_snippet = ""

    # synaptic plasticity snippet
    if synapse_efficacy_adaptation:
        synaptic_efficacy_adaptation_snippet = """for idx, key in enumerate(self.synapse_keys):

        if self.synapse_efficacy_adaptation[idx]:
            self.synapse_update(key, idx)
            self.synaptic_depression[idx] = self.synapses[key].depression"""
    else:
        synaptic_efficacy_adaptation_snippet = ""

    # build state_update method (as a string)
    #########################################

    func_string = f"""
def state_update(self) -> None: 

    # compute average membrane potential
    ####################################
    
    membrane_potential = self.membrane_potential
    {membrane_potential_update_snippet}

    self.membrane_potential = membrane_potential

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
        synapse_update_snippet = ""
        if leaky_capacitor:
            synaptic_current_snippet = """self.synaptic_currents[idx] = \
            syn.get_synaptic_response(membrane_potential)"""
        else:
            synaptic_current_snippet = """self.PSPs[idx] = \
            syn.get_synaptic_response(membrane_potential)"""
    else:
        synapse_update_snippet = """self.synaptic_currents[:], \
        self.PSPs[:] = self.synaptic_currents_new[:], \
        self.PSPs_new[:]"""
        synaptic_current_snippet = """self.PSPs_new[idx] = self.take_step(f=self.get_delta_psp, 
        y_old=self.PSPs[idx], synapse_idx=idx)
        self.synaptic_currents_new[idx] = self.take_step(f=syn.get_synaptic_response,
                                                         y_old=self.synaptic_currents[idx],
                                                         membrane_potential=self.PSPs[idx])"""

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

    func_string = f"""def get_delta_membrane_potential(self,
                                 membrane_potential: Union[float, np.float64]
                                 ) -> Union[float, np.float64]:
    # calculate synaptic currents for each additive synapse
    for idx, (_, syn) in enumerate(self.synapses.items()):
        {synaptic_current_snippet}
    {synapse_update_snippet}
    
    {synaptic_modulation_snippet}

    # calculate leak current
    {leak_current_snippet}
    
    return {net_current_snippet}"""

    return func_string
