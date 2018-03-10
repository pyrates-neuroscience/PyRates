"""Includes various versions of population methods for different population types
"""

from typing import Union, Optional
import numpy as np


########################
# state update methods #
########################

def state_update(self,
                 extrinsic_current: Union[float, np.float64] = 0.,
                 extrinsic_synaptic_modulation: Optional[np.ndarray] = None
                 ) -> None:
    """Updates state of self by making a single step forward in time.

    Parameters
    ----------
    self
        self instance (see :class:`self`).
    extrinsic_current
        Extrinsic current arriving at time-point `t`, affecting the membrane potential of the self.
        (default = 0.) [unit = A].
    extrinsic_synaptic_modulation
        Modulatory (multiplicatory) input to each synapse. Vector with len = number of synapses
        (default = None) [unit = 1].

    """

    # add inputs to internal state variables
    ########################################

    # extrinsic current
    self.extrinsic_current = extrinsic_current

    # extrinsic modulation
    if extrinsic_synaptic_modulation is not None:
        self.extrinsic_synaptic_modulation[0:len(extrinsic_synaptic_modulation)] = extrinsic_synaptic_modulation

    # compute average membrane potential
    ####################################

    membrane_potential = self.state_variables[-1][0]
    membrane_potential = self.take_step(f=self.get_delta_membrane_potential,
                                              y_old=membrane_potential)

    state_vars = [membrane_potential]

    # compute average firing rate
    #############################

    self.axon.compute_firing_rate(membrane_potential)

    # update state variables
    ########################

    # TODO: Implement observer system here!!!
    self.state_variables.append(state_vars)
    if not self.store_state_variables:
        self.state_variables.pop(0)


def state_update_no_modulation(self,
                               extrinsic_current: Union[float, np.float64] = 0.
                               ) -> None:
    """Updates state of self by making a single step forward in time.

    Parameters
    ----------
    self
        self instance (see :class:`self`).
    extrinsic_current
        Extrinsic current arriving at time-point `t`, affecting the membrane potential of the self.
        (default = 0.) [unit = A].

    """

    # add inputs to internal state variables
    ########################################

    # extrinsic current
    self.extrinsic_current = extrinsic_current

    # compute average membrane potential
    ####################################

    membrane_potential = self.state_variables[-1][0]
    membrane_potential = self.take_step(f=self.get_delta_membrane_potential,
                                              y_old=membrane_potential)

    state_vars = [membrane_potential]

    # compute average firing rate
    #############################

    self.axon.compute_firing_rate(membrane_potential)

    # update state variables
    ########################

    # TODO: Implement observer system here!!!
    self.state_variables.append(state_vars)
    if not self.store_state_variables:
        self.state_variables.pop(0)


def state_update_plastic(self,
                         extrinsic_current: Union[float, np.float64] = 0.,
                         extrinsic_synaptic_modulation: Optional[np.ndarray] = None,
                         ) -> None:
    """Updates state of self by making a single step forward in time.

    Parameters
    ----------
    self
        self instance (see :class:`self`).
    extrinsic_current
        Extrinsic current arriving at time-point `t`, affecting the membrane potential of the self.
        (default = 0.) [unit = A].
    extrinsic_synaptic_modulation
        Modulatory input to each synapse. Vector with len = number of synapses (default = 1.0) [unit = 1].

    """

    # call super state update
    #########################

    super().state_update(extrinsic_current=extrinsic_current,
                         extrinsic_synaptic_modulation=extrinsic_synaptic_modulation)

    # update axonal transfer function
    #################################

    if self.spike_frequency_adaptation:

        self.axon_update()
        self.state_variables[-1] += [self.axon.transfer_function_args['adaptation']]

    # update synaptic scaling
    #########################

    for i in range(self.n_synapses):

        if self.synapse_efficacy_adaptation[i]:

            self.synapse_update(i)
            self.state_variables[-1] += [self.synapses[i].depression]


def state_update_plastic_no_modulation(self,
                                       extrinsic_current: Union[float, np.float64] = 0.
                                       ) -> None:
    """Updates state of self by making a single step forward in time.

    Parameters
    ----------
    self
        self instance (see :class:`self`).
    extrinsic_current
        Extrinsic current arriving at time-point `t`, affecting the membrane potential of the self.
        (default = 0.) [unit = A].

    """

    # call super state update
    #########################

    super().state_update(extrinsic_current=extrinsic_current)

    # update axonal transfer function
    #################################

    if self.spike_frequency_adaptation:

        self.axon_update()
        self.state_variables[-1] += [self.axon.transfer_function_args['adaptation']]

    # update synaptic scaling
    #########################

    for i in range(self.n_synapses):

        if self.synapse_efficacy_adaptation[i]:

            self.synapse_update(i)
            self.state_variables[-1] += [self.synapses[i].depression]


def state_update_differential(self,
                              extrinsic_current: Union[float, np.float64] = 0.,
                              extrinsic_synaptic_modulation: Optional[np.ndarray] = None
                              ) -> None:
    """Updates state of self by making a single step forward in time.

    Parameters
    ----------
    self
        self instance ( see :class:`self`)
    extrinsic_current
        Extrinsic current arriving at time-point `t`, affecting the membrane potential of the self.
        (default = 0.) [unit = A].
    extrinsic_synaptic_modulation
        Modulatory (multiplicatory) input to each synapse. Vector with len = number of synapses
        (default = None) [unit = 1].

    """

    # add inputs to internal state variables
    ########################################

    # extrinsic current
    self.extrinsic_current = extrinsic_current

    # extrinsic modulation
    if extrinsic_synaptic_modulation is not None:
        self.extrinsic_synaptic_modulation[0:len(extrinsic_synaptic_modulation)] = extrinsic_synaptic_modulation

    # compute average membrane potential
    ####################################

    for i, psp in enumerate(self.PSPs):
        self.PSPs[i] = self.take_step(f=self.get_delta_psp, y_old=self.PSPs[i], synapse_idx=i)
    membrane_potential = np.sum(self.PSPs).squeeze() + self.extrinsic_current
    state_vars = [membrane_potential]

    # compute average firing rate
    #############################

    self.axon.compute_firing_rate(membrane_potential)

    # update state variables
    ########################

    # TODO: Implement observer system here!!!
    self.state_variables.append(state_vars)
    if not self.store_state_variables:
        self.state_variables.pop(0)


def state_update_differential_lc(self,
                                 extrinsic_current: Union[float, np.float64] = 0.,
                                 extrinsic_synaptic_modulation: Optional[np.ndarray] = None
                                 ) -> None:
    """Updates state of self by making a single step forward in time.

    Parameters
    ----------
    self
        self instance ( see :class:`self`)
    extrinsic_current
        Extrinsic current arriving at time-point `t`, affecting the membrane potential of the self.
        (default = 0.) [unit = A].
    extrinsic_synaptic_modulation
        Modulatory (multiplicatory) input to each synapse. Vector with len = number of synapses
        (default = None) [unit = 1].

    """

    # add inputs to internal state variables
    ########################################

    # extrinsic current
    self.extrinsic_current = extrinsic_current

    # extrinsic modulation
    if extrinsic_synaptic_modulation is not None:
        self.extrinsic_synaptic_modulation[0:len(extrinsic_synaptic_modulation)] = extrinsic_synaptic_modulation

    # compute average membrane potential
    ####################################

    for i, psp in enumerate(self.PSPs):
        self.PSPs[i] = self.take_step(f=self.get_delta_psp, y_old=self.PSPs[i], synapse_idx=i)
    leak_current = self.take_step(f=self.get_leak_current, y_old=self.leak_current)
    membrane_potential = np.sum(self.PSPs).squeeze() + leak_current + self.extrinsic_current
    state_vars = [membrane_potential]

    # compute average firing rate
    #############################

    self.axon.compute_firing_rate(membrane_potential)

    # update state variables
    ########################

    # TODO: Implement observer system here!!!
    self.state_variables.append(state_vars)
    if not self.store_state_variables:
        self.state_variables.pop(0)


def state_update_differential_no_modulation(self,
                                            extrinsic_current: Union[float, np.float64] = 0.,
                                            extrinsic_synaptic_modulation: Optional[np.ndarray] = None
                                            ) -> None:
    """Updates state of self by making a single step forward in time.

    Parameters
    ----------
    self
        self instance ( see :class:`self`)
    extrinsic_current
        Extrinsic current arriving at time-point `t`, affecting the membrane potential of the self.
        (default = 0.) [unit = A].
    extrinsic_synaptic_modulation
        Modulatory (multiplicatory) input to each synapse. Vector with len = number of synapses
        (default = None) [unit = 1].

    """

    # add inputs to internal state variables
    ########################################

    # extrinsic current
    self.extrinsic_current = extrinsic_current

    # compute average membrane potential
    ####################################

    for i, psp in enumerate(self.PSPs):
        self.PSPs[i] = self.take_step(f=self.get_delta_psp, y_old=self.PSPs[i], synapse_idx=i)
    membrane_potential = np.sum(self.PSPs).squeeze() + self.extrinsic_current
    state_vars = [membrane_potential]

    # compute average firing rate
    #############################

    self.axon.compute_firing_rate(membrane_potential)

    # update state variables
    ########################

    # TODO: Implement observer system here!!!
    self.state_variables.append(state_vars)
    if not self.store_state_variables:
        self.state_variables.pop(0)


def state_update_differential_lc_no_modulation(self,
                                               extrinsic_current: Union[float, np.float64] = 0.,
                                               extrinsic_synaptic_modulation: Optional[np.ndarray] = None
                                               ) -> None:
    """Updates state of self by making a single step forward in time.

    Parameters
    ----------
    self
        self instance ( see :class:`self`)
    extrinsic_current
        Extrinsic current arriving at time-point `t`, affecting the membrane potential of the self.
        (default = 0.) [unit = A].

    """

    # add inputs to internal state variables
    ########################################

    # extrinsic current
    self.extrinsic_current = extrinsic_current

    # compute average membrane potential
    ####################################

    for i, psp in enumerate(self.PSPs):
        self.PSPs[i] = self.take_step(f=self.get_delta_psp, y_old=self.PSPs[i], synapse_idx=i)
    leak_current = self.take_step(f=self.get_leak_current, y_old=self.leak_current)
    membrane_potential = np.sum(self.PSPs).squeeze() + leak_current + self.extrinsic_current
    state_vars = [membrane_potential]

    # compute average firing rate
    #############################

    self.axon.compute_firing_rate(membrane_potential)

    # update state variables
    ########################

    # TODO: Implement observer system here!!!
    self.state_variables.append(state_vars)
    if not self.store_state_variables:
        self.state_variables.pop(0)


def get_delta_membrane_potential_lc(self,
                                    membrane_potential: Union[float, np.float64]
                                    ) -> Union[float, np.float64]:
    """Calculates change in membrane potential as function of synaptic current, leak current and
    extrinsic current.

    Parameters
    ----------
    self
        self instance (see :class:`self`).
    membrane_potential
        Current membrane potential of self [unit = V].

    Returns
    -------
    float
        Delta membrane potential [unit = V].

    """

    net_current = self.get_synaptic_currents(membrane_potential) \
                  + self.get_leak_current(membrane_potential) \
                  + self.extrinsic_current

    return net_current / self.membrane_capacitance


def get_delta_membrane_potential(self,
                                 membrane_potential: Union[float, np.float64]
                                 ) -> Union[float, np.float64]:
    """Calculates change in membrane potential as function of synaptic current, leak current and
    extrinsic current.

    Parameters
    ----------
    self
        self instance (see :class:`self`).
    membrane_potential
        Current membrane potential of self [unit = V].

    Returns
    -------
    float
        Delta membrane potential [unit = V].

    """

    return self.get_synaptic_currents(membrane_potential) + self.extrinsic_current


def get_synaptic_currents(self,
                          membrane_potential: Union[float, np.float64]
                          ) -> Union[Union[float, np.float64], np.ndarray]:
    """Calculates the net synaptic current over all synapses.

    Parameters
    ----------
    self
        self instance (see :class:`self`).
    membrane_potential
        Current membrane potential of self [unit = V].

    Returns
    -------
    float
        Net synaptic current [unit = A].

    """

    # compute synaptic currents and modulations
    ###########################################

    # calculate synaptic currents for each additive synapse
    for i, syn in enumerate(self.synapses):
        self.synaptic_currents[i] = syn.get_synaptic_current(membrane_potential)

    return self.synaptic_currents @ self.extrinsic_synaptic_modulation.T


def get_synaptic_currents_no_modulation(self,
                                        membrane_potential: Union[float, np.float64]
                                        ) -> Union[Union[float, np.float64], np.ndarray]:
    """Calculates the net synaptic current over all synapses.

    Parameters
    ----------
    self
        self instance (see :class:`self`).
    membrane_potential
        Current membrane potential of self [unit = V].

    Returns
    -------
    float
        Net synaptic current [unit = A].

    """

    # compute synaptic currents and modulations
    ###########################################

    # calculate synaptic currents for each additive synapse
    for i, syn in enumerate(self.synapses):
        self.synaptic_currents[i] = syn.get_synaptic_current(membrane_potential)

    return np.sum(self.synaptic_currents)


def get_delta_psp(self,
                  psp: Union[float, np.float64],
                  synapse_idx: int
                  ) -> Union[float, np.float64]:
    """Calculates change in membrane potential as function of synaptic current, leak current and
    extrinsic current.

    Parameters
    ----------
    self
        self instance (see :class:`self`).
    psp
        Current membrane potential of self [unit = V].
    synapse_idx
        Index of synapse for which to calculate the PSP.
    Returns
    -------
    float
        Delta membrane potential [unit = V].

    """

    # update synaptic currents
    ###########################

    # update synaptic currents old
    self.synaptic_currents_old[synapse_idx] = self.synaptic_currents[synapse_idx]

    # calculate synaptic current
    self.synaptic_currents[synapse_idx] = self.take_step(f=self.synapses
                                                                     [synapse_idx].get_delta_synaptic_current,
                                                                     y_old=self.synaptic_currents_old
                                                                     [synapse_idx],
                                                                     membrane_potential=psp)

    return self.synaptic_currents_old[synapse_idx] * self.extrinsic_synaptic_modulation[synapse_idx]


def get_delta_psp_no_modulation(self,
                                psp: Union[float, np.float64],
                                synapse_idx: int
                                ) -> Union[float, np.float64]:
    """Calculates change in membrane potential as function of synaptic current, leak current and
    extrinsic current.

    Parameters
    ----------
    self
        self instance (see :class:`self`).
    psp
        Current membrane potential of self [unit = V].
    synapse_idx
        Index of synapse for which to calculate the PSP.
    Returns
    -------
    float
        Delta membrane potential [unit = V].

    """

    # update synaptic currents
    ###########################

    # update synaptic currents old
    self.synaptic_currents_old[synapse_idx] = self.synaptic_currents[synapse_idx]

    # calculate synaptic current
    self.synaptic_currents[synapse_idx] = self.take_step(f=self.synapses
                                                                     [synapse_idx].get_delta_synaptic_current,
                                                                     y_old=self.synaptic_currents_old
                                                                     [synapse_idx],
                                                                     membrane_potential=psp)

    return self.synaptic_currents_old[synapse_idx]
