"""
"""

__author__ = "Daniel Rose, Richard Gast"
__status__ = "Development"

import numpy as np
import pytest

###########
# Utility #
###########


def setup_module():
    print("\n")
    print("===================================")
    print("| Test Suite 3 : Population Class |")
    print("===================================")


#########
# Tests #
#########


@pytest.mark.skip
def test_3_1_population_init():
    """Tests whether synapses and axon of initialized population show expected behavior.

    See Also
    --------
    :class:`Population`: Detailed documentation of population parameters, attributes and methods.

    """

    from pyrates.population import Population
    from pyrates.synapse import AMPACurrentSynapse, GABAACurrentSynapse
    from pyrates.axon import KnoescheAxon

    # population parameters
    #######################

    synapse_types = ['AMPACurrentSynapse', 'GABAACurrentSynapse']
    axon = 'KnoescheAxon'
    init_state = -0.075
    step_size = 5.e-4
    synaptic_kernel_length = 0.05
    tau_leak = 0.016
    resting_potential = -0.075

    # initialize population, synapses and axon
    pop = Population(synapses=synapse_types,
                     axon=axon,
                     init_state=init_state,
                     step_size=step_size,
                     max_synaptic_delay=synaptic_kernel_length,
                     tau_leak=tau_leak,
                     resting_potential=resting_potential,
                     )
    syn1 = AMPACurrentSynapse(bin_size=step_size)
    syn2 = GABAACurrentSynapse(bin_size=step_size)
    axon = KnoescheAxon()

    # define firing rate input and membrane potential
    #################################################

    time_steps = int(0.05 / step_size)
    firing_rate = 300.0
    membrane_potential = -0.06

    # calculate population, synapse and axon response
    #################################################

    # pass input
    pop.synapses['AMPA_current'].pass_input(firing_rate)
    pop.synapses['GABAA_current'].pass_input(firing_rate)
    syn1.pass_input(firing_rate)
    syn2.pass_input(firing_rate)

    # calculate synaptic response
    pop_syn1_response = pop.synapses['AMPA_current'].get_synaptic_response()
    pop_syn2_response = pop.synapses['GABAA_current'].get_synaptic_response()
    syn1_response = syn1.get_synaptic_response()
    syn2_response = syn2.get_synaptic_response()

    pop_ax_response = pop.axon.update_firing_rate(membrane_potential)
    ax_response = axon.update_firing_rate(membrane_potential)

    # perform unit tests
    ####################

    # Test V - Population Init

    # test whether population synapses show expected response to firing rate input
    assert pop_syn1_response == syn1_response
    assert pop_syn2_response == syn2_response

    # test whether population axon shows expected response to membrane potential input
    assert pop_ax_response == ax_response


@pytest.mark.skip
def test_3_2_population_dynamics():
    """Tests whether population develops as expected over time given some input.

    See Also
    --------
    :class:`Population`: Detailed documentation of population parameters, attributes and methods.

    """

    from pyrates.population import Population

    # set population parameters
    ###########################

    synapse_types = ['AMPACurrentSynapse', 'GABAACurrentSynapse']
    axon = 'KnoescheAxon'
    step_size = 5e-4  # unit = s
    synaptic_kernel_length = 0.05  # unit = s
    tau_leak = 0.016  # unit = s
    resting_potential = -0.075  # unit = V
    membrane_capacitance = 1e-12  # unit = q/V
    init_state = 0.  # unit = (A)

    # define population input
    #########################

    time_steps = int(0.05 / step_size)
    synaptic_inputs = np.zeros((5 * time_steps, 2, 4))
    synaptic_inputs[:, 0, 1] = 300.0
    synaptic_inputs[:, 1, 2] = 300.0
    synaptic_inputs[0:time_steps, 0, 3] = 300.0

    extrinsic_inputs = np.zeros((5 * time_steps, 2), dtype=float)
    extrinsic_inputs[0:time_steps, 1] = 1e-14

    # for each combination of inputs calculate state vector of population instance
    ##############################################################################

    states = np.zeros((synaptic_inputs.shape[2], extrinsic_inputs.shape[1], extrinsic_inputs.shape[0]))
    for i in range(synaptic_inputs.shape[2]):
        for j in range(extrinsic_inputs.shape[1]):
            pop = Population(synapses=synapse_types,
                             axon=axon,
                             init_state=init_state,
                             step_size=step_size,
                             max_synaptic_delay=synaptic_kernel_length,
                             tau_leak=tau_leak,
                             resting_potential=resting_potential,
                             membrane_capacitance=membrane_capacitance)
            for k in range(synaptic_inputs.shape[0]):
                pop.synapses['AMPA_current'].pass_input(synaptic_input=synaptic_inputs[k, 0, i].squeeze())
                pop.synapses['GABAA_current'].pass_input(synaptic_input=synaptic_inputs[k, 1, i].squeeze())
                pop.extrinsic_current = extrinsic_inputs[k, j]
                pop.state_update()
                states[i, j, k] = pop.membrane_potential

    # perform unit tests
    ####################

    # Population Dynamics
    # test whether resulting membrane potential for zero input is equal to resting potential
    assert states[0, 0, -1] == pytest.approx(resting_potential, rel=1e-2)

    # test whether constant excitatory synaptic input leads to increased membrane potential
    assert states[1, 0, -1] > resting_potential

    # test whether constant inhibitory input leads to decreased membrane potential
    assert states[2, 0, -1] < resting_potential

    # test whether extrinsic current leads to expected change in membrane potential
    assert states[0, 1, 0] == pytest.approx(resting_potential + step_size * (1e-14 / membrane_capacitance), rel=1e-4)

    # test whether membrane potential goes back to resting potential after step-function input
    assert states[3, 0, -1], pytest.approx(resting_potential, rel=1e-4)
    assert states[0, 1, -1], pytest.approx(resting_potential, rel=1e-4)
