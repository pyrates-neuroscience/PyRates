"""Pytest test file for ...
"""

import numpy as np
import pytest
import pickle

###########
# Utility #
###########

from core.utility import nmrse, deep_compare


def setup_module():
    print("\n")
    print("================================")
    print("| Test Suite 4 : Circuit Class |")
    print("================================")


#########
# Tests #
#########


# noinspection PyTypeChecker
# @pytest.mark.xfail
@pytest.mark.parametrize("test_case", ["alpha", "spiking", "flat"])
def test_4_1_jr_circuit_bifurcation(test_case):
    """Tests whether current implementation shows expected behavior when standard Jansen-Rit circuit ([1]_) with three
    sets of synaptic input corresponding to alpha oscillation, spiking neurons or flat output, respectively.

    See Also
    --------
    :class:`JansenRitCircuit`: Documentation of Jansen-Rit NMM parametrization
    :class:`Circuit`: Detailed documentation of NMM parameters, attributes and methods.

    References
    ----------
    .. [1] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical model
       of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.

    """

    from core.utility.construct import construct_circuit_from_file
    from core.utility import read_simulation_data_from_file

    # Construct from file and test against template
    ###############################################

    path = "resources/"
    dirname = f"test_4_1_JR_{test_case}_data"
    filename = f"test_4_1_JR_{test_case}.json"

    circuit = construct_circuit_from_file(filename, path)

    # Load target data from file
    ################################

    target_data = read_simulation_data_from_file(dirname, path)

    synaptic_inputs = target_data["synaptic_inputs"]
    # unstack DataFrame
    time_vec = synaptic_inputs.index
    columns = synaptic_inputs.columns
    synaptic_inputs = np.asarray([[synaptic_inputs[pop][syn]
                                   for pop in columns.levels[0]]
                                  for syn in columns.levels[1]])
    # resulting shape: (n_syn, n_pop, n_time_steps)

    # swap axes
    synaptic_inputs = np.swapaxes(synaptic_inputs, 0, 2)  # --> (n_time_steps, n_pop, n_syn)

    # simulations parameters
    simulation_time = 1.0  # s
    # simulation_time = time_vec[-1]

    # just checking if thus gives the correct result, assuming default 1
    assert simulation_time == 1.

    # interpolate to new time step
    # time_vec = np.arange(0, simulation_time, 0.001)
    # from scipy import interpolate
    # func = interpolate.interp1d(time_vec, synaptic_inputs, axis=0, fill_value="extrapolate")
    # new_time_vec = np.arange(0, simulation_time, circuit.step_size)
    # new_synaptic_inputs = func(new_time_vec)
    # circuit.run(synaptic_inputs=new_synaptic_inputs,
    #             simulation_time=simulation_time)

    circuit.run(synaptic_inputs=synaptic_inputs,
                simulation_time=simulation_time)

    states = circuit.get_population_states(state_variable_idx=0)  # - 0.075
    # fixme: do we need to keep the - 0.075 there?

    # Save new data, if necessary due to syntax change
    ##################################################
    # from core.utility import get_simulation_data, save_simulation_data_to_file
    #
    # run_info, states_frame = get_simulation_data(circuit)
    # save_simulation_data_to_file(output_data=states_frame, run_info=run_info,
    #                              dirname=dirname, path=path)

    # calculate nmrse between time-series
    #####################################

    # error = nmrse(states[1:, :], target_states)
    # error = np.mean(error)

    # perform unit test
    ###################

    assert deep_compare(states, target_data["output"], approx={"rtol": 1e-10, "atol": 0})


@pytest.mark.skip
def test_4_2_jr_circuit_ii():
    """
    Tests whether current implementation shows expected behavior when standard Jansen-Rit circuit is fed with step-
    function input to the excitatory interneurons plus constant input to the pyramidal cells.
    """

    from core.circuit import JansenRitCircuit

    # set parameters
    ################

    # circuit parameters
    n_populations = 3
    n_synapses = 2

    # simulations parameters
    simulation_time = 1.0  # s
    # cutoff_time = 0.0  # s
    step_size = 5.0e-4  # s

    # synaptic inputs
    start_stim = 0.3  # s
    len_stim = 0.05  # s
    mag_stim = 300.0  # 1/s

    synaptic_inputs = np.zeros((int(simulation_time / step_size), n_populations, n_synapses))
    synaptic_inputs[int(start_stim / step_size):int(start_stim / step_size + len_stim / step_size), 1, 0] = mag_stim
    synaptic_inputs[:, 0, 0] = mag_stim / 3.

    # initialize neural mass network
    ################################

    nmm = JansenRitCircuit(max_synaptic_delay=0.045)

    # run network simulation
    ########################

    nmm.run(synaptic_inputs=synaptic_inputs,
            simulation_time=simulation_time)

    states = nmm.get_population_states(state_variable_idx=0)

    # load target data
    ###################

    with open('../resources/JR_results_II.pickle', 'rb') as f:
        target_states = pickle.load(f)

    # calculate nmrse between time-series
    #####################################

    error = nmrse(states[1:, :], target_states.T)
    error = np.mean(error)

    # perform unit test
    ###################

    # test response to step-function input to EINs plus constant input to PCs
    assert pytest.approx(0, abs=0.5) == error


# noinspection PyTypeChecker
@pytest.mark.skip
def test_4_3_jr_circuit_iii():
    """
    Tests whether expected bifurcation occurs when synaptic efficiency of JR circuit is altered (given constant
    input).
    """

    from core.circuit import CircuitFromScratch

    # set parameters
    ################

    # simulations parameters
    simulation_time = 3.0  # s
    cutoff_time = 0.0  # s
    step_size = 5.0e-4  # s

    # populations
    population_labels = ['PC', 'EIN', 'IIN']
    N = len(population_labels)
    n_synapses = 2

    # synapses
    connections = np.zeros((N, N, n_synapses))

    # AMPA connections (excitatory)
    connections[:, :, 0] = [[0, 0.8 * 135, 0], [1.0 * 135, 0, 0], [0.25 * 135, 0, 0]]

    # GABA-A connections (inhibitory)
    connections[:, :, 1] = [[0, 0, 0.25 * 135], [0, 0, 0], [0, 0, 0]]

    gaba_a_dict = {'efficacy': 0.5 * 1.273 * -1e-12,  # A
                   'tau_decay': 0.02,  # s
                   'tau_rise': 0.0004,  # s
                   'conductivity_based': False}

    max_synaptic_delay = 0.05  # s

    # axon
    axon_dict = {'max_firing_rate': 5.,  # 1/s
                 'membrane_potential_threshold': -0.069,  # V
                 'sigmoid_steepness': 555.56}  # 1/V
    axon_params = [axon_dict for i in range(N)]

    init_states = np.zeros(N)

    # synaptic inputs
    mag_stim = 200.0  # 1/s
    synaptic_inputs = np.zeros((int(simulation_time / step_size), N, n_synapses))
    synaptic_inputs[:, 1, 0] = mag_stim

    # loop over different AMPA synaptic efficiencies and simulate network behavior
    ##############################################################################

    ampa_efficiencies = np.linspace(0.1, 1.0, 20) * 1.273 * 3e-13

    final_state = np.zeros(len(ampa_efficiencies))

    for i, efficiency in enumerate(ampa_efficiencies):
        # set ampa parameters
        #####################

        ampa_dict = {'efficacy': float(efficiency),  # A
                     'tau_decay': 0.006,  # s
                     'tau_rise': 0.0006,  # s
                     'conductivity_based': False}

        synapse_params = [ampa_dict, gaba_a_dict]

        # initialize neural mass network
        ################################

        nmm = CircuitFromScratch(connectivity=connections,
                                 step_size=step_size,
                                 synapse_params=synapse_params,
                                 axon_params=axon_params,
                                 max_synaptic_delay=max_synaptic_delay,
                                 init_states=init_states,
                                 population_labels=population_labels,
                                 delays=None)

        # run network simulation
        ########################

        nmm.run(synaptic_inputs=synaptic_inputs,
                simulation_time=simulation_time)

        final_state[i] = nmm.get_population_states(state_variable_idx=0)[-1, 0]

    # load target data
    ###################

    with open('../resources/JR_results_III.pickle', 'rb') as f:
        target_states = pickle.load(f)

    # calculate nmrse between time-series
    #####################################

    error = nmrse(final_state, target_states)
    error = np.mean(error)

    # perform unit test
    ###################

    # test response to varying AMPA synapse efficiencies given constant input to EINs.
    assert pytest.approx(0, abs=0.5) == error


# @pytest.mark.skip
def test_4_4_jr_network_i():
    """
    tests whether 2 delay-connected vs unconnected JR circuits behave as expected.
    """

    # set parameters
    ################

    # connections
    connection_strengths_1 = [0.]
    connection_strengths_2 = [100.]
    source_populations = [0]
    target_populations = [3]

    # delays
    delays = [0.001]

    # neural mass circuits
    from core.circuit import JansenRitCircuit
    nmm1 = JansenRitCircuit()
    nmm2 = JansenRitCircuit()
    nmm3 = JansenRitCircuit()
    nmm4 = JansenRitCircuit()

    # simulation step-size
    step_size = 5e-4

    # simulation time
    simulation_time = 1.
    timesteps = np.int(simulation_time / step_size)

    # synaptic input
    stim_time = 0.3
    stim_timesteps = np.int(stim_time / step_size)
    synaptic_input = np.zeros((timesteps, 6, 2))
    synaptic_input[0:stim_timesteps, 1, 0] = 300.

    # initialize nmm network
    ########################
    from core.circuit import CircuitFromCircuit
    circuit1 = CircuitFromCircuit(circuits=[nmm1, nmm2],
                                  connection_strengths=connection_strengths_1,
                                  source_populations=source_populations,
                                  target_populations=target_populations,
                                  delays=delays,
                                  circuit_labels=['NMM1', 'NMM2'])
    circuit2 = CircuitFromCircuit(circuits=[nmm3, nmm4],
                                  connection_strengths=connection_strengths_2,
                                  source_populations=source_populations,
                                  target_populations=target_populations,
                                  delays=delays,
                                  circuit_labels=['NMM1', 'NMM2'])

    # run network simulations
    #########################

    circuit1.run(synaptic_inputs=synaptic_input, simulation_time=simulation_time)
    circuit2.run(synaptic_inputs=synaptic_input, simulation_time=simulation_time)

    # perform unit tests
    ####################

    states1 = circuit1.get_population_states(state_variable_idx=0)
    states2 = circuit2.get_population_states(state_variable_idx=0)

    error = nmrse(states1, states2)
    error = np.mean(error)

    # test information transfer between two delay-connected JR circuits...
    assert not pytest.approx(0, abs=0.5) == error


def test_4_5_circuit_run_method():
    """Testing whether the method Circuit.run does what it is supposed to"""

    from core.circuit import JansenRitCircuit

    # set parameters
    ################

    N = 3
    n_synapses = 2

    # simulations parameters
    simulation_time = 1e-3  # s
    step_size = 5e-4  # s
    n_time_steps = int(simulation_time / step_size)
    synaptic_inputs = np.zeros((n_time_steps, N, n_synapses))

    # initialize neural mass network
    ################################

    circuit = JansenRitCircuit(step_size=step_size)

    # test if run is executed as expected
    #####################################

    with pytest.raises(ValueError):
        circuit.run(synaptic_inputs=synaptic_inputs,
                    simulation_time=-1)

    # check if dimensions of synaptic input are correct
    ###################################################

    indices = ((1, 0, 0),
               (0, 1, 0),
               (0, 0, 1))
    for i, j, k in indices:
        with pytest.raises(ValueError):
            wrong_synaptic_inputs = synaptic_inputs[i:, j:, k:]
            circuit.run(synaptic_inputs=wrong_synaptic_inputs,
                        simulation_time=simulation_time)

    # check shape of extrinsic current
    ##################################

    ext_current = np.zeros((n_time_steps, N))
    indices = ((1, 0),
               (0, 1))
    for i, j in indices:
        with pytest.raises(ValueError):
            wrong_ext_current = ext_current[i:, j:]
            circuit.run(synaptic_inputs=synaptic_inputs,
                        simulation_time=simulation_time,
                        extrinsic_current=wrong_ext_current)

    # check extrinsic modulation
    ############################

    ext_mod = np.ones((n_time_steps, N))
    indices = ((1, 0),
               (0, 1))
    for i, j in indices:
        with pytest.raises(ValueError):
            wrong_ext_mod = ext_mod[i:, j:]
            circuit.run(synaptic_inputs=synaptic_inputs,
                        simulation_time=simulation_time,
                        extrinsic_modulation=wrong_ext_mod)

    # check actual runtime
    ######################

    # TODO: think of a simple case to check for (like a test population/test circuit)


def test_circuit_run_verbosity(capsys):
    from core.circuit import JansenRitCircuit

    # set parameters
    ################

    N = 3
    n_synapses = 2

    # simulations parameters
    simulation_time = 0.1  # s
    step_size = 5e-4  # s
    n_time_steps = int(simulation_time / step_size)
    synaptic_inputs = np.zeros((n_time_steps, N, n_synapses))

    # initialize neural mass network
    ################################

    circuit = JansenRitCircuit(step_size=step_size)
    # check verbosity
    #################

    circuit.run(synaptic_inputs=synaptic_inputs,
                simulation_time=simulation_time,
                verbose=True)

    out, err = capsys.readouterr()
    exp_out = 'simulation progress: 0 %\n' \
              'simulation progress: 10 %\n' \
              'simulation progress: 20 %\n' \
              'simulation progress: 30 %\n' \
              'simulation progress: 40 %\n' \
              'simulation progress: 50 %\n' \
              'simulation progress: 60 %\n' \
              'simulation progress: 70 %\n' \
              'simulation progress: 80 %\n' \
              'simulation progress: 90 %\n'
    assert out == exp_out
