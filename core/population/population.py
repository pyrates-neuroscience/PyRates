"""
Includes basic population class and functions to instantiate synapses or axons as well as a function to update
parameters of an object instance. A 'population' is supposed to be smallest computational unit in the resulting network.
"""

from matplotlib.pyplot import *
from scipy.interpolate import interp1d
from scipy.integrate import LSODA

from core.axon import Axon
import core.axon.templates as axon_templates
from core.synapse import Synapse
import core.synapse.templates as synapse_templates

__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"

# TODO: Implement synaptic plasticity mechanism(s)


class Population(object):
    """
    Basic neural mass or population class. A population is defined via a number of synapses and an axon. Includes a
    method to update the state of the population.

    :var synapses: list of synapse names.
    :var state_variables: list of state variable vectors collected over updates with the following entries:
            [0] - membrane potential [unit = V]
    :var input_firing_rate: list of input firing rates collected over updates [unit = 1/s]
    :var output_firing_rate: list of output firing rates collected over updates [unit = 1/s]
    :var store_state_variables: indicates whether all or only most recent state variables are stored
    :var store_input_firing_rate: indicates whether all or only last few inputs are stored
    :var store_output_firing_rate: indicates whether all or only most recent output firing rate is stored
    :var axon_plasticity: indicates whether axon plasticity mechanism is enabled or not
    :var tau_leak: time delay with which the population goes back to resting state potential [unit = s]
    :var resting_potential: resting-state membrane potential of the population [unit = V]

    """

    def __init__(self, synapses, axon='JansenRit', init_state=(0., 0), step_size=0.0001, synaptic_kernel_length=100,
                 tau_leak=0.016, resting_potential=-0.075, membrane_capacitance=1e-12, synapse_params=None,
                 axon_params=None, store_state_variables=False, store_input_firing_rate=False,
                 store_output_firing_rate=False):
        """
        Initializes a single neural mass.

        :param synapses: list of character strings that indicate synapse type (see pre-implemented synapse sub-classes)
        :param axon: character string that indicates axon type (see pre-implemented axon sub-classes)
        :param init_state: vector of length 2, containing initial state of neural mass, i.e. membrane potential
               [unit = V] & firing rate [unit = 1/s] (default = (0,0)).
        :param step_size: scalar, size of the time step for which the population state will be updated according
               to euler formalism [unit = s] (default = 0.1).
        :param synaptic_kernel_length: scalar that indicates number of bins the kernel should be evaluated for
               [unit = 1] (default = 100).
        :param tau_leak: scalar, time-scale with which the membrane potential of the population goes back to resting
               potential [unit = s] (default = 0.001).
        :param resting_potential: scalar, membrane potential at which no synaptic currents flow if no input arrives at
               population [unit = V] (default = -0.07).
        :param membrane_capacitance: scalar, determines average capacitance of the population cell membranes
               [unit = q/V] (default = 1e-12).
        :param synapse_params: list of dictionaries containing parameters for custom synapse type. For parameter
               explanation see synapse class (default = None).
        :param axon_params: dictionary containing parameters for custom axon type. For parameter explanation see
               axon class (default = None).
        :param store_state_variables: If false, old state variables will be erased after each state-update
               (default = False).
        :param store_input_firing_rate: If false, old input firing rates will only be kept for as much time-steps as
               necessary by synaptic kernel length (default = False).
        :param store_output_firing_rate: If false, old output firing rates will be erased after each state-update
               (default = False).

        """

        ##########################
        # check input parameters #
        ##########################

        assert type(synapses) is list
        assert type(axon) is str or axon is None
        assert len(init_state) == 2
        assert init_state[1] >= 0 and (type(init_state[0]) is float or np.float64)
        assert step_size > 0
        assert synaptic_kernel_length > 0
        assert tau_leak > 0
        assert type(resting_potential) is float or type(resting_potential) is np.float64
        assert membrane_capacitance > 0
        assert synapse_params is None or type(synapse_params) is list
        assert axon_params is None or type(axon_params) is dict

        #############################
        # set population parameters #
        #############################

        self.synapses = list()
        self.state_variables = list()
        self.input_firing_rate = list()
        self.output_firing_rate = list()
        self.store_state_variables = store_state_variables
        self.store_input_firing_rate = store_input_firing_rate
        self.store_output_firing_rate = store_output_firing_rate
        self.tau_leak = tau_leak
        self.resting_potential = resting_potential
        self.step_size = step_size
        self.membrane_capacitance = membrane_capacitance
        self.t = 0

        # set initial state and firing rate values
        self.state_variables.append(init_state[0])
        self.output_firing_rate.append(init_state[1])

        ####################################
        # set axonal plasticity parameters #
        ####################################

        if axon_params:

            if 'tau' in axon_params:

                # if axon timescale tau is in axon parameters, set relevant parameters for axon plasticity
                self.axon_plasticity = True
                self.tau_axon = axon_params['tau']
                if 'firing_rate_target' in axon_params:
                    self.firing_rate_target = axon_params['firing_rate_target']
                else:
                    self.firing_rate_target = 2.5

            else:

                self.axon_plasticity = False

        else:

            self.axon_plasticity = False

        ######################################
        # set synaptic plasticity parameters #
        ######################################

        # TODO: implement synaptic plasticity mechanism

        ###############
        # set synapse #
        ###############

        if not synapse_params:

            for i in range(len(synapses)):
                self.synapses.append(set_synapse(synapses[i], step_size, synaptic_kernel_length))

        else:

            for i in range(len(synapse_params)):
                self.synapses.append(set_synapse(synapses[i], step_size, synaptic_kernel_length, synapse_params[i]))

        ############
        # set axon #
        ############

        self.axon = set_axon(axon, axon_params)

    def state_update(self, synaptic_input, extrinsic_current=0., extrinsic_synaptic_modulation=1.0,
                     synaptic_modulation_direction=1.0, variable_step_size=False):
        """
        Updates state according to current synapse and axon parametrization.

        :param synaptic_input: vector, average firing rate arriving at each synapse [unit = 1/s].
        :param extrinsic_current: extrinsic current added to membrane potential change [unit = A] (default = 0).
        :param extrinsic_synaptic_modulation: modulatory input to each synapse. Can be scalar (applied to all synapses
               then) or vector with len = number of synapses (default = 1.0) [unit = 1].
        :param synaptic_modulation_direction: Can be scalar or list of vectors used as a power for each modulatory
               synapse as neuromodulatory effect. Should be either 1.0 or -1.0 (default = 1.0) [unit = 1].
        :param variable_step_size: If false, DEs will be solved via Euler formalism. If false, 4/5th order Runge-Kutta
               formalism will be used (default = False).

        """

        ##########################
        # check input parameters #
        ##########################

        assert all(synaptic_input) >= 0
        assert len(synaptic_input) == len(self.synapses)
        assert type(extrinsic_current) is float or np.float64

        ##########################################
        # add inputs to internal state variables #
        ##########################################

        self.variable_step_size = variable_step_size
        self.input_firing_rate.append(synaptic_input)
        self.extrinsic_current = extrinsic_current
        self.extrinsic_synaptic_modulation = extrinsic_synaptic_modulation
        self.synaptic_modulation_direction = synaptic_modulation_direction
        membrane_potential = self.state_variables[-1]

        ######################################
        # compute average membrane potential #
        ######################################

        membrane_potential = self.take_step(f=self.get_delta_membrane_potential,
                                            y_old=membrane_potential)

        ###############################
        # compute average firing rate #
        ###############################

        firing_rate = self.axon.compute_firing_rate(membrane_potential)

        ###########################################
        # update state variables and firing rates #
        ###########################################

        self.state_variables.append(membrane_potential)
        self.output_firing_rate.append(firing_rate)

        if not self.store_output_firing_rate:
            self.output_firing_rate.pop(0)
        if not self.store_state_variables:
            self.state_variables.pop(0)
        if not self.store_input_firing_rate and (len(self.input_firing_rate) > self.synapses[0].kernel_length):
            self.input_firing_rate.pop(0)

        ###################################
        # update axonal transfer function #
        ###################################

        if self.axon_plasticity:

            self.axon.membrane_potential_threshold = self.take_step(f=self.get_delta_membrane_potential_threshold,
                                                                    y_old=self.axon.membrane_potential_threshold)

        ###########################
        # update synaptic kernels #
        ###########################

        # TODO: implement differential equation that adapts synaptic efficiencies

        ###################
        # advance in time #
        ###################

        self.t += self.step_size

    def take_step(self, f, y_old):
        """
        Function that takes a step of an ODE with right-hand-side f using Euler or 4/5th order Adams/BDF formalism.

        :param f: Right-hand-side of ODE (function) that takes t as an argument.
        :param y_old: old value of y that needs to be updated according to dy(t)/dt = f(t, y)

        :return: updated value of left-hand-side (scalar)

        """

        if self.variable_step_size:

            # initialize RK45 solver
            solver = LSODA(fun=f,
                           t0=0.,
                           y0=[y_old],
                           t_bound=float('inf'),
                           min_step=self.synapses[0].step_size,
                           max_step=0.01)

            # perform integration step and calculate output
            solver.step()
            y_new = np.squeeze(solver.y)

            # update internal step-size
            self.update_step_size(solver.step_size)

        else:

            # perform Euler update
            y_new = y_old + self.step_size * f(self.t, y_old)

        return y_new

    def get_delta_membrane_potential(self, t, membrane_potential):
        """
        Method that calculates the change in membrane potential as a function of synaptic current, leak current and
        extrinsic current.

        :param t: time for which to calculate the delta term [unit = s].
        :param membrane_potential: current membrane potential of population [unit = mV].

        :return: Delta membrane potential (scalar) [unit = mV].

        """

        net_current = self.get_synaptic_currents(t, membrane_potential) + \
                      self.get_leak_current(membrane_potential) + \
                      self.extrinsic_current

        return net_current / self.membrane_capacitance

    def get_synaptic_currents(self, t, membrane_potential):
        """
        Method that calculates the net synaptic current over all synapses for time t.

        :param t: time for which to calculate the synaptic current [unit = s].
        :param membrane_potential: current membrane potential of population [unit = mV].

        :return: net synaptic current [unit = mA].
        """

        ############################################################
        # get input firing rate history and interpolate it until t #
        ############################################################

        inputs = np.asarray(self.input_firing_rate)
        step_size_diff = self.synapses[0].kernel_length * self.step_size - \
                         self.synapses[0].kernel_length * self.synapses[0].step_size

        if inputs.shape[0] > 1 and self.variable_step_size and abs(step_size_diff) > self.step_size:

            inputs = interpolate_array(old_step_size=self.step_size,
                                       new_step_size=self.synapses[0].step_size,
                                       y=inputs,
                                       axis=0,
                                       interpolation_type='linear')

        ######################################
        # compute average membrane potential #
        ######################################

        # calculate synaptic currents and modulations for each synapse
        synaptic_currents = list()
        synaptic_modulation = list()

        for i in range(len(self.synapses)):

            # synaptic modulation value
            if self.synapses[i].neuromodulatory:

                if self.synapses[i].conductivity_based:
                    synaptic_modulation.append(self.synapses[i].get_synaptic_current(inputs[:, i], membrane_potential))
                else:
                    synaptic_modulation.append(self.synapses[i].get_synaptic_current(inputs[:, i]))

            else:

                # synaptic current
                if self.synapses[i].conductivity_based:
                    synaptic_currents.append(self.synapses[i].get_synaptic_current(inputs[:, i], membrane_potential))
                else:
                    synaptic_currents.append(self.synapses[i].get_synaptic_current(inputs[:, i]))

        synaptic_currents = np.array(synaptic_currents)
        synaptic_modulation = np.array(synaptic_modulation)

        if type(self.synaptic_modulation_direction) is float:
            self.synaptic_modulation_direction = [np.ones(len(synaptic_currents) + 2) *
                                                  self.synaptic_modulation_direction]

        # set modulation direction for each synapse
        synaptic_modulation_tmp = np.zeros((len(synaptic_modulation), len(self.synaptic_modulation_direction[0])))
        for i in range(len(synaptic_modulation)):
            synaptic_modulation_tmp[i] = synaptic_modulation[i] ** self.synaptic_modulation_direction[i]
        synaptic_modulation = np.prod(synaptic_modulation_tmp, axis=0)

        # combine extrinsic and intrinsic synaptic modulation
        if type(self.extrinsic_synaptic_modulation) is float:
            self.extrinsic_synaptic_modulation = np.ones(len(synaptic_currents) + 2) *\
                                                 self.extrinsic_synaptic_modulation
        synaptic_modulation *= self.extrinsic_synaptic_modulation

        return np.dot(synaptic_currents, synaptic_modulation[0:-2])

    def get_leak_current(self, membrane_potential):
        """
        Method that calculates the leakage current at a given point in time (instantaneous).

        :param membrane_potential: current membrane potential of population [unit = mV].

        :return: leak current [unit = mA].

        """

        return (self.resting_potential - membrane_potential) * self.membrane_capacitance / self.tau_leak

    def get_delta_membrane_potential_threshold(self, t, membrane_potential_threshold):
        """
        Method that calculate the change in the axonal membrane potential threshold given the current firing rate.

        :return: Delta membrane potential threshold (scalar) [unit = mV].

        """

        return (self.output_firing_rate[-1] - self.firing_rate_target) / self.tau_axon

    def update_step_size(self, new_step_size, interpolation_type='linear'):
        """
        Updates internal step-size to new step-size. Interpolates input history to fit new step-size.

        :param new_step_size: new step-size with which population dynamics develop [unit = s].
        :param interpolation_type: character string, indicates the type of interpolation used for up-/down-sampling the
               firing-rate look-up matrix (default = 'cubic').

        """

        step_size_diff = self.synapses[0].kernel_length * new_step_size - \
                         self.synapses[0].kernel_length * self.step_size

        # get input history
        inputs = np.asarray(self.input_firing_rate)

        if abs(step_size_diff) > new_step_size and inputs.shape[0] > 1:

            # interpolate input history to fit new step size
            inputs = interpolate_array(old_step_size=self.step_size,
                                       new_step_size=new_step_size,
                                       y=inputs,
                                       interpolation_type=interpolation_type,
                                       axis=0)

            # save new input history
            self.input_firing_rate = inputs.tolist()

            # set new step-size
            self.step_size = new_step_size

    def plot_synaptic_kernels(self, synapse_idx=None, create_plot=True, fig=None):
        """
        Creates plot of all specified synapses over time.

        :param synapse_idx: Can be list of synapse indices, specifying for which synapses to plot the kernels
               (default = None).
        :param create_plot: If false, no plot will be shown (default = True).
        :param fig: figure handle, can be passed optionally (default = None).

        :return: figure handle

        """

        ####################
        # check parameters #
        ####################

        assert synapse_idx is None or type(synapse_idx) is list

        #############################
        # check positional argument #
        #############################

        if synapse_idx is None:
            synapse_idx = range(len(self.synapses))

        #########################
        # plot synaptic kernels #
        #########################

        if fig is None:
            fig = figure('Synaptic Kernel Functions')

        synapse_types = list()
        for i in synapse_idx:
            fig = self.synapses[i].plot_synaptic_kernel(create_plot=False, fig=fig)
            synapse_types.append(self.synapses[i].synapse_type)

        legend(synapse_types)

        if create_plot:
            fig.show()

        return fig


def set_synapse(synapse, step_size, kernel_length, synapse_params=None):
    """
    Instantiates a synapse, including a method to calculate the synaptic current resulting from the average firing rate
    arriving at the population.

    :param synapse: character string that indicates synapse type (see pre-implemented synapse sub-classes)
    :param step_size: scalar, size of the time step for which the population state will be updated according
           to euler formalism [unit = s].
    :param kernel_length: scalar that indicates number of bins the kernel should be evaluated for
           [unit = 1].
    :param synapse_params: dictionary containing parameters for custom synapse type. For parameter explanation see
           synapse class (default = None).

    :return synapse_instance: instance of synapse class, including method to compute synaptic current

    """

    ####################
    # check parameters #
    ####################

    assert (type(synapse) is str) or (not synapse)
    assert (type(synapse_params) is dict) or (not synapse_params)
    assert step_size > 0
    assert kernel_length > 0

    ###############################
    # initialize synapse instance #
    ###############################

    pre_defined_synapse = True

    if synapse == 'AMPA_current':

        syn_instance = synapse_templates.AMPACurrentSynapse(step_size, kernel_length)

    elif synapse == 'GABAA_current':

        syn_instance = synapse_templates.GABAACurrentSynapse(step_size, kernel_length)

    elif synapse_params:

        syn_instance = Synapse(synapse_params['efficiency'], synapse_params['tau_decay'],
                                   synapse_params['tau_rise'], step_size, kernel_length)
        pre_defined_synapse = False

    else:

        raise ValueError('Not a valid synapse type! Check synapses.py for all possible synapse types')

    #########################################################################
    # adjust pre-defined parametrization if relevant parameters were passed #
    #########################################################################

    if pre_defined_synapse and synapse_params:

        param_list = ['efficiency', 'tau_decay', 'tau_rise', 'conductivity_based', 'reversal_potential']

        for p in param_list:
            syn_instance = update_param(p, synapse_params, syn_instance)

    syn_instance.synaptic_kernel = syn_instance.evaluate_kernel(build_kernel=True)

    return syn_instance


def set_axon(axon, axon_params=None):
    """
    Instantiates an axon, including a method to calculate the average output firing rate of a population given its
    current membrane potential.

    :param axon: character string that indicates axon type (see pre-implemented axon sub-classes)
    :param axon_params: dictionary containing parameters for custom axon type. For parameter explanation see axon
           class (default = None).

    :return ax_instance: instance of axon class, including method to compute firing rate

    """

    ####################
    # check parameters #
    ####################

    assert (type(axon) is str) or (not axon)
    assert (type(axon_params) is dict) or (not axon_params)

    ############################
    # initialize axon instance #
    ############################

    pre_defined_axon = True

    if axon == 'Knoesche':

        ax_instance = axon_templates.KnoescheAxon()

    elif axon == 'JansenRit':

        ax_instance = axon_templates.JansenRitAxon()

    elif axon_params:

        ax_instance = Axon(axon_params['max_firing_rate'], axon_params['membrane_potential_threshold'],
                              axon_params['sigmoid_steepness'])
        pre_defined_axon = False

    else:

        raise ValueError('Not a valid axon type!')

    #########################################################################
    # adjust pre-defined parametrization if relevant parameters were passed #
    #########################################################################

    if pre_defined_axon and axon_params:

        param_list = ['max_firing_rate', 'membrane_potential_threshold', 'sigmoid_steepness']

        for p in param_list:
            ax_instance = update_param(p, axon_params, ax_instance)

    return ax_instance


def update_param(param, param_dict, object_instance):
    """
    Checks whether param is a key in param_dict. If yes, the corresponding value in param_dict will be updated in
    object_instance.

    :param param: string, specifies parameter to check
    :param param_dict: dictionary, potentially contains param
    :param object_instance: object instance for which to update parameter

    :return: object
    """

    assert param in object_instance

    if param in param_dict:
        setattr(object_instance, param, param_dict[param])

    return object_instance


def interpolate_array(old_step_size, new_step_size, y, interpolation_type='cubic', axis=0):
    """
    Interpolates time-vectors with scipy.interpolate.interp1d.

    :param old_t: total time-length of y [unit = s].
    :param old_step_size: old simulation step size [unit = s].
    :param new_step_size: new simulation step size [unit = s].
    :param new_t: new total length of y [unit = s].
    :param y: vector to be interpolated
    :param interpolation_type: can be 'linear' or spline stuff.
    :param axis: axis along which y is to be interpolated (has to have same length as t/old_step_size)

    :return: interpolated vector

    """

    # create time vectors
    x_old = np.arange(y.shape[axis]) * old_step_size

    new_steps_n = int(np.ceil(x_old[-1] / new_step_size))
    x_new = np.linspace(0, x_old[-1], new_steps_n)

    # create interpolation function
    f = interp1d(x_old, y, axis=axis, kind=interpolation_type, bounds_error=False, fill_value='extrapolate')

    return f(x_new)
