"""
Includes basic synapse class, pre-parametrized sub-classes and an exponential kernel function
"""

import numpy as np
from scipy.integrate import trapz
from matplotlib.pyplot import *

__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"

# TODO: find out reversal potentials for conductance based synapses


class Synapse(object):
    """
    Basic class for synapses. Includes a method to calculate synaptic current based on firing rate input.
    
    :var synapse_type: string, indicates type of synapse
    :var synaptic_kernel: vector including weights for a certain number of time-steps after a synaptic input has
         arrived, determining how strong that input affects the synapse at each time-step.
    :var efficiency: scalar, determines strength of the synaptic response to input.
    :var tau_decay: scalar, determines time-scale with which synaptic response to previous input decays.
    :var tau_rise: scalar, determines time-scale with which synaptic response to previous input increases.
    :var step_size: scalar, determines length of time-interval between time-steps.
    :var kernel_length: scalar, determines length of the synaptic response kernel.
    :var reversal_potential: scalar, determines reversal potential for that specific synapse.
                              
    """

    def __init__(self, efficiency, tau_decay, tau_rise, step_size, kernel_length, conductivity_based=False,
                 reversal_potential=-0.075, synapse_type=None, neuromodulatory=False):
        """
        Initializes a standard double-exponential synapse kernel defined by 3 parameters.

        :param efficiency: scalar, real-valued, defines strength and effect (excitatory vs inhibitory) of synapse
               [unit = A or S].
        :param tau_decay: scalar, positive & real-valued, lumped time delay constant that determines how steep
               the exponential synaptic kernel decays [unit = s].
        :param tau_rise: scalar, positive & real-valued, lumped time delay constant that determines how steep the
               exponential synaptic kernel rises [unit = s].
        :param step_size: scalar, size of the time step for which the population state will be updated according
               to euler formalism [unit = s].
        :param kernel_length: scalar that indicates number of bins the kernel should be evaluated for
               [unit = 1].
        :param conductivity_based: if true, synaptic input will be translated into synaptic current indirectly via
               a change in synaptic conductivity. Else translation to synaptic current will be direct.
        :param reversal_potential: scalar, determines the reversal potential of the synapse. Only necessary for
               conductivity based synapses [unit = V] (default = -0.075).
        :param synapse_type: if not None, will be treated as type of synapse (should be character string)
        :param neuromodulatory: if True, synapse will be treated as having a modulatory (multiplicative) effect.
               Synaptic output is then treated as having unit 1.

        """

        ##########################
        # check input parameters #
        ##########################

        assert type(efficiency) is float
        assert tau_decay >= 0 and type(tau_decay) is float
        assert tau_rise >= 0 and type(tau_rise) is float
        assert step_size >= 0 and type(step_size) is float
        assert kernel_length > 0 and type(kernel_length) is int

        ##################
        # set parameters #
        ##################

        self.efficiency = efficiency
        self.tau_decay = tau_decay
        self.tau_rise = tau_rise
        self.conductivity_based = conductivity_based
        self.reversal_potential = reversal_potential
        self.step_size = step_size
        self.kernel_length = kernel_length
        self.neuromodulatory = neuromodulatory

        ####################
        # set synapse type #
        ####################

        if synapse_type is None:

            # define synapse type via synaptic kernel efficiency and type (current- vs conductivity-based)
            if conductivity_based:
                self.reversal_potential = reversal_potential
                self.synapse_type = 'excitatory_conductance' if efficiency >= 0 else 'inhibitory_conductance'
            else:
                self.synapse_type = 'excitatory_current' if efficiency >= 0 else 'inhibitory_current'

        else:

            self.synapse_type = synapse_type

        #########################
        # build synaptic kernel #
        #########################

        self.synaptic_kernel = self.evaluate_kernel(build_kernel=True)

    def evaluate_kernel(self, build_kernel, t=0.):
        """
        Computes value of synaptic kernel at specific time point

        :param build_kernel: if true, kernel will be evaluated at kernel_length timepoints. Else, t needs to be provided
               and kernel will be evaluated at t.
        :param t: scalar or vector, time(s) at which to evaluate kernel [unit = s] (default = 0.).

        :return: scalar or vector, synaptic kernel value at each t [unit = A or S]

        """

        #########################
        # check input parameter #
        #########################

        if type(t) is float:
            assert t >= 0
        else:
            assert all(t) >= 0

        ############################################################
        # check whether to build kernel or just evaluate its value #
        ############################################################

        if build_kernel:

            t = np.arange(self.kernel_length-1, -1, -1)
            t = t * self.step_size

        return self.efficiency * (np.exp(-t / self.tau_decay) - np.exp(-t / self.tau_rise))

    def get_synaptic_current(self, x, membrane_potential=None):
        """
        Applies synaptic kernel to input vector (should resemble incoming firing rate).

        :param x: vector of values, kernel should be applied to [unit = 1/s].
        :param membrane_potential: scalar, determines membrane potential of post-synapse. Only to be used for
               conductivity based synapses [unit = V] (default = None).

        :return: resulting synaptic current [unit = A]

        """

        assert all(x) >= 0
        assert membrane_potential is None or type(membrane_potential) is float

        #########################
        # apply synaptic kernel #
        #########################

        # multiply firing rate input with kernel
        if len(x) < len(self.synaptic_kernel):
            kernel_value = x * self.synaptic_kernel[-len(x):]
        else:
            kernel_value = x[-len(self.synaptic_kernel):] * self.synaptic_kernel

        # integrate over time
        kernel_value = trapz(kernel_value, dx=self.step_size)

        ##############################
        # calculate synaptic current #
        ##############################

        if membrane_potential is None:
            synaptic_current = kernel_value
        else:
            synaptic_current = kernel_value * (self.reversal_potential - membrane_potential)

        return synaptic_current

    def plot_synaptic_kernel(self, create_plot=True, fig=None):
        """
        Creates plot of synaptic kernel over time.

        :param create_plot: If false, no plot will be shown (default = True).
        :param fig: figure handle, can be passed optionally (default = None).

        :return: figure handle

        """

        ##################################
        # plot synaptic kernel over time #
        ##################################

        if fig is None:
            fig = figure('Synaptic Kernel')

        hold('on')
        plot(self.synaptic_kernel[-1:0:-1])
        hold('off')
        xlabel('time-steps')
        if self.neuromodulatory:
            ylabel('modulation strength')
        elif self.conductivity_based:
            ylabel('synaptic conductivity [S]')
        else:
            ylabel('synaptic current [A]')
        title('Impulse Response Function')

        if create_plot:
            fig.show()

        return fig


class AMPACurrentSynapse(Synapse):
    """
    This defines a current-based synapse with AMPA neuroreceptor.
    """

    def __init__(self, step_size, kernel_length, efficiency=1.273*3e-13, tau_decay=0.006, tau_rise=0.0006):
        """
        Initializes a current-based synapse with parameters resembling an AMPA receptor.

        :param step_size: scalar, size of the time step for which the population state will be updated according
               to euler formalism [unit = s].
        :param kernel_length: scalar that indicates number of bins the kernel should be evaluated for
               [unit = 1].

        """

        super(AMPACurrentSynapse, self).__init__(efficiency=efficiency,
                                                 tau_decay=tau_decay,
                                                 tau_rise=tau_rise,
                                                 step_size=step_size,
                                                 kernel_length=kernel_length,
                                                 synapse_type='AMPA_current',
                                                 neuromodulatory=False)


class GABAACurrentSynapse(Synapse):
    """
    This defines a current-based synapse with GABA_A neuroreceptor.
    """

    def __init__(self, step_size, kernel_length, efficiency=1.273*(-1e-12), tau_decay=0.02, tau_rise=0.0004):
        """
        Initializes a current-based synapse with parameters resembling an GABA_A receptor.

        :param step_size: scalar, size of the time step for which the population state will be updated according
               to euler formalism [unit = s].
        :param kernel_length: scalar that indicates number of bins the kernel should be evaluated for
               [unit = 1].

        """

        super(GABAACurrentSynapse, self).__init__(efficiency=efficiency,
                                                  tau_decay=tau_decay,
                                                  tau_rise=tau_rise,
                                                  step_size=step_size,
                                                  kernel_length=kernel_length,
                                                  synapse_type='GABAA_current',
                                                  neuromodulatory=False)


class AMPAConductanceSynapse(Synapse):
    """
    This defines a conductivity-based synapse with AMPA neuroreceptor.
    """

    def __init__(self, step_size, kernel_length, efficiency=1.273*7.2e-10, tau_decay=0.0015, tau_rise=0.000009,
                 reversal_potential=0.0):
        """
        Initializes a current-based synapse with parameters resembling an GABA_A receptor.

        :param step_size: scalar, size of the time step for which the population state will be updated according
               to euler formalism [unit = s].
        :param kernel_length: scalar that indicates number of bins the kernel should be evaluated for
               [unit = 1].

        """

        super(AMPAConductanceSynapse, self).__init__(efficiency=efficiency,
                                                     tau_decay=tau_decay,
                                                     tau_rise=tau_rise,
                                                     step_size=step_size,
                                                     kernel_length=kernel_length,
                                                     reversal_potential=reversal_potential,
                                                     synapse_type='AMPA_conductance',
                                                     neuromodulatory=False)


class GABAAConductanceSynapse(Synapse):
    """
    This defines a conductivity-based synapse with GABA_A neuroreceptor.
    """

    def __init__(self, step_size, kernel_length, efficiency=1.358*(-4e-11), tau_decay=0.02, tau_rise=0.0004,
                 reversal_potential=-0.060):
        """
        Initializes a current-based synapse with parameters resembling an GABA_A receptor.

        :param step_size: scalar, size of the time step for which the population state will be updated according
               to euler formalism [unit = s].
        :param kernel_length: scalar that indicates number of bins the kernel should be evaluated for
               [unit = 1].

        """

        super(GABAAConductanceSynapse, self).__init__(efficiency=efficiency,
                                                      tau_decay=tau_decay,
                                                      tau_rise=tau_rise,
                                                      step_size=step_size,
                                                      kernel_length=kernel_length,
                                                      reversal_potential=reversal_potential,
                                                      synapse_type='GABAA_current',
                                                      neuromodulatory=False)
