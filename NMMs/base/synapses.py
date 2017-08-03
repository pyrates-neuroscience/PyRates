"""
Includes basic synapse class, pre-parametrized sub-classes and an exponential kernel function
"""

import numpy as np

__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


class Synapse(object):
    """
    Basic class for synapses. Includes a method to calculate synaptic current based on firing rate input.
    
    :var synapse_type: string, indicates type of synapse
    :var synaptic_kernel: vector including weights for a certain number of time-steps after a synaptic input has
         arrived, determining how strong that input affects the synapse at each time-step.
                              
    """

    def __init__(self, efficiency, tau_decay, tau_rise, step_size, kernel_length, conductivity_based=False,
                 reversal_potential=-70.0, synapse_type=None):
        """
        Initializes a standard synapse exponential synapse kernel defined by 3 parameters.

        :param efficiency: scalar, real-valued, defines strength and effect (excitatory vs inhibitory) of synapse
        :param tau_decay: scalar, positive & real-valued, lumped time delay constant that determines how steep
               the exponential synaptic kernel decays
        :param tau_rise: scalar, positive & real-valued, lumped time delay constant that determines how steep the
               exponential synaptic kernel rises
        :param step_size: scalar, size of the time step for which the population state will be updated according
               to euler formalism [unit = s].
        :param kernel_length: scalar that indicates number of bins the kernel should be evaluated for
               [unit = time-steps].
        :param conductivity_based: if true, synaptic input will be translated into synaptic current indirectly via
               a change in synaptic conductivity. Else translation to synaptic current will be direct.
        :param reversal_potential: scalar, determines the reversal potential of the synapse. Only necessary for
               conductivity based synapses [unit = mV] (default = 0).
        :param synapse_type: if not None, will be treated as type of synapse (should be character string)

        """

        ##########################
        # check input parameters #
        ##########################

        assert type(efficiency) is float
        assert tau_decay >= 0 and type(tau_decay) is float
        assert tau_rise >= 0 and type(tau_rise) is float
        assert step_size >= 0 and type(step_size) is float
        assert kernel_length > 0 and type(kernel_length) is int

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

        time_points = np.arange(kernel_length - 1, 0, -1)
        time_points = time_points * step_size
        self.synaptic_kernel = evaluate_kernel(time_points, efficiency, tau_decay, tau_rise)

    def get_synaptic_current(self, x, membrane_potential=None):
        """
        Applies synaptic kernel to input vector (should resemble incoming firing rate).

        :param x: vector of values, kernel should be applied to [unit = firing rate].
        :param membrane_potential: scalar, determines membrane potential of post-synapse. Only to be used for
               conductivity based synapses [unit = mV] (default = None).

        :return: resulting synaptic current [mA]

        """

        #########################
        # apply synaptic kernel #
        #########################

        if len(x) < len(self.synaptic_kernel):
            kernel_value = x @ self.synaptic_kernel[-len(x):]
        else:
            kernel_value = x[-len(self.synaptic_kernel):] @ self.synaptic_kernel

        ##############################
        # calculate synaptic current #
        ##############################

        if membrane_potential is None:
            synaptic_current = kernel_value
        else:
            synaptic_current = kernel_value * (self.reversal_potential - membrane_potential)

        return synaptic_current


def evaluate_kernel(t, efficiency, tau_decay, tau_rise):
    """
    Computes value of synaptic kernel at specific time point

    :param t: scalar, determines time point at which synaptic kernel value should be evaluated [unit = s]
    :param efficiency: scalar, real-valued, defines strength and effect (excitatory vs inhibitory) of synapse
           [unit = mA or mS/m]
    :param tau_decay: scalar, positive & real-valued, lumped time delay constant that determines how steep
           the exponential synaptic kernel decays [unit = s]
    :param tau_rise: scalar, positive & real-valued, lumped time delay constant that determines how steep the
           exponential synaptic kernel rises [unit = s]

    :return: scalar, synaptic kernel value [unit = mA or mS/m]

    """

    #########################
    # check input parameter #
    #########################

    if type(t) is float:
        assert t >= 0
    else:
        assert all(t) >= 0

    return efficiency * (np.exp(-t / tau_decay) - np.exp(-t / tau_rise))


class AMPACurrentSynapse(Synapse):
    """
    This defines an current-based synapse with AMPA neuroreceptor
    """

    def __init__(self, step_size, kernel_length, efficiency=1.273*3e-3, tau_decay=0.006, tau_rise=0.0006):
        """
        Initializes a current-based synapse with parameters resembling an AMPA receptor.

        :param step_size: scalar, size of the time step for which the population state will be updated according
               to euler formalism [unit = s].
        :param kernel_length: scalar that indicates number of bins the kernel should be evaluated for
               [unit = time-steps].

        """

        super(AMPACurrentSynapse, self).__init__(efficiency, tau_decay, tau_rise, step_size, kernel_length,
                                                 synapse_type='AMPA_current')


class GABAACurrentSynapse(Synapse):
    """
    This defines a current-based synapse with GABA_A neuroreceptor.
    """

    def __init__(self, step_size, kernel_length, efficiency=1.273*(-1e-2), tau_decay=0.02, tau_rise=0.0004):
        """
        Initializes a current-based synapse with parameters resembling an GABA_A receptor.

        :param step_size: scalar, size of the time step for which the population state will be updated according
               to euler formalism [unit = s].
        :param kernel_length: scalar that indicates number of bins the kernel should be evaluated for
               [unit = time-steps].

        """

        super(GABAACurrentSynapse, self).__init__(efficiency, tau_decay, tau_rise, step_size, kernel_length,
                                                  synapse_type='GABAA_current')
