"""
"""
from synapse import Synapse

__author__ = "Richard Gast, Daniel F. Rose"
__status__ = "Development"


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