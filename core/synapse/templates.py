"""Templates for specific synapse parametrizations.
"""

from core.synapse import DoubleExponentialSynapse, ExponentialSynapse
from typing import Optional

__author__ = "Richard Gast, Daniel F. Rose"
__status__ = "Development"


############################
# leaky capacitor synapses #
############################


class AMPACurrentSynapse(DoubleExponentialSynapse):
    """Current-based synapse with AMPA neuroreceptor.

    Parameters
    ----------
    bin_size
        See documentation of parameter `bin_size` of :class:`Synapse`.
    max_delay
        See documentation of parameter `max_delay` of :class:`Synapse`.
    efficacy
        Default = 1.273 * 3e-13 A. See Also documentation of parameter `efficacy` of :class:`Synapse`.
    tau_decay
        Default = 0.006 s. See Also documentation of parameter `tau_decay` of :class:`DoubleExponentialSynapse`.
    tau_rise
        Default = 0.0006 s. See Also documentation of parameter `tau_rise` of :class:`DoubleExponentialSynapse`.

    See Also
    --------
    :class:`DoubleExponentialSynapse`: Detailed documentation of parameters of double exponential synapse.
    :class:`Synapse`: Detailed documentation of synapse attributes and methods.

    """

    def __init__(self,
                 bin_size: float,
                 max_delay: float,
                 efficacy: float = 1.273 * 3e-13,
                 tau_decay: float = 0.006,
                 tau_rise: float = 0.0006
                 ) -> None:
        """
        Instantiates current-based synapse with AMPA receptor.
        """

        super(AMPACurrentSynapse, self).__init__(efficacy=efficacy,
                                                 tau_decay=tau_decay,
                                                 tau_rise=tau_rise,
                                                 bin_size=bin_size,
                                                 max_delay=max_delay,
                                                 synapse_type='AMPA_current',
                                                 modulatory=False)


class GABAACurrentSynapse(DoubleExponentialSynapse):
    """Current-based synapse with GABA_A neuroreceptor.

    Parameters
    ----------
    bin_size
        See documentation of parameter `bin_size` of :class:`Synapse`.
    max_delay
        See documentation of parameter `max_delay` of :class:`Synapse`.
    efficacy
        Default = 1.273 * -1e-12 A. See Also documentation of parameter `efficacy` of :class:`Synapse`.
    tau_decay
        Default = 0.02 s. See Also documentation of parameter `tau_decay` of :class:`DoubleExponentialSynapse`.
    tau_rise
        Default = 0.0004 s. See Also documentation of parameter `tau_rise` of :class:`DoubleExponentialSynapse`.

    See Also
    --------
    :class:`DoubleExponentialSynapse`: Detailed documentation of parameters of double exponential synapse.
    :class:`Synapse`: Detailed documentation of synapse attributes and methods.

    """

    def __init__(self,
                 bin_size: float,
                 max_delay: float,
                 efficacy: float = 1.273 * (-1e-12),
                 tau_decay: float = 0.02,
                 tau_rise: float = 0.0004
                 ) -> None:
        """
        Instantiates current-based synapse with GABA_A receptor.
        """

        super(GABAACurrentSynapse, self).__init__(efficacy=efficacy,
                                                  tau_decay=tau_decay,
                                                  tau_rise=tau_rise,
                                                  bin_size=bin_size,
                                                  max_delay=max_delay,
                                                  synapse_type='GABAA_current',
                                                  modulatory=False)


class AMPAConductanceSynapse(DoubleExponentialSynapse):
    """Defines a conductivity-based synapse with AMPA neuroreceptor.

    Parameters
    ----------
    bin_size
        See documentation of parameter `bin_size` of :class:`Synapse`.
    max_delay
        See documentation of parameter `max_delay` of :class:`Synapse`.
    efficacy
        Default = 1.273 * 7.2e-10 A. See Also documentation of parameter `efficacy` of :class:`Synapse`.
    tau_decay
        Default = 0.0015 s. See Also documentation of parameter `tau_decay` of :class:`DoubleExponentialSynapse`.
    tau_rise
        Default = 0.000009 s. See Also documentation of parameter `tau_rise` of :class:`DoubleExponentialSynapse`.
    reversal_potential
        Default = 0.0 V. See Also documentation of parameter `reversal_potential` of :class:`Synapse`.

    See Also
    --------
    :class:`DoubleExponentialSynapse`: Detailed documentation of parameters of double exponential synapse.
    :class:`Synapse`: Detailed documentation of synapse attributes and methods.

    """

    def __init__(self,
                 bin_size: float,
                 max_delay: int,
                 efficacy: float = 1.273 * 7.2e-10,
                 tau_decay: float = 0.0015,
                 tau_rise: float = 0.000009,
                 reversal_potential: float = 0.0
                 ) -> None:
        """
        Instantiates a conductance-based synapse with GABA_A receptor.
        """

        super(AMPAConductanceSynapse, self).__init__(efficacy=efficacy,
                                                     tau_decay=tau_decay,
                                                     tau_rise=tau_rise,
                                                     bin_size=bin_size,
                                                     max_delay=max_delay,
                                                     reversal_potential=reversal_potential,
                                                     synapse_type='AMPA_conductance',
                                                     modulatory=False,
                                                     conductivity_based=True)


class GABAAConductanceSynapse(DoubleExponentialSynapse):
    """Conductivity-based synapse with GABA_A neuroreceptor.

    Parameters
    ----------
    bin_size
        See documentation of parameter `bin_size` of :class:`Synapse`.
    max_delay
        See documentation of parameter `max_delay` of :class:`Synapse`.
    efficacy
        Default = 1.358 * -4e-11 A. See Also documentation of parameter `efficacy` of :class:`Synapse`.
    tau_decay
        Default = 0.02 s. See Also documentation of parameter `tau_decay` of :class:`DoubleExponentialSynapse`.
    tau_rise
        Default = 0.0004 s. See Also documentation of parameter `tau_rise` of :class:`DoubleExponentialSynapse`.
    reversal_potential
        Default = -0.060 V. See Also documentation of parameter `reversal_potential` of :class:`Synapse`.

    See Also
    --------
    :class:`DoubleExponentialSynapse`: Detailed documentation of parameters of double exponential synapse.
    :class:`Synapse`: Detailed documentation of synapse attributes and methods.

    """

    def __init__(self,
                 bin_size: float,
                 max_delay: float,
                 efficacy: float = 1.358 * (-4e-11),
                 tau_decay: float = 0.02,
                 tau_rise: float = 0.0004,
                 reversal_potential: float = -0.070
                 ) -> None:
        """
        Instantiates a current-based synapse with GABA_A receptor.
        """

        super().__init__(efficacy=efficacy,
                         tau_decay=tau_decay,
                         tau_rise=tau_rise,
                         bin_size=bin_size,
                         max_delay=max_delay,
                         reversal_potential=reversal_potential,
                         synapse_type='GABAA_current',
                         modulatory=False,
                         conductivity_based=True)


###########################
# JansenRit-type synapses #
###########################


class JansenRitExcitatorySynapse(ExponentialSynapse):
    """Excitatory second-order synapse as defined in [1]_.

        Parameters
        ----------
        bin_size
            See documentation of parameter `bin_size` of :class:`Synapse`.
        epsilon
            See documentation of parameter `epsilon` of :class:`Synapse`.
        max_delay
            See documentation of parameter `max_delay` of :class:`Synapse`.
        efficacy
            Default = 3.25 * 1e-3 V. See Also documentation of parameter `efficacy` of :class:`Synapse`.
        tau
            Default = 0.01 s. See Also documentation of parameter `tau` of :class:`ExponentialSynapse`.

        See Also
        --------
        :class:`ExponentialSynapse`: Detailed documentation of parameters of exponential synapse.
        :class:`Synapse`: Detailed documentation of synapse attributes and methods.

        References
        ----------
        .. [1] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical
           model of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.

        """

    def __init__(self,
                 bin_size: float,
                 epsilon: float = 5e-5,
                 max_delay: Optional[float] = None,
                 efficacy: float = 3.25 * 1e-3,
                 tau: float = 0.01
                 ) -> None:
        """Initializes excitatory exponential synapse as defined in [1]_.
        """

        super().__init__(efficacy=efficacy,
                         tau=tau,
                         bin_size=bin_size,
                         epsilon=epsilon,
                         max_delay=max_delay,
                         synapse_type='JR_excitatory')


class JansenRitInhibitorySynapse(ExponentialSynapse):
    """Inhibitory second-order synapse as defined in [1]_.

        Parameters
        ----------
        bin_size
            See documentation of parameter `bin_size` of :class:`Synapse`.
        epsilon
            See documentation of parameter `epsilon` of :class:`Synapse`.
        max_delay
            See documentation of parameter `max_delay` of :class:`Synapse`.
        efficacy
            Default = -22 * 1e-3 V. See Also documentation of parameter `efficacy` of :class:`Synapse`.
        tau
            Default = 0.02 s. See Also documentation of parameter `tau` of :class:`ExponentialSynapse`.

        See Also
        --------
        :class:`ExponentialSynapse`: Detailed documentation of parameters of exponential synapse.
        :class:`Synapse`: Detailed documentation of synapse attributes and methods.

        References
        ----------
        .. [1] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical
           model of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.

        """

    def __init__(self,
                 bin_size: float,
                 epsilon: float = 5e-5,
                 max_delay: Optional[float] = None,
                 efficacy: float = -22e-3,
                 tau: float = 0.02
                 ) -> None:
        """Initializes excitatory exponential synapse as defined in [1]_.
        """

        super().__init__(efficacy=efficacy,
                         tau=tau,
                         bin_size=bin_size,
                         epsilon=epsilon,
                         max_delay=max_delay,
                         synapse_type='JR_inhibitory')


class MoranExcitatorySynapse(ExponentialSynapse):
    """Excitatory second-order synapse as defined in [1]_.

        Parameters
        ----------
        bin_size
            See documentation of parameter `bin_size` of :class:`Synapse`.
        epsilon
            See documentation of parameter `epsilon` of :class:`Synapse`.
        max_delay
            See documentation of parameter `max_delay` of :class:`Synapse`.
        efficacy
            Default = 4e-3 V. See Also documentation of parameter `efficacy` of :class:`Synapse`.
        tau
            Default = 4e-3 s. See Also documentation of parameter `tau` of :class:`ExponentialSynapse`.

        See Also
        --------
        :class:`ExponentialSynapse`: Detailed documentation of parameters of exponential synapse.
        :class:`Synapse`: Detailed documentation of synapse attributes and methods.

        References
        ----------
        .. [1] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical
           model of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.

        """

    def __init__(self,
                 bin_size: float,
                 epsilon: float = 5e-5,
                 max_delay: Optional[float] = None,
                 efficacy: float = 4e-3,
                 tau: float = 4e-3
                 ) -> None:
        """Initializes excitatory exponential synapse as defined in [1]_.
        """

        super().__init__(efficacy=efficacy,
                         tau=tau,
                         bin_size=bin_size,
                         epsilon=epsilon,
                         max_delay=max_delay,
                         synapse_type='Moran_excitatory')


class MoranInhibitorySynapse(ExponentialSynapse):
    """Inhibitory second-order synapse as defined in [1]_.

        Parameters
        ----------
        bin_size
            See documentation of parameter `bin_size` of :class:`Synapse`.
        epsilon
            See documentation of parameter `epsilon` of :class:`Synapse`.
        max_delay
            See documentation of parameter `max_delay` of :class:`Synapse`.
        efficacy
            Default = -32e-3 V. See Also documentation of parameter `efficacy` of :class:`Synapse`.
        tau
            Default = 16e-3 s. See Also documentation of parameter `tau` of :class:`ExponentialSynapse`.

        See Also
        --------
        :class:`ExponentialSynapse`: Detailed documentation of parameters of exponential synapse.
        :class:`Synapse`: Detailed documentation of synapse attributes and methods.

        References
        ----------
        .. [1] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical
           model of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.

        """

    def __init__(self,
                 bin_size: float,
                 epsilon: float = 5e-5,
                 max_delay: Optional[float] = None,
                 efficacy: float = -32e-3,
                 tau: float = 16e-3
                 ) -> None:
        """Initializes excitatory exponential synapse as defined in [1]_.
        """

        super().__init__(efficacy=efficacy,
                         tau=tau,
                         bin_size=bin_size,
                         epsilon=epsilon,
                         max_delay=max_delay,
                         synapse_type='Moran_inhibitory')
