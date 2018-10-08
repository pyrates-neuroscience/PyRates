"""Templates for specific synapse parametrizations.
"""

# external packages
from typing import Optional
import tensorflow as tf

# pyrates internal imports
from pyrates.synapse import DoubleExponentialSynapse, TransformedInputSynapse, \
    DEExponentialSynapse, DEDoubleExponentialSynapse
from pyrates.utility import synaptic_sigmoid, double_exponential

# meta infos
__author__ = "Richard Gast, Daniel F. Rose"
__status__ = "Development"


#################################
# integro-differential synapses #
#################################


class AMPACurrentSynapse(DoubleExponentialSynapse):
    """Current-based synapse with AMPA neuroreceptor.

    See Also
    --------
    :class:`DoubleExponentialSynapse`: Detailed documentation of parameters of double exponential synapse.
    :class:`Synapse`: Detailed documentation of synapse attributes and methods.

    """

    def __init__(self,
                 bin_size: float,
                 max_delay: Optional[float] = None,
                 buffer_size: float = 0.,
                 epsilon: float = 1e-15,
                 efficacy: float = 1.273 * 3e-13,
                 tau_decay: float = 0.006,
                 tau_rise: float = 0.0006
                 ) -> None:
        """
        Instantiates current-based synapse with AMPA receptor.
        """

        super().__init__(efficacy=efficacy,
                         tau_decay=tau_decay,
                         tau_rise=tau_rise,
                         bin_size=bin_size,
                         max_delay=max_delay,
                         buffer_size=buffer_size,
                         epsilon=epsilon,
                         key='AMPA_current'
                         )


class GABAACurrentSynapse(DoubleExponentialSynapse):
    """Current-based synapse with GABA_A neuroreceptor.

    See Also
    --------
    :class:`DoubleExponentialSynapse`: Detailed documentation of parameters of double exponential synapse.
    :class:`Synapse`: Detailed documentation of synapse attributes and methods.

    """

    def __init__(self,
                 bin_size: float,
                 max_delay: Optional[float] = None,
                 buffer_size: float = 0.,
                 epsilon: float = 1e-15,
                 efficacy: float = -1.273 * 1e-12,
                 tau_decay: float = 0.02,
                 tau_rise: float = 0.0004
                 ) -> None:
        """
        Instantiates current-based synapse with GABA_A receptor.
        """

        super().__init__(efficacy=efficacy,
                         tau_decay=tau_decay,
                         tau_rise=tau_rise,
                         bin_size=bin_size,
                         max_delay=max_delay,
                         buffer_size=buffer_size,
                         epsilon=epsilon,
                         key='GABAA_current'
                         )


class GABABCurrentSynapse(TransformedInputSynapse):
    """Defines a current-based synapse with GABAB neuroreceptor.
    
    Parameters
    ----------
    bin_size
        See documentation of parameter `bin_size` of :class:`Synapse`.
    max_delay
        See documentation of parameter `max_delay` of :class:`Synapse`.
    buffer_size
        See documentation of parameter `buffer_size` of :class:`Synapse`.
    epsilon
        See documentation of parameter `epsilon` of :class:`Synapse`.
    efficacy
        See documentation of parameter `efficacy` of :class:`Synapse`.
    tau_decay
        See documentation of parameter `tau_decay` of :class:`DoubleExponentialSynapse`.
    tau_rise
        See documentation of parameter `tau_rise` of :class:`DoubleExponentialSynapse`.
    threshold
        Threshold of the sigmoidal transform applied to input of the synapse.
    steepness
        Steepness of the sigmoidal transform applied to input of the synapse.

    See Also
    --------
    :class:`TransformedInputSynapse`: Detailed documentation of parameters of synapses with additional input transform.
    :class:`Synapse`: Detailed documentation of synapse attributes and methods.
    
    """

    def __init__(self,
                 bin_size: float,
                 max_delay: Optional[float] = None,
                 buffer_size: float = 0.,
                 epsilon: float = 1e-15,
                 efficacy: float = -1.273 * 1e-12,
                 tau_decay: float = 0.02,
                 tau_rise: float = 0.0004,
                 threshold: float = 11.,
                 steepness: float = 100.,
                 max_firing_rate: float = 1.
                 ) -> None:
        """
        Instantiates current-based synapse with GABA_B receptor.
        """

        # define input transform attributes
        ###################################

        input_transform_args = {'threshold': threshold,
                                'steepness': steepness,
                                'max_firing_rate': max_firing_rate}

        # call super method
        ###################

        super().__init__(kernel_function=double_exponential,
                         input_transform=synaptic_sigmoid,
                         efficacy=efficacy,
                         max_delay=max_delay,
                         buffer_size=buffer_size,
                         bin_size=bin_size,
                         epsilon=epsilon,
                         key='GABAB_current',
                         input_transform_kwargs=input_transform_args,
                         tau_decay=tau_decay,
                         tau_rise=tau_rise
                         )


class AMPAConductanceSynapse(DoubleExponentialSynapse):
    """Defines a conductivity-based synapse with AMPA neuroreceptor.

    See Also
    --------
    :class:`DoubleExponentialSynapse`: Detailed documentation of parameters of double exponential synapse.
    :class:`Synapse`: Detailed documentation of synapse attributes and methods.

    """

    def __init__(self,
                 bin_size: float,
                 max_delay: Optional[float] = None,
                 buffer_size: float = 0.,
                 epsilon: float = 1e-13,
                 efficacy: float = 7.2e-10 * 1.273,
                 tau_decay: float = 1.0e-3,
                 tau_rise: float = 2e-4,
                 reversal_potential: float = 0.0
                 ) -> None:
        """
        Instantiates a conductance-based synapse with GABA_A receptor.
        """

        super().__init__(efficacy=efficacy,
                         tau_decay=tau_decay,
                         tau_rise=tau_rise,
                         bin_size=bin_size,
                         max_delay=max_delay,
                         buffer_size=buffer_size,
                         epsilon=epsilon,
                         reversal_potential=reversal_potential,
                         key='AMPA_conductance')


class GABAAConductanceSynapse(DoubleExponentialSynapse):
    """Conductivity-based synapse with GABA_A neuroreceptor.

    See Also
    --------
    :class:`DoubleExponentialSynapse`: Detailed documentation of parameters of double exponential synapse.
    :class:`Synapse`: Detailed documentation of synapse attributes and methods.

    """

    def __init__(self,
                 bin_size: float,
                 max_delay: Optional[float] = None,
                 buffer_size: float = 0.,
                 epsilon: float = 1e-13,
                 efficacy: float = 4e-11 * 1.358,
                 tau_decay: float = 6e-3,  # 0.02
                 tau_rise: float = 3e-4,
                 reversal_potential: float = -0.080
                 ) -> None:
        """
        Instantiates a current-based synapse with GABA_A receptor.
        """

        super().__init__(efficacy=efficacy,
                         tau_decay=tau_decay,
                         tau_rise=tau_rise,
                         bin_size=bin_size,
                         max_delay=max_delay,
                         buffer_size=buffer_size,
                         epsilon=epsilon,
                         reversal_potential=reversal_potential,
                         key='GABAA_conductance')


#########################
# differential synapses #
#########################


class AMPACurrentDESynapse(DEDoubleExponentialSynapse):
    """Current-based synapse with AMPA neuroreceptor.

    See Also
    --------
    :class:`DoubleExponentialSynapse`: Detailed documentation of parameters of double exponential synapse.
    :class:`Synapse`: Detailed documentation of synapse attributes and methods.

    """

    def __init__(self,
                 buffer_size: int = 0,
                 efficacy: float = 1.273 * 3e-13,
                 tau_decay: float = 0.006,
                 tau_rise: float = 0.0006
                 ) -> None:
        """
        Instantiates current-based synapse with AMPA receptor.
        """

        super().__init__(efficacy=efficacy,
                         tau_decay=tau_decay,
                         tau_rise=tau_rise,
                         buffer_size=buffer_size,
                         key='AMPA_current'
                         )


class GABAACurrentDESynapse(DEDoubleExponentialSynapse):
    """Current-based synapse with GABA_A neuroreceptor.

    See Also
    --------
    :class:`DoubleExponentialSynapse`: Detailed documentation of parameters of double exponential synapse.
    :class:`Synapse`: Detailed documentation of synapse attributes and methods.

    """

    def __init__(self,
                 buffer_size: int = 0,
                 efficacy: float = -1.273 * 1e-12,
                 tau_decay: float = 0.02,
                 tau_rise: float = 0.0004
                 ) -> None:
        """
        Instantiates current-based synapse with GABA_A receptor.
        """

        super().__init__(efficacy=efficacy,
                         tau_decay=tau_decay,
                         tau_rise=tau_rise,
                         buffer_size=buffer_size,
                         key='GABAA_current'
                         )


class GABABDESynapse(DEDoubleExponentialSynapse):
    """Defines a current-based synapse with GABAB neuroreceptor.

    Parameters
    ----------
    buffer_size
        See documentation of parameter `buffer_size` of :class:`Synapse`.
    efficacy
        See documentation of parameter `efficacy` of :class:`Synapse`.
    tau_decay
        See documentation of parameter `tau_decay` of :class:`DoubleExponentialSynapse`.
    tau_rise
        See documentation of parameter `tau_rise` of :class:`DoubleExponentialSynapse`.
    max_firing_rate
         Maximum output of the sigmoidal transform applied to input of the synapse.
    threshold
        Threshold of the sigmoidal transform applied to input of the synapse.
    steepness
        Steepness of the sigmoidal transform applied to input of the synapse.

    See Also
    --------
    :class:`TransformedInputSynapse`: Detailed documentation of parameters of synapses with additional input transform.
    :class:`Synapse`: Detailed documentation of synapse attributes and methods.

    """

    def __init__(self,
                 buffer_size: int = 0,
                 efficacy: float = -1.273 * 1e-12,
                 tau_decay: float = 0.02,
                 tau_rise: float = 0.0004,
                 max_firing_rate: float = 1.,
                 threshold: float = 11.,
                 steepness: float = 100.
                 ) -> None:
        """
        Instantiates current-based synapse with GABA_B receptor.
        """

        # define input transform attributes
        ###################################

        input_transform_args = {'threshold': threshold,
                                'steepness': steepness,
                                'max_firing_rate': max_firing_rate}

        # call super method
        ###################

        super().__init__(efficacy=efficacy,
                         buffer_size=buffer_size,
                         key='GABAB_DE',
                         tau_decay=tau_decay,
                         tau_rise=tau_rise
                         )

        # add input transform
        #####################

        self.input_transform = synaptic_sigmoid
        self.input_transform_args = input_transform_args

    def pass_input(self, synaptic_input: float, delay: int = 0
                   ) -> None:
        """Passes synaptic input to synaptic_input array.

        See Also
        --------
        :class:`Synapse`: ...for a detailed description of the method's parameters

        """

        # transform synaptic input
        # TODO: enable dependence of input transform on membrane potential of population
        synaptic_input = self.input_transform(synaptic_input, **self.input_transform_args)

        return super().pass_input(synaptic_input, delay)


class AMPAConductanceDESynapse(DEDoubleExponentialSynapse):
    """Defines a conductivity-based synapse with AMPA neuroreceptor.

    See Also
    --------
    :class:`DoubleExponentialSynapse`: Detailed documentation of parameters of double exponential synapse.
    :class:`Synapse`: Detailed documentation of synapse attributes and methods.

    """

    def __init__(self,
                 buffer_size: int = 0,
                 efficacy: float = 7.2e-10 * 1.273,
                 tau_decay: float = 1.0e-3,
                 tau_rise: float = 2e-4,
                 reversal_potential: float = 0.0
                 ) -> None:
        """
        Instantiates a conductance-based synapse with GABA_A receptor.
        """

        super().__init__(efficacy=efficacy,
                         tau_decay=tau_decay,
                         tau_rise=tau_rise,
                         buffer_size=buffer_size,
                         reversal_potential=reversal_potential,
                         key='AMPA_conductance')


class GABAAConductanceDESynapse(DEDoubleExponentialSynapse):
    """Conductivity-based synapse with GABA_A neuroreceptor.

    See Also
    --------
    :class:`DoubleExponentialSynapse`: Detailed documentation of parameters of double exponential synapse.
    :class:`Synapse`: Detailed documentation of synapse attributes and methods.

    """

    def __init__(self,
                 buffer_size: int = 0,
                 efficacy: float = 4e-11 * 1.358,
                 tau_decay: float = 6e-3,  # 0.02
                 tau_rise: float = 3e-4,
                 reversal_potential: float = -0.080
                 ) -> None:
        """
        Instantiates a current-based synapse with GABA_A receptor.
        """

        super().__init__(efficacy=efficacy,
                         tau_decay=tau_decay,
                         tau_rise=tau_rise,
                         buffer_size=buffer_size,
                         reversal_potential=reversal_potential,
                         key='GABAA_conductance')


###########################
# JansenRit-type synapses #
###########################


class JansenRitExcitatorySynapse(DEExponentialSynapse):
    """Excitatory second-order synapse as defined in [1]_.

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
                 step_size: float,
                 buffer_size: int = 0,
                 efficacy: float = 3.25e-3,
                 tau: float = 0.01,
                 tf_graph: Optional[tf.Graph] = None,
                 key: str = 'excitatory'
                 ) -> None:
        """Initializes excitatory exponential synapse as defined in [1]_.
        """

        super().__init__(step_size=step_size,
                         efficacy=efficacy,
                         tau=tau,
                         buffer_size=buffer_size,
                         tf_graph=tf_graph,
                         key=key)


class JansenRitInhibitorySynapse(DEExponentialSynapse):
    """Inhibitory second-order synapse as defined in [1]_.

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
                 step_size: float,
                 buffer_size: int = 0,
                 efficacy: float = -22e-3,
                 tau: float = 0.02,
                 tf_graph: Optional[tf.Graph] = None,
                 key: str = 'inhibitory'
                 ) -> None:
        """Initializes excitatory exponential synapse as defined in [1]_.
        """

        super().__init__(step_size=step_size,
                         efficacy=efficacy,
                         tau=tau,
                         buffer_size=buffer_size,
                         tf_graph=tf_graph,
                         key=key)


class MoranExcitatorySynapse(DEExponentialSynapse):
    """Excitatory second-order synapse as defined in [1]_.

    See Also
    --------
    :class:`ExponentialSynapse`: Detailed documentation of parameters of exponential synapse.
    :class:`Synapse`: Detailed documentation of synapse attributes and methods.

    References
    ----------
    .. [1] R.J. Moran, S.J. Kiebel, K.E. Stephan, R.B. Reilly, J. Daunizeau & K.J. Friston, "A Neural Mass Model of
       Spectral Responses in Electrophysiology" NeuroImage, vol. 37, pp. 706-720, 2007.

    """

    def __init__(self,
                 buffer_size: int = 0,
                 efficacy: float = 4e-3,
                 tau: float = 4e-3
                 ) -> None:
        """Initializes excitatory exponential synapse as defined in [1]_.
        """

        super().__init__(efficacy=efficacy,
                         tau=tau,
                         buffer_size=buffer_size,
                         key='Moran_excitatory')


class MoranInhibitorySynapse(DEExponentialSynapse):
    """Inhibitory second-order synapse as defined in [1]_.

    See Also
    --------
    :class:`ExponentialSynapse`: Detailed documentation of parameters of exponential synapse.
    :class:`Synapse`: Detailed documentation of synapse attributes and methods.

    References
    ----------
    .. [1] R.J. Moran, S.J. Kiebel, K.E. Stephan, R.B. Reilly, J. Daunizeau & K.J. Friston, "A Neural Mass Model of
       Spectral Responses in Electrophysiology" NeuroImage, vol. 37, pp. 706-720, 2007.

    """

    def __init__(self,
                 buffer_size: int = 0,
                 efficacy: float = -32e-3,
                 tau: float = 16e-3
                 ) -> None:
        """Initializes excitatory exponential synapse as defined in [1]_.
        """

        super().__init__(efficacy=efficacy,
                         tau=tau,
                         buffer_size=buffer_size,
                         key='Moran_inhibitory')
