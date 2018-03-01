"""Contains functions representing various biophysical features/mechanisms.
"""

import numpy as np
from typing import Union

__author__ = "Daniel F. Rose, Richard Gast"
__status__ = "Development"


########################
# activation functions #
########################


def parametric_sigmoid(membrane_potential: Union[float, np.ndarray],
                       max_firing_rate: float,
                       membrane_potential_threshold: float,
                       sigmoid_steepness: float
                       ) -> Union[float, np.ndarray]:
    """Sigmoidal axon hillok transfer function. Transforms membrane potentials into firing rates.

    Parameters
    ----------
    membrane_potential
        Membrane potential for which to calculate firing rate [unit = V].
    max_firing_rate
        See parameter description of `max_firing_rate` of :class:`SigmoidAxon`.
    membrane_potential_threshold
        See parameter description of `membrane_potential_threshold` of :class:`SigmoidAxon`.
    sigmoid_steepness
        See parameter description of `sigmoid_steepness` of :class:`SigmoidAxon`.

    Returns
    -------
    float
        average firing rate [unit = 1/s]

    """

    return max_firing_rate / (1 + np.exp(sigmoid_steepness * (membrane_potential_threshold - membrane_potential)))


def normalized_sigmoid(membrane_potential: Union[float, np.ndarray],
                       max_firing_rate: float,
                       membrane_potential_threshold: float,
                       sigmoid_steepness: float
                       ) -> Union[float, np.ndarray]:
    """Sigmoidal axon hillok transfer function. Transforms membrane potentials into firing rates.

    Parameters
    ----------
    membrane_potential
        Membrane potential for which to calculate firing rate [unit = V].
    max_firing_rate
        See parameter description of `max_firing_rate` of :class:`SigmoidAxon`.
    membrane_potential_threshold
        See parameter description of `membrane_potential_threshold` of :class:`SigmoidAxon`.
    sigmoid_steepness
        See parameter description of `sigmoid_steepness` of :class:`SigmoidAxon`.

    Returns
    -------
    float
        average firing rate [unit = 1/s]

    """

    return max_firing_rate / (1 + np.exp(sigmoid_steepness * (membrane_potential_threshold - membrane_potential))) - \
           max_firing_rate / (1 + np.exp(sigmoid_steepness * membrane_potential_threshold))


def plastic_sigmoid(membrane_potential,
                    max_firing_rate,
                    membrane_potential_threshold,
                    sigmoid_steepness,
                    adaptation):
    """

    Parameters
    ----------
    membrane_potential
        See above parameter description.
    max_firing_rate
        See above parameter description.
    membrane_potential_threshold
        See above parameter description.
    sigmoid_steepness
        See above parameter description.
    adaptation
        See above parameter description.

    Returns
    -------
    float
        firing rate [unit = 1/s]

    """
    return max_firing_rate / (1 + np.exp(sigmoid_steepness *
                                         (membrane_potential_threshold - membrane_potential + adaptation)))


def plastic_normalized_sigmoid(membrane_potential,
                               max_firing_rate,
                               membrane_potential_threshold,
                               sigmoid_steepness,
                               adaptation):
    """

    Parameters
    ----------
    membrane_potential
        See above parameter description.
    max_firing_rate
        See above parameter description.
    membrane_potential_threshold
        See above parameter description.
    sigmoid_steepness
        See above parameter description.
    adaptation
        See above parameter description.

    Returns
    -------
    float
        firing rate [unit = 1/s]

    """
    return max_firing_rate / (1 + np.exp(sigmoid_steepness *
                                         (membrane_potential_threshold - membrane_potential + adaptation))) - \
           max_firing_rate / (1 + np.exp(sigmoid_steepness * membrane_potential_threshold))


def activation_sigmoid(membrane_potential: Union[float, np.ndarray],
                       activation_threshold: float,
                       activation_steepness: float
                       ) -> Union[float, np.ndarray]:
    """Sigmoidal axon hillok transfer function. Transforms membrane potentials into firing rates.

    Parameters
    ----------
    membrane_potential
        Membrane potential for which to calculate firing rate [unit = V].
    activation_threshold
        See parameter description of `membrane_potential_threshold` of :class:`SigmoidAxon`.
    activation_steepness
        See parameter description of `sigmoid_steepness` of :class:`SigmoidAxon`.

    Returns
    -------
    float
        average firing rate [unit = 1/s]

    """

    return 1. / (1 + np.exp((membrane_potential - activation_threshold) / activation_steepness))


def inactivation_sigmoid(membrane_potential: Union[float, np.ndarray],
                         inactivation_threshold: float,
                         inactivation_steepness: float
                         ) -> Union[float, np.ndarray]:
    """Sigmoidal axon hillok transfer function. Transforms membrane potentials into firing rates.

    Parameters
    ----------
    membrane_potential
        Membrane potential for which to calculate firing rate [unit = V].
    inactivation_threshold
        See parameter description of `membrane_potential_threshold` of :class:`SigmoidAxon`.
    inactivation_steepness
        See parameter description of `sigmoid_steepness` of :class:`SigmoidAxon`.

    Returns
    -------
    float
        average firing rate [unit = 1/s]

    """

    return 1 / (1 + np.exp((membrane_potential - inactivation_threshold) / inactivation_steepness))


def synaptic_sigmoid(firing_rate: Union[float, np.ndarray],
                     threshold: float,
                     steepness: float
                     ) -> Union[float, np.ndarray]:
    """Sigmoidal axon hillok transfer function. Transforms membrane potentials into firing rates.

    Parameters
    ----------
    firing_rate
        Membrane potential for which to calculate firing rate [unit = V].
    threshold
        See parameter description of `membrane_potential_threshold` of :class:`SigmoidAxon`.
    steepness
        See parameter description of `sigmoid_steepness` of :class:`SigmoidAxon`.

    Returns
    -------
    float
        average firing rate [unit = 1/s]

    """

    return 1 / (1 + np.exp((firing_rate - threshold) / steepness))


###########
# kernels #
###########


def double_exponential(time_points: Union[float, np.ndarray],
                       tau_decay: float,
                       tau_rise: float
                       ) -> Union[float, np.ndarray]:
    """Uses double exponential function to calculate synaptic kernel value for each passed time-point.

    Parameters
    ----------
    time_points : Union[float, np.ndarray]
        Vector of time-points for which to calculate kernel value [unit = s].
    tau_decay
        See parameter documentation of `tau_decay` of :class:`DoubleExponentialSynapse`.
    tau_rise
        See parameter documentation of `tau_rise` of :class:`DoubleExponentialSynapse`.

    Returns
    -------
    Union[float, np.ndarray]
        Kernel values at the time-points [unit = S if conductivity based else A].

    """

    return np.exp(-time_points / tau_decay) - np.exp(-time_points / tau_rise)


def exponential(time_points: Union[float, np.ndarray],
                tau: float,
                ) -> Union[float, np.ndarray]:
    """Uses exponential function to calculate synaptic kernel value for each passed time-point.

    Parameters
    ----------
    time_points : Union[float, np.ndarray]
        Vector of time-points for which to calculate kernel value [unit = s].
    tau
        See parameter documentation of `tau` of :class:`ExponentialSynapse`.

    Returns
    -------
    Union[float, np.ndarray]
        Kernel values at the time-points [unit = S if conductivity based else A].

    """

    return time_points * np.exp(-time_points / tau) / tau


def axon_exponential(time_points: Union[float, np.ndarray],
                     tau_decay: float,
                     tau_rise: float
                     ) -> Union[float, np.ndarray]:
    """Uses double exponential function to calculate synaptic kernel value for each passed time-point.

    Parameters
    ----------
    time_points : Union[float, np.ndarray]
        Vector of time-points for which to calculate kernel value [unit = s].
    tau_decay
        See parameter documentation of `tau_decay` of :class:`DoubleExponentialSynapse`.
    tau_rise
        See parameter documentation of `tau_rise` of :class:`DoubleExponentialSynapse`.

    Returns
    -------
    Union[float, np.ndarray]
        Kernel values at the time-points [unit = S if conductivity based else A].

    """

    n1 = 1./tau_decay
    n2 = 1./tau_rise

    normalization = n1*n2 / (n2 - n1)

    return (np.exp(-n1 * time_points) - np.exp(-n2 * time_points)) * normalization

########################
# plasticity functions #
########################


def moran_spike_frequency_adaptation(adaptation: float,
                                     firing_rate: float,
                                     tau: float
                                     ) -> float:
    """Calculates adaption in sigmoid threshold of axonal transfer function.

    Parameters
    ----------
    adaptation
        Determines strength of spike-frequency-adaptation [unit = V].
    firing_rate
        Current firing rate towards which spike frequency is adapted [unit = 1/s].
    tau
        Time constant of adaptation process [unit = s].

    Returns
    -------
    float
        Change in threshold of sigmoidal axonal transfer function.

    """

    return (firing_rate - adaptation) / tau


def spike_frequency_adaptation(adaptation: float,
                               firing_rate: float,
                               firing_rate_target: float,
                               tau: float
                               ) -> float:
    """Calculates adaption in sigmoid threshold of axonal transfer function.

    Parameters
    ----------
    firing_rate
        Current firing rate of population [unit = 1/s].
    firing_rate_target
        Target firing rate towards which spike frequency is adapted [unit = 1/s].
    tau
        Time constant of adaptation process [unit = s].

    Returns
    -------
    float
        Change in threshold of sigmoidal axonal transfer function.

    """

    return (firing_rate - firing_rate_target) / tau


def synaptic_efficacy_adaptation(efficacy: float,
                                 firing_rate: float,
                                 max_firing_rate: float,
                                 tau_depression: float,
                                 tau_recycle: float
                                 ) -> float:
    """Calculates synaptic efficacy change.

    Parameters
    ----------
    efficacy
        Synaptic efficacy, See 'efficacy' parameter documentation of :class:`Synapse`.
    firing_rate
        Pre-synaptic firing rate [unit = 1/s].
    max_firing_rate
        Maximum pre-synaptic firing rate [unit = 1/s].
    tau_depression
        Defines synaptic depression time constant [unit = s].
    tau_recycle
        Defines synaptic recycling time constant [unit = s].

    Returns
    -------
    float
        Synaptic efficacy change.

    """

    depression_rate = (efficacy * firing_rate) / (max_firing_rate * tau_depression)
    recycle_rate = (1 - efficacy) / tau_recycle

    return recycle_rate - depression_rate if firing_rate > 0. else recycle_rate
