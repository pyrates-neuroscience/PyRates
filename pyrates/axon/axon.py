"""Module that includes basic axon class plus derivations of it.

This module includes a basic axon class and parametric sub-classes that can calculate average firing rates from
average membrane potentials. Its behavior approximates the average axon hillok of a homogeneous neural population.

"""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from typing import Optional, Callable, Union, overload, List
from pyrates.utility import parametric_sigmoid, normalized_sigmoid, plastic_sigmoid, plastic_normalized_sigmoid

from pyrates.utility.filestorage import RepresentationBase

__author__ = "Richard Gast, Daniel F. Rose"
__status__ = "Development"


################
# generic axon #
################


class Axon(RepresentationBase):
    """Base axon class. Represents average behavior of generic axon hillok.

    Parameters
    ----------
    transfer_function
        Function used to transfer average membrane potentials into average firing rates. Takes membrane potentials
        [unit = V] as input and returns average firing rates [unit = 1/s].
    axon_type
        Name of axon type.
    **transfer_function_args
        Keyword arguments for the transfer function

    Attributes
    ----------
    transfer_function
        See `transfer_function` in parameters section.
    axon_type
        See `axon_type` in parameters section.
    transfer_function_args
        Keyword arguments will be saved as dict on the object.

    """

    def __init__(self,
                 transfer_function: Callable[..., float],
                 axon_type: Optional[str] = None,
                 **transfer_function_args: float
                 ) -> None:
        """Instantiates base axon.
        """

        # set attributes
        ################

        self.transfer_function = transfer_function
        self.axon_type = 'custom' if axon_type is None else axon_type
        self.transfer_function_args = transfer_function_args
        self.firing_rate = 0.

    def compute_firing_rate(self, membrane_potential: float) -> float:
        """Computes average firing rate from membrane potential based on transfer function.

        Parameters
        ----------
        membrane_potential
            current average membrane potential of neural mass [unit = V].

        Returns
        -------
        float
            average firing rate at axon hillok [unit = 1/s]

        """

        # TODO: use functools.partial to convert transfer function to compute_firing_rate function?
        self.firing_rate = self.transfer_function(membrane_potential, **self.transfer_function_args)  # type: ignore

        return self.firing_rate

    def clear(self):

        self.firing_rate = 0.

    def update(self):
        pass

    def plot_transfer_function(self,
                               membrane_potentials: np.ndarray,
                               create_plot: bool = True,
                               axes: Optional[Axes] = None
                               ) -> object:
        """Creates figure of the transfer function transforming membrane potentials into output firing rates.

        Parameters
        ----------
        membrane_potentials
            Membrane potential values for which to plot transfer function value [unit = V].
        create_plot
            If false, plot will bot be shown (default = True).
        axes
            If passed, plot will be created in respective figure axis (default = None).

        Returns
        -------
        figure handle
            Handle of the newly created or updated figure.

        """

        # calculate firing rates
        ########################

        firing_rates = np.array([self.compute_firing_rate(v) for v in membrane_potentials])

        # plot firing rates over membrane potentials
        ############################################

        # check whether new figure needs to be created
        if axes is None:
            fig, axes = plt.subplots(num='Wave-To-Pulse-Function')
        else:
            fig = axes.get_figure()

        # plot firing rates over membrane potentials
        axes.plot(membrane_potentials, firing_rates)

        # set figure labels
        axes.set_xlabel('membrane potential [V]')
        axes.set_ylabel('firing rate [Hz]')
        axes.set_title('Axon Hillok Transfer Function')

        # show plot
        if create_plot:
            fig.show()

        return axes


###################
# sigmoidal axons #
###################


class SigmoidAxon(Axon):
    """Sigmoid axon class. Represents average firing behavior at axon hillok via sigmoid as described by [1]_.

    Parameters
    ----------
    max_firing_rate
        Determines maximum firing rate of axon [unit = 1/s].
    membrane_potential_threshold
        Determines membrane potential for which output firing rate is half the maximum firing rate [unit = V].
    sigmoid_steepness
        Determines steepness of the sigmoidal transfer function mapping membrane potential to firing rate
        [unit = 1/V].
    normalize
        If true, firing rate will be normalized to be zero at membrane potential threshold.
    axon_type
        See documentation of parameter `axon_type` in :class:`Axon`

    See Also
    --------
    :class:`Axon`: documentation for a detailed description of the object attributes and methods.

    References
    ----------
    .. [1] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical model
       of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.

    """

    def __init__(self,
                 max_firing_rate: float,
                 membrane_potential_threshold: float,
                 sigmoid_steepness: float,
                 normalize: bool = False,
                 axon_type: Optional[str] = None
                 ) -> None:
        """Initializes sigmoid axon instance.
        """

        # check input parameters
        ########################

        if max_firing_rate < 0:
            raise ValueError('Maximum firing rate cannot be negative.')

        # call super init
        #################

        super().__init__(transfer_function=parametric_sigmoid if not normalize else normalized_sigmoid,
                         axon_type=axon_type,
                         max_firing_rate=max_firing_rate,
                         membrane_potential_threshold=membrane_potential_threshold,
                         sigmoid_steepness=sigmoid_steepness)

    def plot_transfer_function(self,
                               membrane_potentials: Optional[np.ndarray] = None,
                               epsilon: float = 1e-4,
                               bin_size: float = 0.001,
                               create_plot: bool = True,
                               axes: Optional[Axes] = None):
        """Plots axon hillok sigmoidal transfer function.

        Parameters
        ----------
        membrane_potentials
            See method docstring of :class:`Axon`.
        epsilon
            Determines min/max function value to plot, if membrane_potentials is None.
            Min = 0 + epsilon, max = max_firing_rate - epsilon [unit = 1/s] (default = 1e-4).
        bin_size
            Determines size of steps between membrane potentials at which function is evaluated.
            [unit = V] (default = 0.001).
        create_plot
            See method docstring of :class:`Axon`.
        axes
            See method docstring of :class:`Axon`.


        See Also
        --------
        :class:`Axon`: See description of `plot_transfer_function` method for a detailed input/output description.

        Notes
        -----
        Membrane potential is now an optional argument compared to base axon plotting function.

        """

        # check input parameters
        ########################

        if epsilon < 0:
            raise ValueError('Epsilon cannot be negative. See parameter description for further information.')

        if bin_size < 0:
            raise ValueError('Bin size cannot be negative. See parameter description for further information.')

        # calculate firing rates
        ########################

        if membrane_potentials is None:

            # start at membrane_potential_threshold and successively calculate the firing rate & reduce the membrane
            # potential until the firing rate is smaller or equal to epsilon
            membrane_potential = list()
            membrane_potential.append(self.transfer_function_args['membrane_potential_threshold'])
            firing_rate = self.compute_firing_rate(membrane_potential[-1])
            if firing_rate == 0.:
                raise ValueError('Automatic range detection not implemented for normalized axons yet. '
                                 'Please pass a membrane potential vector to plotting function.')
            while np.abs(firing_rate) > epsilon:
                membrane_potential.append(membrane_potential[-1] - bin_size)
                firing_rate = self.compute_firing_rate(membrane_potential[-1])
            membrane_potential_1 = np.array(membrane_potential)
            membrane_potential_1 = np.flipud(membrane_potential_1)

            # start at membrane_potential_threshold and successively calculate the firing rate & increase the membrane
            # potential until the firing rate is greater or equal to max_firing_rate - epsilon
            membrane_potential = list()
            membrane_potential.append(self.transfer_function_args['membrane_potential_threshold'] + bin_size)
            firing_rate = self.compute_firing_rate(membrane_potential[-1])
            while np.abs(firing_rate) <= self.transfer_function_args['max_firing_rate'] - epsilon:
                membrane_potential.append(membrane_potential[-1] + bin_size)
                firing_rate = self.compute_firing_rate(membrane_potential[-1])
            membrane_potential_2 = np.array(membrane_potential)

            # concatenate the resulting membrane potentials
            membrane_potentials = np.concatenate((membrane_potential_1, membrane_potential_2))

        # call super method
        ###################

        axes = super().plot_transfer_function(membrane_potentials=membrane_potentials,
                                              create_plot=create_plot,
                                              axes=axes)

        return axes


class PlasticSigmoidAxon(Axon):
    """Sigmoid axon class. Represents average firing behavior at axon hillok via sigmoid as described by [1]_. with
    spike frequency adaptation enabled as described in [2]_.

    Parameters
    ----------
    max_firing_rate
        Determines maximum firing rate of axon [unit = 1/s].
    membrane_potential_threshold
        Determines membrane potential for which output firing rate is half the maximum firing rate [unit = V].
    sigmoid_steepness
        Determines steepness of the sigmoidal transfer function mapping membrane potential to firing rate
        [unit = 1/V].
    normalize
        If true, firing rate will be normalized to be zero at membrane potential threshold.
    axon_type
        See documentation of parameter `axon_type` in :class:`Axon`

    See Also
    --------
    :class:`Axon`: documentation for a detailed description of the object attributes and methods.

    References
    ----------
    .. [1] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical model
       of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.
    .. [2] R.J. Moran, S.J. Kiebel, K.E. Stephan, R.B. Reilly, J. Daunizeau & K.J. Friston, "A Neural Mass Model of
       Spectral Responses in Electrophysiology" NeuroImage, vol. 37, pp. 706-720, 2007.

    """

    def __init__(self,
                 max_firing_rate: float,
                 membrane_potential_threshold: float,
                 sigmoid_steepness: float,
                 normalize: bool = False,
                 axon_type: Optional[str] = None
                 ) -> None:
        """Initializes sigmoid axon instance.
        """

        # check input parameters
        ########################

        if max_firing_rate < 0:
            raise ValueError('Maximum firing rate cannot be negative.')

        if sigmoid_steepness < 0:
            raise ValueError('Sigmoid steepness cannot be negative.')

        # call super init
        #################

        super().__init__(transfer_function=plastic_sigmoid if not normalize else plastic_normalized_sigmoid,
                         axon_type=axon_type,
                         max_firing_rate=max_firing_rate,
                         membrane_potential_threshold=membrane_potential_threshold,
                         sigmoid_steepness=sigmoid_steepness,
                         adaptation=0.)


##################
# bursting axons #
##################


class BurstingAxon(Axon):
    """Axon that features bursting behavior initialized by low-threshold spikes.
    
    Parameters
    ----------
    transfer_function
        See description of parameter `transfer_function` of :class:`Axon`.
    kernel_functions
        List of two functions, with the first one being the kernel function and the second the non-linearity applied
        to the membrane potential before convolving it with the kernel.
    bin_size
        time window one bin of the axonal kernel is representing [unit = s].
    axon_type
        See description of parameter `axon_type` of :class:`Axon`.
    max_delay
        Maximal time delay after which a certain membrane potential still affects the firing rate [unit = s].
    max_firing_rate
        Maximum firing rate of axon [unit = 1/s] (default = 800.).
    epsilon
        Accuracy of the synaptic kernel representation.
    resting_potential
        Resting membrane potential of the population the axon belongs to [unit = V].
    kernel_function_args
        List of parameter dictionaries (name-value pairs) for each of the kernel functions.
    **transfer_function_args
        See description of parameter `transfer_function_args` of :class:`Axon`.
    
    See Also
    --------
    :class:`Axon`: ... for a detailed description of the object attributes and methods.
    
    References
    ----------
    """

    def __init__(self,
                 transfer_function: Callable[..., float],
                 kernel_functions: List[Callable[..., float]],
                 bin_size: float = 1e-3,
                 axon_type: Optional[str] = None,
                 epsilon: float = 1e-10,
                 max_delay: Optional[float] = None,
                 max_firing_rate: float = 800.,
                 resting_potential: float = -0.075,
                 kernel_function_args: Optional[List[dict]] = None,
                 **transfer_function_args: float
                 ) -> None:
        """Instantiate bursting axon.
        """

        # check input parameters
        ########################

        if bin_size <= 0 or (max_delay is not None and max_delay <= 0):
            raise ValueError('Time constants (bin_size, max_delay) cannot be negative or zero. '
                             'See docstring for further information.')

        if epsilon <= 0:
            raise ValueError('Epsilon is an absolute error term that cannot be negative or zero.')

        if (len(kernel_functions) != len(kernel_function_args)) or len(kernel_functions) > 2:
            raise ValueError('Number of kernel functions can be either 1 or 2 and the number of provided argument'
                             'dictionaries has to be equal to the number of functions passed.')

        # set attributes
        ################

        self.bin_size = bin_size
        self.epsilon = epsilon
        self.max_delay = max_delay
        self.max_firing_rate = max_firing_rate
        self.resting_potential = resting_potential
        self.kernel_function = kernel_functions[0]
        self.kernel_function_args = kernel_function_args[0] if kernel_function_args else dict()
        self.kernel_nonlinearity = kernel_functions[1] if len(kernel_functions) > 1 else lambda x: x
        self.kernel_nonlinearity_args = kernel_function_args[1] if len(kernel_functions) > 1 else dict()

        # call super init
        #################

        super().__init__(transfer_function=transfer_function,
                         axon_type=axon_type,
                         **transfer_function_args)

        # build axonal kernel
        #####################

        self.axon_kernel = self.build_kernel()
        self.kernel_length = len(self.axon_kernel)

        # initialize membrane potential buffer
        ######################################

        self.membrane_potentials = np.zeros(len(self.axon_kernel)) + self.resting_potential

    def build_kernel(self) -> np.ndarray:
        """Builds axonal kernel.
        
        Returns
        -------
        np.ndarray
            Axonal kernel value at each t [unit = Hz]
            
        """

        # check whether to build kernel or just evaluate it at time_points
        ##################################################################

        if self.max_delay:

            # create time vector from max_delay
            time_points = np.arange(0., self.max_delay + 0.5 * self.bin_size, self.bin_size)

        else:

            # create time vector from epsilon
            time_points = [0.]
            kernel_val = self.evaluate_kernel(time_points[-1])

            while time_points[-1] < 100:  # cuts off at a maximum delay of 100 s (which is already too long)

                # calculate kernel value for next time-step
                time_points.append(time_points[-1] + self.bin_size)
                kernel_val_tmp = self.evaluate_kernel(time_points[-1])

                # check whether kernel value is decaying towards zero
                if (kernel_val_tmp - kernel_val) < 0.:

                    # if yes, check whether kernel value is already smaller epsilon
                    if abs(kernel_val_tmp) < self.epsilon:
                        break

                kernel_val = kernel_val_tmp
            else:
                raise ValueError("The synaptic kernel reached the break condition of 100s. This could either mean that "
                                 "your `kernel_function` does not decay to zero (fast enough) or your chosen `epsilon`"
                                 "error is too small. If you want to have a synapse with a longer synaptic memory than "
                                 "100s, you have to specify a `max_delay` accordingly.")

        # flip time points array
        time_points = np.flip(np.array(time_points), axis=0)

        # noinspection PyTypeChecker
        return self.kernel_function(time_points, **self.kernel_function_args)

    @overload
    def evaluate_kernel(self, time_points: float = 0.) -> float:
        ...

    @overload
    def evaluate_kernel(self, time_points: np.ndarray = 0.) -> np.ndarray:
        ...

    def evaluate_kernel(self, time_points=0.):
        """Builds axonal kernel or computes value of it at specified time point(s).

        Parameters
        ----------
        time_points
            Time(s) at which to evaluate kernel. Only necessary if build_kernel is False (default = None) [unit = s].

        Returns
        -------
        np.ndarray, float
            Synaptic kernel value at each t [unit = A or S]
        """

        return self.kernel_function(time_points, **self.kernel_function_args)

    def compute_firing_rate(self, membrane_potential: float) -> float:
        """Computes average firing rate from membrane potential based on transfer function and axonal kernel.

        Parameters
        ----------
        membrane_potential
            current average membrane potential of neural mass [unit = V].

        Returns
        -------
        float
            average firing rate at axon hillok [unit = 1/s]

        """

        # update membrane potential buffer
        ##################################

        self.membrane_potentials[0:-1] = self.membrane_potentials[1:]
        self.membrane_potentials[-1] = membrane_potential

        # apply synaptic kernel
        #######################

        # multiply membrane potentials with kernel
        kernel_value = self.kernel_nonlinearity(self.membrane_potentials,
                                                **self.kernel_nonlinearity_args) * self.axon_kernel

        # integrate over time
        kernel_value = np.trapz(kernel_value, dx=self.bin_size)

        self.firing_rate = kernel_value * super().compute_firing_rate(membrane_potential) * self.max_firing_rate

        return self.firing_rate

    def clear(self):
        """Function that clears membrane potential inputs.
        """

        super().clear()
        self.membrane_potentials = np.zeros(self.kernel_length) + self.resting_potential

    def update(self):
        """Updates axon attributes.
        """

        # update kernel
        self.axon_kernel = self.build_kernel()
        self.kernel_length = len(self.axon_kernel)

        # update buffer
        # TODO: implement interpolation from old to new array
        self.membrane_potentials = np.zeros(len(self.axon_kernel)) + self.resting_potential
