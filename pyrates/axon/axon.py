"""Module that includes two base axon classes (one for instantaneous and one for bursting axons) plus derivations of it.

This module includes base axon classes and parametric sub-classes that can calculate average firing rates from
average membrane potentials. Its behavior approximates the average axon hillock of a homogeneous neural population and
thus represents its output behavior. The base axon classes allow for any kind of functional transform between average
membrane potential and average firing rate, while the sub-classes provide specific pre-defined transforms.
For a more detailed explanation, see the docstrings of the respective axon classes.

"""

# external packages
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from typing import Optional, Callable, overload, List
import tensorflow as tf

# pyrates internal imports
from pyrates.utility import parametric_sigmoid, normalized_sigmoid, plastic_sigmoid, plastic_normalized_sigmoid
from pyrates.utility.filestorage import RepresentationBase
from pyrates.parser import RHSParser

# meta infos
__author__ = "Richard Gast, Daniel F. Rose"
__status__ = "Development"


################
# generic axon #
################


class Axon(RepresentationBase):
    """Base axon class. Represents average behavior of generic axon hillock.

    Parameters
    ----------
    transfer_function
        Function used to transfer average membrane potentials into average firing rates. Takes membrane potentials
        [unit = V] as input and returns average firing rates [unit = 1/s].
    key
        Name of axon type.
    **transfer_function_kwargs
        Keyword arguments for the transfer function

    Attributes
    ----------
    transfer_function
        See `transfer_function` in parameter docstrings.
    key
        See `axon_type` in parameter docstrings.
    transfer_function_args
        See `transfer_function_kwargs` in parameter docstrings.
    firing_rate
        Current average firing rate at axon hillock [unit = 1/s].

    """

    def __init__(self,
                 transfer_function: str,
                 transfer_function_kwargs: Optional[dict] = None,
                 init_val: float = 0.,
                 key: Optional[str] = None,
                 tf_graph: Optional[tf.Graph] = None
                 ) -> None:
        """Instantiates base axon.
        """

        self.transfer_function = transfer_function
        self.key = '' if key is None else key
        self.transfer_function_args = transfer_function_kwargs
        self.tf_graph = tf.get_default_graph() if tf_graph is None else tf_graph

        with self.tf_graph.as_default():

            self.firing_rate = tf.get_variable(name=self.key + '_firing_rate',
                                               trainable=False,
                                               dtype=tf.float32,
                                               initializer=tf.constant_initializer(value=init_val),
                                               shape=()
                                               )
            self.membrane_potential = tf.get_variable(name=self.key + '_v',
                                                      trainable=False,
                                                      dtype=tf.float32,
                                                      initializer=tf.constant_initializer(value=init_val),
                                                      shape=()
                                                      )
            self.transfer_function_args['v'] = self.membrane_potential

            transfer_function_parser = RHSParser(self.transfer_function, self.transfer_function_args, self.tf_graph)
            tf_op, tf_op_args = transfer_function_parser.parse()

            for arg in tf_op_args:
                setattr(self, arg.name, arg)

            self.update_firing_rate = self.firing_rate.assign(tf_op)

    def clear(self):
        """Clears all time-dependent variables of axon.
        """

        return self.firing_rate.assign(0.)

    def update(self):
        """Updates all time-dependent parameters/methods of axon.
        """

        pass

    def plot_transfer_function(self,
                               membrane_potentials: np.ndarray,
                               create_plot: bool = True,
                               axis: Optional[Axes] = None
                               ) -> object:
        """Creates figure of the transfer function transforming membrane potentials into output firing rates.

        Parameters
        ----------
        membrane_potentials
            Membrane potential values for which to plot transfer function value [unit = V].
        create_plot
            If false, plot will bot be shown (default = True).
        axis
            If passed, plot will be created in respective figure axis (default = None).

        Returns
        -------
        axis
            Handle of the axis in which transfer function was plotted.

        """

        # calculate firing rates
        ########################

        firing_rates = []
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for v in membrane_potentials:
            sess.run(self.update_firing_rate(v))
            firing_rates.append(self.firing_rate.eval(session=sess))

        # plot firing rates over membrane potentials
        ############################################

        # check whether new figure needs to be created
        if axis is None:
            fig, axis = plt.subplots(num='Potential-To-Rate-Function')
        else:
            fig = axis.get_figure()

        # plot firing rates over membrane potentials
        axis.plot(membrane_potentials, firing_rates)

        # set figure labels
        axis.set_xlabel('membrane potential [V]')
        axis.set_ylabel('firing rate [Hz]')
        axis.set_title('Axon Hillock Transfer Function')

        # show plot
        if create_plot:
            fig.show()

        sess.close()

        return axis


###################
# sigmoidal axons #
###################


class SigmoidAxon(Axon):
    """Sigmoid axon class.

    Represents average firing behavior at axon hillock via a parametrized sigmoid as described in _[1].

    Parameters
    ----------
    max_firing_rate
        Determines maximum firing rate of axon [unit = 1/s].
    firing_threshold
        Determines membrane potential for which output firing rate is half the maximum firing rate [unit = V].
    slope
        Determines slope of the sigmoidal transfer function at the firing threshold [unit = 1/V].
    normalize
        If true, firing rate will be normalized to be zero at firing threshold.
    key
        See documentation of parameter `key` in :class:`Axon`

    See Also
    --------
    :class:`Axon`: documentation for a detailed description of the object attributes and methods.

    Notes
    -----

    Parametrized sigmoid:
    .. math:: r(t) = r_{max} (1 + exp(s (v_{thr} - v)))^{-1}

    References
    ----------
    .. [1] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical model
       of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.

    """

    def __init__(self,
                 max_firing_rate: float,
                 firing_threshold: float,
                 slope: float,
                 normalize: bool = False,
                 tf_graph: Optional[tf.Graph] = None,
                 key: Optional[str] = None
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
                         tf_graph=tf_graph,
                         key=key,
                         max_firing_rate=max_firing_rate,
                         firing_threshold=firing_threshold,
                         slope=slope)

    def plot_transfer_function(self,
                               membrane_potentials: Optional[np.ndarray] = None,
                               epsilon: float = 1e-4,
                               bin_size: float = 0.001,
                               create_plot: bool = True,
                               axis: Optional[Axes] = None
                               ) -> object:
        """Plots axon hillock sigmoidal transfer function.

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
        axis
            See method docstring of :class:`Axon`.

        Returns
        -------
        axis
            Handle of axis in which sigmoid was plotted.

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
            firing_rate = self.update_firing_rate(membrane_potential[-1])
            if firing_rate == 0.:
                raise ValueError('Automatic range detection not implemented for normalized axons yet. '
                                 'Please pass a membrane potential vector to plotting function.')
            while np.abs(firing_rate) > epsilon:
                membrane_potential.append(membrane_potential[-1] - bin_size)
                firing_rate = self.update_firing_rate(membrane_potential[-1])
            membrane_potential_1 = np.array(membrane_potential)
            membrane_potential_1 = np.flipud(membrane_potential_1)

            # start at membrane_potential_threshold and successively calculate the firing rate & increase the membrane
            # potential until the firing rate is greater or equal to max_firing_rate - epsilon
            membrane_potential = list()
            membrane_potential.append(self.transfer_function_args['membrane_potential_threshold'] + bin_size)
            firing_rate = self.update_firing_rate(membrane_potential[-1])
            while np.abs(firing_rate) <= self.transfer_function_args['max_firing_rate'] - epsilon:
                membrane_potential.append(membrane_potential[-1] + bin_size)
                firing_rate = self.update_firing_rate(membrane_potential[-1])
            membrane_potential_2 = np.array(membrane_potential)

            # concatenate the resulting membrane potentials
            membrane_potentials = np.concatenate((membrane_potential_1, membrane_potential_2))

        # call super method
        ###################

        axis = super().plot_transfer_function(membrane_potentials=membrane_potentials,
                                              create_plot=create_plot,
                                              axis=axis)

        return axis


class PlasticSigmoidAxon(Axon):
    """Sigmoid axon class. Represents average firing behavior at axon hillok via sigmoid as described by [1]_. with
    spike frequency adaptation enabled as described in [2]_.

    See Also
    --------
    :class:`Axon`: documentation for a detailed description of the object attributes and methods.
    :class:`SigmoidAxon`: documentation for a detailed description of all arguments needed for initialization.

    References
    ----------
    .. [1] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical model
       of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.
    .. [2] R.J. Moran, S.J. Kiebel, K.E. Stephan, R.B. Reilly, J. Daunizeau & K.J. Friston, "A Neural Mass Model of
       Spectral Responses in Electrophysiology" NeuroImage, vol. 37, pp. 706-720, 2007.

    """

    def __init__(self,
                 max_firing_rate: float,
                 firing_threshold: float,
                 slope: float,
                 normalize: bool = False,
                 key: Optional[str] = None
                 ) -> None:
        """Initializes sigmoid axon instance.
        """

        # check input parameters
        ########################

        if max_firing_rate < 0:
            raise ValueError('Maximum firing rate cannot be negative.')

        # call super init
        #################

        super().__init__(transfer_function=plastic_sigmoid if not normalize else plastic_normalized_sigmoid,
                         key=key,
                         max_firing_rate=max_firing_rate,
                         firing_threshold=firing_threshold,
                         slope=slope,
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
    key
        See description of parameter `key` of :class:`Axon`.
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

    """

    def __init__(self,
                 transfer_function: Callable[..., float],
                 kernel_functions: List[Callable[..., float]],
                 bin_size: float = 1e-3,
                 key: Optional[str] = None,
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
                         key=key,
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
                raise ValueError("The axonal kernel reached the break condition of 100s. This could either mean that "
                                 "your `kernel_function` does not decay to zero (fast enough) or your chosen `epsilon`"
                                 "error is too small. If you want to have an axon with a longer memory than "
                                 "100s, you have to specify a `max_delay` accordingly.")

        # flip time points array
        time_points = np.flip(np.array(time_points), axis=0)

        # noinspection PyTypeChecker
        return self.kernel_function(time_points, **self.kernel_function_args)

    @overload
    def evaluate_kernel(self,
                        time_points: float = 0.
                        ) -> float:
        ...

    @overload
    def evaluate_kernel(self,
                        time_points: np.ndarray = 0.
                        ) -> np.ndarray:
        ...

    def evaluate_kernel(self,
                        time_points=0.
                        ):
        """Computes axonal kernel value at specified time point(s).

        Parameters
        ----------
        time_points
            Time(s) at which to evaluate kernel. Only necessary if build_kernel is False (default = None) [unit = s].

        Returns
        -------
        np.ndarray, float
            Axonal kernel value at each t [unit = 1].

        """

        return self.kernel_function(time_points, **self.kernel_function_args)

    def update_firing_rate(self,
                           membrane_potential: float
                           ) -> float:
        """Computes average firing rate from membrane potential via transfer function and axonal kernel.

        Parameters
        ----------
        membrane_potential
            current average membrane potential of neural mass [unit = V].

        Returns
        -------
        float
            average firing rate at axon hillock [unit = 1/s]

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

        self.firing_rate = kernel_value * super().update_firing_rate(membrane_potential) * self.max_firing_rate

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
