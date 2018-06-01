"""Module that includes basic synapse classes plus derivations of it.

This module includes a two generic synapse classes that can transform incoming average firing rates into average
synaptic responses (currents or potentials). For this purpose, both classes perform a convolution of a synaptic
response kernel with the synaptic input, but the Synapse class solves the convolution numerically, while the DESynapse
class does it analytically (based on a second-order differential equation).

Several less generic sub-classes are included as well.

For more information, see the docstrings of the respective classes.

"""

# external packages
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union, Callable, overload
from matplotlib.axes import Axes
import tensorflow as tf

# pyrates internal imports
from pyrates.utility.filestorage import RepresentationBase
from pyrates.utility import exponential, double_exponential

# type settings
FloatOrArray = Union[float, np.ndarray]

# meta infos
__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


####################
# generic synapses #
####################


class Synapse(RepresentationBase):
    """Basic synapse class. Represents average behavior of a certain post-synapse type of a population.

    Parameters
    ----------
    kernel_function
        Function that specifies kernel value given the time-point after arrival of synaptic input. Output will be either
        a synaptic current or conductance change [unit = S or A respectively].
    efficacy
        Determines strength and direction of the synaptic response to input [unit = S if synapse is modulatory else A].
    bin_size
        Size of the time-steps between successive bins of the synaptic kernel [unit = s].
    epsilon
        Minimum kernel value. Kernel values below epsilon will be set to zero [unit = S or A] (default = 1e-14).
    max_delay
        Maximum time after which incoming synaptic input still affects the synapse [unit = s] (default = None). If set,
        epsilon will be ignored.
    buffer_size
        Maximum time that information passed to the synapse needs to affect the synapse [unit = s] (default = 0.).
    reversal_potential
        Synaptic reversal potential. If passed, synapse will be conductivity based (default = None) [unit = V].
    key
        Name of synapse type (default = None).
    **kernel_function_kwargs
        Keyword arguments for the kernel function

    Attributes
    ----------
    synaptic_kernel : np.ndarray
        Vector including the synaptic kernel value at each time-bin [unit = S if conductivity based else A].
    kernel_function: Callable
        See parameter description.
    kernel_function_args: dict
        Keyword arguments will be saved as dict on the object.
    efficacy: float
        Determines strength and direction of the synaptic response to input [unit = S if synapse is modulatory else A].
    reversal_potential: float
        See parameter description.
    bin_size: float
        See parameter description.
    max_delay: float
        See parameter description.
    key: str
        See parameter description.

    """

    def __init__(self,
                 kernel_function: Callable[..., FloatOrArray],
                 efficacy: float,
                 bin_size: float,
                 epsilon: float = 1e-14,
                 max_delay: Optional[float] = None,
                 buffer_size: float = 0.,
                 reversal_potential: Optional[float] = None,
                 key: Optional[str] = None,
                 **kernel_function_kwargs: float
                 ) -> None:
        """Instantiates base synapse.
        """

        # check input parameters
        ########################

        if bin_size <= 0 or (max_delay is not None and max_delay <= 0):
            raise ValueError('Time constants (bin_size, max_delay) cannot be negative or zero. '
                             'See docstring for further information.')

        if epsilon <= 0:
            raise ValueError('Epsilon is an absolute error term that cannot be negative or zero.')

        # set attributes
        ################

        self.efficacy = efficacy
        self.reversal_potential = reversal_potential
        self.bin_size = bin_size
        self.epsilon = epsilon
        self.max_delay = max_delay
        self.kernel_function = kernel_function
        self.kernel_function_args = kernel_function_kwargs
        self.buffer_size = buffer_size

        # set synapse type
        ##################

        if key is None:

            # define synapse type via synaptic kernel efficacy and type (current- vs conductivity-based)
            if reversal_potential:
                self.reversal_potential = reversal_potential
                self.key = 'excitatory_conductance' if efficacy >= 0 else 'inhibitory_conductance'
            else:
                self.key = 'excitatory_current' if efficacy >= 0 else 'inhibitory_current'

        else:

            self.key = key

        # set synaptic depression (for plasticity mechanisms)
        self.depression = 1.0

        # set decorator for synaptic current getter (only relevant for conductivity based synapses)
        if reversal_potential:
            self.synaptic_response_scaling = lambda membrane_potential: (self.reversal_potential - membrane_potential)
        else:
            self.synaptic_response_scaling = lambda membrane_potential: 1.0

        # build synaptic kernel
        #######################

        self.synaptic_kernel = self.build_kernel()

        # build input buffer
        ####################

        self.synaptic_input = np.zeros(int(self.buffer_size / self.bin_size) + len(self.synaptic_kernel))
        self.kernel_length = len(self.synaptic_kernel)

    def build_kernel(self) -> np.ndarray:
        """Builds synaptic kernel.

        Returns
        -------
        np.ndarray
            Synaptic kernel value at each t [unit = A or S]

        """

        # check whether to build kernel or just evaluate it at time_points
        ##################################################################

        if self.max_delay:

            # create time vector from max_delay
            time_points = np.arange(0., self.max_delay + 0.5*self.bin_size, self.bin_size)

        else:

            # create time vector from epsilon
            time_points = [0.]
            kernel_val = self.evaluate_kernel(time_points[-1])

            while time_points[-1] < 100:  # cuts off at a maximum delay of 100 s (which is already too long)

                # calculate kernel value for next time-step
                time_points.append(time_points[-1] + self.bin_size)
                kernel_val_tmp = self.evaluate_kernel(time_points[-1])

                # check whether kernel value is decaying towards zero
                if ((kernel_val_tmp - kernel_val < 0.) and (self.efficacy > 0.)) or \
                        ((kernel_val - kernel_val_tmp < 0.) and (self.efficacy < 0.)):

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
        return self.evaluate_kernel(time_points)

    @overload
    def evaluate_kernel(self,
                        time_points: float = 0.
                        ) -> float: ...

    @overload
    def evaluate_kernel(self,
                        time_points: np.ndarray = 0.
                        ) -> np.ndarray: ...

    def evaluate_kernel(self,
                        time_points=0.
                        ):
        """Computes synaptic kernel or computes value of it at specified time point(s).

        Parameters
        ----------
        time_points
            Time(s) at which to evaluate kernel. Only necessary if build_kernel is False (default = None) [unit = s].

        Returns
        -------
        np.ndarray, float
            Synaptic kernel value at each t [unit = A or S]
        """

        return self.kernel_function(time_points, **self.kernel_function_args) * self.efficacy

    def get_synaptic_response(self,
                              membrane_potential: Union[float, np.float64] = -0.075
                              ) -> Union[np.float64, float]:
        """Applies synaptic kernel to synaptic input (should be incoming firing rate).

        Parameters
        ----------
        membrane_potential
            Membrane potential of post-synapse. Only to be used for conductivity based synapses (default = None)
            [unit = V].

        Returns
        -------
        float
            Resulting synaptic response [unit = A or S or V].

        """

        # apply synaptic kernel
        #######################

        # multiply firing rate input with kernel
        # noinspection PyTypeChecker
        kernel_value = self.synaptic_input[0:self.kernel_length] * self.synaptic_kernel

        # integrate over time
        kernel_value = np.trapz(kernel_value, dx=self.bin_size)

        return kernel_value * self.synaptic_response_scaling(membrane_potential) * self.depression

    def pass_input(self,
                   synaptic_input: float,
                   delay: int = 0
                   ) -> None:
        """Passes synaptic input to synaptic_input array.
        
        Parameters
        ----------
        synaptic_input
            Incoming firing rate input [unit = 1/s].
        delay
            Number of time steps after which the input will arrive at synapse [unit = 1].
            
        """

        self.synaptic_input[self.kernel_length + delay - 1] += synaptic_input

    def rotate_input(self):
        """Shifts input values in synaptic input vector one position to the left.
        """

        self.synaptic_input[0:-1] = self.synaptic_input[1:]
        self.synaptic_input[-1] = 0.

    def clear(self):
        """Clears synaptic input and depression.
        """

        self.synaptic_input[:] = 0.
        self.depression = 1.0

    def update(self):
        """Updates synapse attributes.
        """

        # update kernel
        self.synaptic_kernel = self.build_kernel()

        # update buffer
        # TODO: implement interpolation from old to new array
        self.synaptic_input = np.zeros(int(self.buffer_size / self.bin_size) + len(self.synaptic_kernel))
        self.kernel_length = len(self.synaptic_kernel)

        # set decorator for synaptic current getter (only relevant for conductivity based synapses)
        if self.reversal_potential:
            self.synaptic_response_scaling = lambda membrane_potential: (self.reversal_potential - membrane_potential)
        else:
            self.synaptic_response_scaling = lambda membrane_potential: 1.0

    def plot_synaptic_kernel(self,
                             create_plot: bool = True,
                             axis: Optional[Axes] = None
                             ) -> object:
        """Creates plot of synaptic kernel over time.

        Parameters
        ----------
        create_plot
            If false, no plot will be shown (default = True).
        axis
            figure handle, can be passed optionally (default = None).

        Returns
        -------
        :obj:`figure handle`
            Handle of the newly created or updated figure.

        """

        # check whether new figure has to be created
        ############################################

        if axis is None:
            fig, axis = plt.subplots(num='Impulse Response Function')
        else:
            fig = axis.get_figure()

        # plot synaptic kernel
        ######################

        axis.plot(self.synaptic_kernel[-1:0:-1] * self.depression)

        # set figure labels
        axis.set_xlabel('time-steps')
        if self.reversal_potential:
            axis.set_ylabel('synaptic conductivity [S]')
        else:
            axis.set_ylabel('synaptic current [A]')
        axis.set_title('Synaptic Kernel')

        # show plot
        if create_plot:
            fig.show()

        return axis


class DESynapse(RepresentationBase):
    """Differential equation version of synapse class. Represents average behavior of a defined post-synapse
     of a population.

    Parameters
    ----------
    efficacy
        Determines strength and direction of the synaptic response to input [unit = S if synapse is modulatory else A].
    buffer_size
        Maximum time that information passed to the synapse needs to affect the synapse [unit = s] (default = 0.).
    reversal_potential
        Synaptic reversal potential. If passed synapse will be conductance based (default = None) [unit = V].
    key
        Name of synapse type (default = None).

    Attributes
    ----------
    efficacy
        Determines strength and direction of the synaptic response to input [unit = S if synapse is modulatory else A].
    reversal_potential
        See parameter description.
    key
        See parameter description.

    """

    def __init__(self,
                 step_size: float,
                 efficacy: float,
                 buffer_size: float = 0.,
                 reversal_potential: Optional[float] = None,
                 tf_graph: Optional[tf.Graph] = None,
                 key: Optional[str] = None
                 ) -> None:
        """Instantiates base synapse.
        """

        # set synapse type
        ##################

        if key is None:

            # define synapse type via synaptic kernel efficacy and type (current- vs conductivity-based)
            if reversal_potential:
                self.key = 'excitatory_conductance' if efficacy >= 0 else 'inhibitory_conductance'
            else:
                self.key = 'excitatory_current' if efficacy >= 0 else 'inhibitory_current'

        else:

            self.key = key

        # build tensorflow graph
        ########################

        self.tf_graph = tf.get_default_graph() if tf_graph is None else tf_graph

        with self.tf_graph.as_default():

            # set attributes
            ################

            self.step_size = step_size
            self.efficacy = efficacy
            self.reversal_potential = reversal_potential
            self.buffer_size = buffer_size

            # set synaptic depression (for plasticity mechanisms)
            self.depression = 1.0

            # set decorator for synaptic current getter (only relevant for conductivity based synapses)
            if reversal_potential:
                self.synaptic_response_scaling = lambda membrane_potential: (self.reversal_potential-membrane_potential)
            else:
                self.synaptic_response_scaling = lambda membrane_potential: 1.0

            # build input buffer
            ####################

            self.synaptic_buffer = tf.Variable(np.zeros(1 + int(self.buffer_size / self.step_size), dtype=np.float64),
                                               trainable=False,
                                               name=self.key + '_synaptic_buffer')

            # initialize state variables
            ############################

            self.synaptic_response = tf.get_variable(name=self.key + '_synaptic_response',
                                                     dtype=tf.float64,
                                                     trainable=False,
                                                     initializer=tf.constant_initializer(value=0.),
                                                     shape=()
                                                     )

            self.psp = tf.get_variable(name=self.key + '_psp',
                                       dtype=tf.float64,
                                       trainable=False,
                                       initializer=tf.constant_initializer(value=0.),
                                       shape=()
                                       )

    def get_synaptic_response(self) -> tf.Variable:
        """Calculates change in synaptic response from synaptic input (should be incoming firing rate).

        Returns
        -------
        tf.Variable
            Resulting synaptic response [unit = A].

        """

        raise AttributeError('This method needs to be implemented in child classes for the synapse to work!')

    def pass_input(self, synaptic_input: float, delay: int = 0
                   ) -> None:
        """Passes synaptic input to synaptic_input array.

        Parameters
        ----------
        synaptic_input
            Incoming firing rate input [unit = 1/s].
        delay
            Number of time steps after which the input will arrive at synapse [unit = 1].

        """

        return self.synaptic_buffer[delay].assign(self.synaptic_buffer[delay] + synaptic_input)

    def clear(self):
        """Clears synaptic input and depression.
        """

        self.depression = 1.0

        return self.synaptic_buffer[:].assign(0.)

    def update(self):
        """Updates synapse attributes.
        """

        with self.tf_graph.as_default():

            # update buffer
            # TODO: implement interpolation from old to new array
            self.synaptic_input = tf.placeholder(shape=(),
                                                 dtype=float,
                                                 name=self.key + '_synaptic_input')
            self.synaptic_buffer = tf.Variable(np.zeros(1 + int(self.buffer_size / self.step_size)),
                                               trainable=False,
                                               name=self.key + '_synaptic_buffer')

            # set decorator for synaptic current getter (only relevant for conductivity based synapses)
            if self.reversal_potential:
                self.synaptic_response_scaling = lambda membrane_potential: \
                    (self.reversal_potential - membrane_potential)
            else:
                self.synaptic_response_scaling = lambda membrane_potential: 1.0


##############################
# double exponential synapse #
##############################


class DoubleExponentialSynapse(Synapse):
    """Basic synapse class. Represents average behavior of a defined post-synapse of a population.

    Parameters
    ----------
    efficacy
        See documentation of parameter `efficacy` in :class:`Synapse`.
    tau_decay
        Lumped time delay constant that determines how fast the exponential synaptic kernel decays [unit = s].
    tau_rise
        Lumped time delay constant that determines how fast the exponential synaptic kernel rises [unit = s].
    bin_size
        See documentation of parameter `bin_size` in :class:`Synapse`.
    max_delay
        See documentation of parameter `max_delay` in :class:`Synapse`.
    buffer_size
        See documentation of parameter `buffer_size` in :class:`Synapse`.
    reversal_potential
        See documentation of parameter `reversal_potential` in :class:`Synapse`.
    key
        Name of synapse type (default = None).

    See Also
    --------
    :class:`Synapse`: documentation for a detailed description of the object attributes and methods.
                              
    """

    def __init__(self,
                 efficacy: float,
                 tau_decay: float,
                 tau_rise: float,
                 bin_size: float,
                 epsilon: float = 1e-14,
                 max_delay: Optional[float] = None,
                 buffer_size: float = 0.,
                 reversal_potential: Optional[float] = None,
                 key: Optional[str] = None
                 ) -> None:

        # check input parameters
        ########################

        if tau_decay < 0 or tau_rise < 0:
            raise ValueError('Time constants tau cannot be negative. See docstring for further information.')

        # call super init
        #################

        super().__init__(kernel_function=double_exponential,
                         efficacy=efficacy,
                         bin_size=bin_size,
                         epsilon=epsilon,
                         max_delay=max_delay,
                         buffer_size=buffer_size,
                         reversal_potential=reversal_potential,
                         key=key,
                         tau_rise=tau_rise,
                         tau_decay=tau_decay)


class DEDoubleExponentialSynapse(DESynapse):
    """Differential equation version of bi-exponential synapse class. Represents average behavior of a defined 
    post-synapse of a population.

    Parameters
    ----------
    efficacy
        See documentation of parameter `efficacy` in :class:`Synapse`.
    tau_decay
        Lumped time delay constant that determines how fast the exponential synaptic kernel decays [unit = s].
    tau_rise
        Lumped time delay constant that determines how fast the exponential synaptic kernel rises [unit = s].
    buffer_size
        See documentation of parameter `buffer_size` in :class:`Synapse`.
    reversal_potential
        See documentation of parameter `reversal_potential` in :class:`Synapse`.
    key
        Name of synapse type (default = None).

    See Also
    --------
    :class:`Synapse`: documentation for a detailed description of the object attributes and methods.

    """

    def __init__(self,
                 efficacy: float,
                 tau_decay: float,
                 tau_rise: float,
                 buffer_size: int = 0,
                 reversal_potential: Optional[float] = None,
                 key: Optional[str] = None
                 ) -> None:
        """Instantiates base synapse.
        """

        # call super init
        #################

        super().__init__(efficacy=efficacy,
                         buffer_size=buffer_size,
                         reversal_potential=reversal_potential,
                         key=key)

        # set additional attributes
        ###########################

        # time constant
        self.tau_decay = tau_decay
        self.tau_rise = tau_rise

        # pre-calculate DE constants
        self.input_scaling = self.efficacy / (self.tau_decay - self.tau_rise)
        self.d1_scaling = 1 / (self.tau_decay**2 - self.tau_rise**2)
        self.d2_scaling = 2 / (self.tau_decay - self.tau_rise)

    def get_synaptic_response(self,
                              synaptic_response_old: Union[float, np.float64],
                              membrane_potential: Union[float, np.float64]
                              ) -> Union[np.float64, float]:
        """Calculates change in synaptic current from synaptic input (should be incoming firing rate).

        Parameters
        ----------
        synaptic_response_old
            Synaptic current from last time-step [unit = A].
        membrane_potential
            Membrane potential of post-synapse. Only to be used for conductivity based synapses (default = None)
            [unit = V].

        Returns
        -------
        float
            Resulting synaptic current [unit = A].

        """

        # calculate delta current
        #########################

        delta_current = self.input_scaling * self.synaptic_input[self.kernel_length - 1] - \
                        self.d1_scaling * membrane_potential - self.d2_scaling * synaptic_response_old

        # update synaptic input buffer
        ##############################

        self.synaptic_input[0:-1] = self.synaptic_input[1:]
        self.synaptic_input[-1] = 0.

        return delta_current * self.synaptic_response_scaling(membrane_potential) * self.depression

    def update(self):
        """Updates attributes depending on current synapse state.
        """

        # call super method
        ###################

        super().update()

        # re-calculate DE constants
        ###########################

        self.input_scaling = self.efficacy / (self.tau_decay - self.tau_rise)
        self.d1_scaling = 1 / (self.tau_decay ** 2 - self.tau_rise ** 2)
        self.d2_scaling = 2 / (self.tau_decay - self.tau_rise)

        # set decorator for synaptic current getter (only relevant for conductivity based synapses)
        if self.reversal_potential:
            self.synaptic_response_scaling = lambda membrane_potential: (self.reversal_potential - membrane_potential)
        else:
            self.synaptic_response_scaling = lambda membrane_potential: 1.0


#######################
# exponential synapse #
#######################


class ExponentialSynapse(Synapse):
    """Basic synapse class. Represents average behavior of a defined post-synapse of a population. Follows definition of
    [1]_.

    Parameters
    ----------
    efficacy
        See documentation of parameter `efficacy` in :class:`Synapse`.
    tau
        Lumped time delay constant that determines the shape of the exponential synaptic kernel [unit = s].
    bin_size
        See documentation of parameter `bin_size` in :class:`Synapse`.
    max_delay
        See documentation of parameter `max_delay` in :class:`Synapse`.
    buffer_size
        See documentation of parameter `buffer_size` in :class:`Synapse`.
    key
        Name of synapse type (default = None).

    See Also
    --------
    :class:`Synapse`: documentation for a detailed description of the object attributes and methods.

    References
    ----------
    .. [1] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical model
       of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.

    """

    def __init__(self,
                 efficacy: float,
                 tau: float,
                 bin_size: float = 5e-4,
                 epsilon: float = 1e-10,
                 max_delay: Optional[float] = None,
                 buffer_size: float = 0.,
                 key: Optional[str] = None,
                 reversal_potential: Optional[float] = None
                 ) -> None:

        # check input parameters
        ########################

        if tau < 0.:
            raise ValueError('Time constant tau cannot be negative. See docstring for further information.')

        # call super init
        #################

        super().__init__(kernel_function=exponential,
                         efficacy=efficacy,
                         bin_size=bin_size,
                         epsilon=epsilon,
                         max_delay=max_delay,
                         buffer_size=buffer_size,
                         key=key,
                         tau=tau,
                         reversal_potential=reversal_potential)


class DEExponentialSynapse(DESynapse):
    """Differential equation version of exponential synapse class. Represents average behavior of a defined post-synapse
     of a population.

    Parameters
    ----------
    efficacy
        Determines strength and direction of the synaptic response to input [unit = S if synapse is modulatory else A].
    tau
        Lumped time delay constant that determines the shape of the exponential synaptic kernel [unit = s].
    buffer_size
        Maximum time that information passed to the synapse needs to affect the synapse [unit = s] (default = 0.).
    reversal_potential
        See parameter docstring of :class:`DESynapse`.
    key
        Name of synapse type (default = None).

    Attributes
    ----------
    tau
        See parameter description.
    efficacy
        Determines strength and direction of the synaptic response to input [unit = S if synapse is modulatory else A].
    reversal_potential
        See parameter description.
    key
        See parameter description.

    """

    def __init__(self,
                 step_size: float,
                 efficacy: float,
                 tau: float,
                 buffer_size: float = 0.,
                 reversal_potential: Optional[float] = None,
                 tf_graph: Optional[tf.Graph] = None,
                 key: Optional[str] = None
                 ) -> None:
        """Instantiates base synapse.
        """

        # call super init
        #################

        super().__init__(step_size=step_size,
                         efficacy=efficacy,
                         buffer_size=buffer_size,
                         reversal_potential=reversal_potential,
                         tf_graph=tf_graph,
                         key=key)

        # set additional attributes
        ###########################

        self.tau = tau

        with self.tf_graph.as_default():

            # pre-calculate DE constants
            self.input_scaling = tf.constant(self.efficacy / self.tau, dtype=tf.float64)
            self.d1_scaling = tf.constant(1 / self.tau ** 2, dtype=tf.float64)
            self.d2_scaling = tf.constant(2 / self.tau, dtype=tf.float64)

            # update state variables
            delta_synaptic_response = self.input_scaling * self.synaptic_buffer[0] - self.d1_scaling * self.psp \
                                      - self.d2_scaling * self.synaptic_response

            # update synapse
            with tf.control_dependencies([delta_synaptic_response]):
                self.update_psp = self.psp.assign_add(self.step_size * self.synaptic_response)
            with tf.control_dependencies([delta_synaptic_response, self.update_psp]):
                self.update_synaptic_response = self.synaptic_response.assign_add(self.step_size *
                                                                                  delta_synaptic_response)

            # update buffer
            with tf.control_dependencies([delta_synaptic_response, self.update_synaptic_response]):
                self.update_buffer_1 = self.synaptic_buffer[0:-1].assign(self.synaptic_buffer[1:])
            with tf.control_dependencies([delta_synaptic_response, self.update_buffer_1]):
                self.update_buffer_2 = self.synaptic_buffer[-1].assign(0.)

    def update(self):
        """Updates attributes depending on current synapse state.
        """

        # call super method
        ###################

        super().update()

        # re-calculate DE constants
        ###########################

        with self.tf_graph.as_default():

            self.input_scaling = self.efficacy / self.tau
            self.d1_scaling = 1 / self.tau ** 2
            self.d2_scaling = 2 / self.tau

            # set decorator for synaptic current getter (only relevant for conductivity based synapses)
            if self.reversal_potential:
                self.synaptic_response_scaling = lambda membrane_potential: \
                    (self.reversal_potential - membrane_potential)
            else:
                self.synaptic_response_scaling = lambda membrane_potential: 1.0


################################################
# synapse with additional input transformation #
################################################


class TransformedInputSynapse(Synapse):
    """Synapse class that enables an additional transform of the synaptic input previous to the kernel convolution.
    
    Parameters
    ----------
    kernel_function
        See description of parameter `kernel_function` of :class:`Synapse`.
    input_transform
        Non-linear function that is applied to synaptic input before convolving it with the synaptic kernel.
    efficacy
        See description of parameter `efficacy` of :class:`Synapse`.
    bin_size
        See description of parameter `bin_size` of :class:`Synapse`.
    epsilon
        See description of parameter `epsilon` of :class:`Synapse`.
    max_delay
        See description of parameter `max_delay` of :class:`Synapse`.
    buffer_size
        See documentation of parameter `buffer_size` in :class:`Synapse`.
    reversal_potential
        See description of parameter `reversal_potential` of :class:`Synapse`.
    synapse_type
        See description of parameter `synapse_type` of :class:`Synapse`.
    input_transform_kwargs
        Dictionary with name-value pairs used as arguments for the input_transform.
    **kernel_function_kwargs
        See description of parameter `kernel_function_kwargs` of :class:`Synapse`.
        
    """

    def __init__(self,
                 kernel_function: Callable[..., FloatOrArray],
                 input_transform: Callable[..., FloatOrArray],
                 efficacy: float,
                 bin_size: float,
                 epsilon: float = 1e-14,
                 max_delay: Optional[float] = None,
                 buffer_size: float = 0.,
                 reversal_potential: Optional[float] = None,
                 key: Optional[str] = None,
                 input_transform_kwargs: Optional[dict] = None,
                 **kernel_function_kwargs: float
                 ) -> None:
        """Instantiates base synapse.
        """

        # call super init
        #################

        super().__init__(kernel_function=kernel_function,
                         efficacy=efficacy,
                         bin_size=bin_size,
                         epsilon=epsilon,
                         max_delay=max_delay,
                         buffer_size=buffer_size,
                         reversal_potential=reversal_potential,
                         key=key,
                         **kernel_function_kwargs)

        # add input transform
        #####################

        self.input_transform = input_transform
        self.input_transform_args = input_transform_kwargs if input_transform_kwargs else dict()

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
