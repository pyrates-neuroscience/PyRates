*********************************
Mathematical Framework and Syntax
*********************************

Here, we document the mathematical framework `PyRates` was built upon and the syntax that should be used
to implement dynamical systems models in `PyRates`.

Mathematical Framework
----------------------

`Dynamical systems <http://www.scholarpedia.org/article/Dynamical_systems>`_ (DS) define how a system evolves over time in
`state space <http://www.scholarpedia.org/article/State_space>`_, where time can be either discrete or continuous.
Continuous time DS are typically defined as sets of differential equations (DEs) and it is this family of mathematical models
that `PyRates` supports.
Specifically, `PyRates` allows the implementation of any DS that can be cast either as a system of `ordinary DEs <https://en.wikipedia.org/wiki/Ordinary_differential_equation>`_

.. math::
        \frac{d y}{d t} = \dot y = f(y, \theta, t),

or as a system of `delayed DEs <http://scholarpedia.org/article/Delay-differential_equations>`_ with constant delays:

.. math::
        \dot y = f(y(t), \theta, t, y(t-\tau_1), ..., y(t-\tau_n)).

In those equations, :math:`y(t)` represents the state-vector of the DS at time :math:`t`, :math:`\theta` is a vector of
parameters that control the system behavior, and :math:`\tau_i` represents one of the :math:`n` constant delays that
may be used to define delayed DEs.
The right-hand side of these equations define a vector-field over the :math:`N`-dimensional state-space of the system,
where :math:`N` is the dimensionality of :math:`y`.
It is this vector-field that must be evaluated for the vast majority of the existing numerical DS analysis methods.
`PyRates` provides an intuitive, light-weight interface to implement this vector-field and translates it into various
backends.

Mathematical Syntax
-------------------

Any DS implemented in `PyRates` requires the definition of a set of DEs, which can be complemented by algebraic
equations of the form :math:`a = g(y, \theta, t)` to compute temporary variables to be used within the DE system.
The mathematical syntax used for any model definition is described in detail in [1]_.
In short, it follows the `Python` syntax for the definition of most mathematical operations, while adding a
`PyRates`-specific syntax for the definition of DEs.
Any equation is defined as a string, no matter whether you are using the `YAML` or `Python` interface.
A typical differential equation would look as follows:

.. code-block::

    "d/dt * u = u**2 - a"

Alternatively, DEs can be defined via the dot formalism:

.. code-block::

    "u' = u**2 - a"

where :code:`u'` is the `Python` version of :math:`\dot u`.
Non-differential equations can also be added, and follow standard `Python` conventions:

.. code-block::

    "a = sin(u*t*2*pi)"

In this example, you can see that `PyRates` allows for the usage of function calls and well-known constants in
its equations. Below, you will find a list of all the function calls and constants that are currently supported by
`PyRates`. In addition, you will find a list of variable names that are prohibited from usage, since they are blocked
for `PyRates`-internal variable definitions.
Finally, note that any higher-order DE has to be translated into a set of coupled first-order DEs, prior to implementation
in `PyRates`.
For example, it is *NOT* allowed to implement

.. math::
        \frac{d^2 x}{dt^2} = -x - \frac{d x}{d t} + c

directly. Instead, one would have to transform this second-order DE into a set of two coupled first-order DEs

.. math::
        \frac{d x}{d t} &= z,\\
        \frac{d z}{d t} &= b - x - z.

These can then be implemented in `PyRates`:

.. code-block::

    eqs= ["x' = z",
          "z' = b - x - z"]

Supported Functions
-------------------

The following function calls can be used for model definitions in `PyRates`:

- :code:`sin`: implements the sine function.
- :code:`cos`: implements the cosine function
- :code:`tan`: implements the tangent function
- :code:`sinh`: implements the hyperbolic sine function.
- :code:`cosh`: implements the hyperbolic cosine function.
- :code:`tanh`: implements the hyperbolic tangent function.
- :code:`arcsin`: implements the inverse sine function.
- :code:`arccos`: implements the inverse cosine function.
- :code:`arctan`: implements the inverse tangent function.
- :code:`exp`: implements the exponential function, i.e. :math:`exp(x) = e^x`
- :code:`absv`: returns the absolute value of its argument, i.e. :math:`absv(x) = |x|`
- :code:`sigmoid`: implements the logistic function, i.e. :math:`sigmoid(x) = \frac{1}{1 + e^x}`
- :code:`imag`: returns the imaginary part of its argument, e.g. :code:`imag(1.0+3.0j) = 3.0`
- :code:`real`: returns the real part of its argument, e.g. :code:`real(1.0+3.0j) = 1.0`
- :code:`conj`: returns the complex conjugate of its argument, e.g. :code:`conj(1.0+3.0j) = 1.0-3.0j`
- :code:`log`: implements the natural logarithm, i.e. :math:`log(e^x) = x`
- :code:`sum`: implements the sum operator, e.g. :code:`sum([1, 2, 3]) = 6`
- :code:`mean`: calculates the unweighted average of a vector, e.g. :code:`mean([1, 2, 3]) = 2`
- :code:`max`: returns the maximum value of a vector, e.g. :code:`max([1, 2, 3]) = 3`
- :code:`min`: returns the minimum value of a vector, e.g. :code:`min([1, 2, 3]) = 1`
- :code:`past`: function call that should be used to implement delayed DEs. The function :code:`past` takes two arguments - the first one is the state variable and the second one is the constant delay. As an example, :code:`past(x, tau)` implements :math:`x(t-\tau)`.
- :code:`randn`: function that generates a random number, generated by randomly drawing a sample from a standard Gaussian distribution. No arguments are required.
- :code:`round`: Rounds a real number to the closest natural number.
- :code:`matmul`: Implements a multiplication of two matrices (matching inner dimensions required).
- :code:`matvec`: Implements a multiplication of a matrix with a vector, i.e. :code:`y = matvec(A, x)` implements :math:`y = A x`.
- :code:`index`: Indexes into the first dimension of a variable, i.e. :code:`index(x, 1)` corresponds to :code:`x[1]`.
- :code:`index_range`: Slices into the first dimension of a variable, i.e. :code:`index_range(x, 1, 5)` corresponds to :code:`x[1:5]`.
- :code:`index_axis`: Indexes into a given dimension of a variable, i.e. :code:`index_axis(x, 1, 1)` corresponds to :code:`x[:, 1]`, where the third argument to :code:`index_axis` indicates the dimension where the index (second argument) should be applied.

References
^^^^^^^^^^

.. [1] `Gast, R., Rose, D., Salomon, C., Möller, H. E., Weiskopf, N., & Knösche, T. R. (2019).
        PyRates-A Python framework for rate-based neural simulations. PloS one, 14(12), e0225900. <https://doi.org/10.1371/journal.pone.0225900>`_