*****************************
Installation and Requirements
*****************************

Prerequisites
-------------

`PyRates` requires `Python >= 3.6`.
We recommend to use `Anaconda` to create a new python environment with `Python >= 3.6`.
After that, the installation instructions provided below should work independent of the operating system.

Dependencies
------------

`PyRates` has the following hard dependencies:

- `numpy`
- `networkx`
- `pandas`
- `ruamel.yaml`
- `sympy`
- `scipy`

Following the installation instructions below, these packages will be installed automatically, if not already installed within the `Python` environment you are using.
In addition, `PyRates` has soft dependencies, that are not necessarily installed automatically, but are required to make use of all functionalities of `PyRates`.
These are:

- `torch` for `Python` (required for the `torch` backend of `PyRates`)
- `tensorflow` for `Python` (required for the `tensorflow` backend of `PyRates`)
- `Julia` (required for the `julia` backend of `PyRates`)
- a `Fortran90` compiler (required for the `fortran` backend of `PyRates`)
- `Matlab` (required for the `matlab` backend of `PyRates`)
- `pytest` (required if you want to use the `test` library of `PyRates`)
- `matplotlib` (required if you want to use the `grid_search` functionalities of `PyRates`)
- `jupyter notebook` (required if you want full support for the documentation examples)

Installation
------------

`PyRates` can be installed via the `pip` command.  Simply run the following line from a terminal with the target Python
environment being activated:

.. code-block:: bash

   pip install pyrates


You can install optional (non-default) packages by specifying one or more options in brackets, e.g.:

.. code-block:: bash

   pip install pyrates[backends]


Available options are `backends` (includes `tensorflow` and `torch`), `dev` (includes `pytest` and `bump2version`),
and `all` (includes all optional packages).

Alternatively, it is possible to clone this repository and run one of the following lines
from the directory in which the repository was cloned:

.. code-block:: bash

   python setup.py install

or

.. code-block:: bash

   pip install .[<options>]