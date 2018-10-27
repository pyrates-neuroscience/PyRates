"""This module contains a solver class that manages the integration of all ODE's in a circuit.
"""

# external packages
from typing import Union
import tensorflow as tf

# pyrates internal imports
from pyrates.parser import EquationParser

# meta infos
__author__ = "Richard Gast"
__status__ = "Development"


#####################
# base solver class #
#####################



