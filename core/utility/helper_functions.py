"""Includes various types of useful functions.
"""

import numpy as np
import inspect
from typing import List, Dict, Union

__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


def set_instance(class_handle, instance_type: str=None, instance_params: dict=None, **kwargs) -> object:
    """Instantiates object of `class_handle` and returns instance.

    Parameters
    ----------
    class_handle
        Class that needs to be instantiated.
    instance_type
        Parametrized sub-class of `class_handle`.
    instance_params
        Dictionary with parameter name-value pairs.

    Returns
    -------
    object
        Instance of `class_handle`.

    """

    ##########################
    # check input parameters #
    ##########################

    if not instance_type and not instance_params:
        raise AttributeError('Either instance type or instance params have to be passed!')

    #################################################
    # if passed, instantiate pre-implemented object #
    #################################################

    if instance_type:

        # fetch names and object handles of sub-classes of object
        preimplemented_types = [cls.__name__ for cls in class_handle.__subclasses__()]
        type_objects = class_handle.__subclasses__()

        # loop over pre-implemented types and compare with passed type
        i = 0
        while True:
            if i >= len(preimplemented_types):
                raise AttributeError('Passed type is no pre-implemented parametrization of given object!')
            elif preimplemented_types[i] == instance_type:
                instance = type_objects[i](**kwargs)
                break
            else:
                i += 1

    #################################################################
    # if passed, instantiate/update object with instance parameters #
    #################################################################

    if instance_params and instance_type:

        # fetch attributes on object
        attributes = inspect.getmembers(class_handle, lambda a: not (inspect.isroutine(a)))
        attributes = [a for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))]

        # update passed parameters of instance
        for attr in attributes:
            instance = update_param(attr, instance_params, instance)

    elif instance_params:

        # instantiate new object
        instance = class_handle(**instance_params)

    return instance


def update_param(param: str, param_dict: dict, object_instance) -> object:
    """Checks whether param is a key in param_dict. If yes, the corresponding value in param_dict will be updated in
    object_instance.

    Parameters
    ----------
    param
        Specifies parameter to check.
    param_dict
        Potentially contains param.
    object_instance
        Instance for which to update param.

    Returns
    -------
    object
        Object instance.

    """

    if param not in object_instance:
        raise AttributeError('Param needs to be an attribute of object_instance!')

    if param in param_dict:
        setattr(object_instance, param, param_dict[param])

    return object_instance


def interpolate_array(old_step_size, new_step_size, y, interpolation_type='cubic', axis=0):
    """
    Interpolates time-vectors with scipy.interpolate.interp1d.

    :param old_step_size: old simulation step size [unit = s].
    :param new_step_size: new simulation step size [unit = s].
    :param y: vector to be interpolated
    :param interpolation_type: can be 'linear' or spline stuff.
    :param axis: axis along which y is to be interpolated (has to have same length as t/old_step_size)

    :return: interpolated vector

    """

    from scipy.interpolate import interp1d

    # create time vectors
    x_old = np.arange(y.shape[axis]) * old_step_size

    new_steps_n = int(np.ceil(x_old[-1] / new_step_size))
    x_new = np.linspace(0, x_old[-1], new_steps_n)

    # create interpolation function
    f = interp1d(x_old, y, axis=axis, kind=interpolation_type, bounds_error=False, fill_value='extrapolate')

    return f(x_new)


def NMRSE(x: np.ndarray, y: np.ndarray) -> Union[float, np.ndarray]:
    """Calculates the normalized root mean squared error of two vectors of equal length.

    Parameters
    ----------
    x,y
        1D arrays to calculate the NMRSE between.

    Returns
    -------
    float
        Normalized root mean squared error.

    """

    max_val = np.max((np.max(x, axis=0), np.max(y, axis=0)))
    min_val = np.min((np.min(x, axis=0), np.min(y, axis=0)))

    diff = x - y

    return np.sqrt(np.sum(diff ** 2, axis=0)) / (max_val - min_val)