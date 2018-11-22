"""This module provides wrappers for different backends that are needed by the parser class.
"""

# external imports
import tensorflow as tf
import typing as tp

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


class TensorflowBackend(tf.Graph):
    """

    """

    def add_variable(self, name, value, shape, dtype, scope=None):
        """

        Parameters
        ----------
        name
        value
        shape
        dtype
        scope

        Returns
        -------

        """

        with self.as_default():
            if scope:
                with tf.variable_scope(scope):
                    return tf.get_variable(name, shape, dtype, initializer=tf.constant_initializer(value))
            return tf.get_variable(name, shape, dtype, initializer=tf.constant_initializer(value))

    def add_constant(self, name, value, shape, dtype, scope=None):
        """

        Parameters
        ----------
        name
        value
        shape
        dtype
        scope

        Returns
        -------

        """

        with self.as_default():
            if scope:
                with tf.variable_scope(scope):
                    return tf.constant(value, dtype, shape, name)
            return tf.constant(value, dtype, shape, name)

    def add_placeholder(self, name, shape, dtype, scope=None):
        """

        Parameters
        ----------
        name
        shape
        dtype
        scope

        Returns
        -------

        """

        with self.as_default():
            if scope:
                with tf.variable_scope(scope):
                    return tf.placeholder(name, shape, dtype)
            return tf.placeholder(name, shape, dtype)
