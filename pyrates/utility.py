# -*- coding: utf-8 -*-
#
#
# PyRates software framework for flexible implementation of neural
# network model_templates and simulations. See also:
# https://github.com/pyrates-neuroscience/PyRates
#
# Copyright (C) 2017-2018 the original authors (Richard Gast and
# Daniel Rose), the Max-Planck-Institute for Human Cognitive Brain
# Sciences ("MPI CBS") and contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
#
# CITATION:
#
# Richard Gast and Daniel Rose et. al. in preparation
"""Functions for performing parameter grid simulations with pyrates model_templates.
"""

# meta infos
__author__ = "Christoph Salomon, Richard Gast"
__status__ = "development"

# external imports
from typing import Union, Optional
import pandas as pd
import numpy as np
from copy import deepcopy

# pyrates imports
from pyrates.frontend import CircuitTemplate, OperatorTemplate, template
from pyrates.ir import clear_ir_caches


#####################################
# temporary file clearing functions #
#####################################


def clear(model: CircuitTemplate, **kwargs):
    """Function that clears all temporary files and caches that have been created via a PyRates model.

    Parameters
    ----------
    model
        Instance of a `CircuitTemplate`.
    kwargs
        Additional keyword arguments to be passed to `clear_frontend_caches`.
    """
    try:
        model.clear()
    except AttributeError:
        pass
    clear_frontend_caches(**kwargs)


def clear_frontend_caches(clear_template_cache=True, clear_ir_cache=True):
    """Utility to clear caches in the frontend.

    Parameters
    ----------
    clear_template_cache
        toggles whether or not to clear the template_cache that contains all previously loaded templates
    clear_ir_cache
        toggles whether or not to clear the cache of unique OperatorIR instances
    """
    if clear_template_cache:
        template.clear_cache()

    if clear_ir_cache:
        OperatorTemplate.cache.clear()
        clear_ir_caches()


########################
# simulation functions #
########################


def integrate(circuit: Union[str, CircuitTemplate], **kwargs):
    """Directly simulate dynamics of a circuit."""
    if type(circuit) is str:
        circuit = CircuitTemplate.from_yaml(circuit)
    results = circuit.run(**kwargs)
    if 'clear' in kwargs and kwargs['clear']:
        clear_frontend_caches()
    return results


#############################
# parameter sweep functions #
#############################

def linearize_grid(grid: dict, permute: bool = False) -> pd.DataFrame:
    """Turns the grid into a grid that can be traversed linearly, i.e. pairwise.

    Parameters
    ----------
    grid
        Parameter grid.
    permute
        If true, all combinations of the parameter values in grid will be created.

    Returns
    -------
    pd.DataFrame
        Resulting linear grid in form of a data frame.

    """

    arg_lengths = [len(arg) for arg in grid.values()]

    if len(list(set(arg_lengths))) == 1 and not permute:
        return pd.DataFrame(grid)
    elif permute:
        vals, keys = [], []
        for key, val in grid.items():
            vals.append(val)
            keys.append(key)
        new_grid = np.stack(np.meshgrid(*tuple(vals)), -1).reshape(-1, len(grid))
        return pd.DataFrame(new_grid, columns=keys)
    else:
        raise ValueError('Wrong number of parameter combinations. If `permute` is False, all parameter vectors in grid '
                         'must have the same number of elements.')


def adapt_circuit(circuit: Union[CircuitTemplate, str], params: dict, param_map: dict) -> CircuitTemplate:
    """Changes the parametrization of a circuit.

    Parameters
    ----------
    circuit
        Circuit instance.
    params
        Key-value pairs of the parameters that should be changed.
    param_map
        Map between the keys in params and the circuit variables.

    Returns
    -------
    CircuitIR
        Updated circuit instance.

    """

    if type(circuit) is str:
        circuit = deepcopy(CircuitTemplate.from_yaml(circuit))
    else:
        circuit = deepcopy(circuit)

    node_updates = {}
    edge_updates = []

    for key in params.keys():

        val = params[key]

        if 'nodes' in param_map[key]:
            for n in param_map[key]['nodes']:
                for v in param_map[key]['vars']:
                    node_updates[f"{n}/{v}"] = val
        else:
            edges = param_map[key]['edges']
            if len(edges[0]) < 3:
                for source, target in edges:
                    for var in param_map[key]['vars']:
                        edge = circuit.get_edge(source=source, target=target, idx=0)
                        edge_updates.append((edge[0], edge[1], {var: val}))
            else:
                for source, target, idx in edges:
                    for var in param_map[key]['vars']:
                        edge = circuit.get_edge(source=source, target=target, idx=idx)
                        edge_updates.append((edge[0], edge[1], {var: val}))

    return circuit.update_var(node_vars=node_updates, edge_vars=edge_updates)


def grid_search(circuit_template: Union[CircuitTemplate, str], param_grid: Union[dict, pd.DataFrame], param_map: dict,
                step_size: float, simulation_time: float, outputs: dict, inputs: Optional[dict] = None,
                sampling_step_size: Optional[float] = None, permute_grid: bool = False, **kwargs) -> tuple:
    """Function that runs multiple parametrizations of the same circuit in parallel and returns a combined output.

    Parameters
    ----------
    circuit_template
        Path to the circuit template.
    param_grid
        Key-value pairs for each circuit parameter that should be altered over different circuit parametrizations.
    param_map
        Key-value pairs that map the keys of param_grid to concrete circuit variables.
    step_size
        Simulation step-size in s.
    simulation_time
        Simulation time in s.
    outputs
        Output variables as provided to the `run` method of `:class:ComputeGraph`.
    inputs
        Extrinsic inputs as provided to the `run` method of `:class:ComputeGraph`.
    sampling_step_size
        Sampling step-size as provided to the `run` method of `:class:ComputeGraph`.
    permute_grid
        If true, all combinations of the provided param_grid values will be realized. If false, the param_grid values
        will be traversed pairwise.
    kwargs
        Additional keyword arguments passed to the `CircuitTemplate.run` call.


    Returns
    -------
    tuple
        Simulation results stored in a multi-index data frame, and the mapping between the data frame column names and
        the parameter grid.
    """

    # argument pre-processing
    #########################

    vectorization = kwargs.pop('vectorization', True)

    # linearize parameter grid if necessary
    if type(param_grid) is dict:
        param_grid = linearize_grid(param_grid, permute_grid)

    # create grid-structure of network
    ##################################

    # get parameter names and grid length
    param_keys = list(param_grid.keys())

    # assign parameter updates to each circuit, combine them to unconnected network and remember their parameters
    circuit_names = []
    circuit = CircuitTemplate(name='top_lvl', path='none')
    for idx in param_grid.index:
        new_params = {}
        for key in param_keys:
            new_params[key] = param_grid[key][idx]
        circuit_tmp = adapt_circuit(circuit_template, new_params, param_map)
        circuit_key = f'{circuit_tmp.name}_{idx}'
        circuit = circuit.update_template(circuits={circuit_key: circuit_tmp})
        circuit_names.append(circuit_key)
    param_grid.index = circuit_names

    # adjust input of simulation to combined network
    if inputs:
        for inp_key, inp in inputs.copy().items():
            inputs[f"all/{inp_key}"] = inp
            inputs.pop(inp_key)

    # adjust output of simulation to combined network
    outputs_new = {}
    for key, out in outputs.items():
        outputs_new[key] = f"all/{out}"

    # simulate the circuits behavior
    results = circuit.run(simulation_time=simulation_time,
                          step_size=step_size,
                          sampling_step_size=sampling_step_size,
                          inputs=inputs,
                          outputs=outputs_new,
                          vectorization=vectorization,
                          **kwargs)    # type: pd.DataFrame

    # # create dataframe that maps between output names and parameter sets
    # data, index = [], []
    # for key in results.keys():
    #     param_key = key[1].split('/')[0]
    #     data.append(param_grid.loc[param_key, :].values)
    # param_map = pd.DataFrame(data=np.asarray(data).T, columns=results.columns, index=param_grid.columns)

    # return results
    return results, param_grid


class Interactive2DParamPlot(object):
    def __init__(self, data_map: np.array, data_series: pd.DataFrame, x_values: np.array, y_values: np.array,
                 x_key: str, y_key: str, param_map: pd.DataFrame, tmin=0., title=None, **kwargs):
        """Creates an interactive 2D plot that allows visualization of time series using button press events

        Derive child class and change get_data() respectively to utilize this plotting method

        Parameters
        ----------
        data_map
            2D ndarray containing a value based on each column data_series, respectively.
        data_series
            DataFrame containing all data series used to create the data map
        x_values
            ndarray containing values of data-map columns.
        y_values
            ndarray containing values of data-map rows.
        x_key
        y_key
        param_map
            Dataframe containing the mapping between data-series columns (index) and x/y value pairs (columns)
            (as returned by `grid_search`).
        tmin
            Starting point for time-series plots in time units (float).
        title
            Title of 2D plot.
        kwargs
            Additional information to access a column in data_series if necessary

        Returns
        -------

        """

        import matplotlib.pyplot as plt

        dt = kwargs.pop('step_size', data_series.index[1] - data_series.index[0])
        state_var = kwargs.pop('state_var', '')
        tmin = int(tmin/dt)
        self.data = data_series.iloc[tmin:, :]
        self.x_values = x_values
        self.y_values = y_values

        # set up param map matrix
        self.map = pd.DataFrame(columns=self.x_values, index=self.y_values)
        for key in param_map.index:
            x, y = param_map.loc[key, x_key], param_map.loc[key, y_key]
            x_idx = np.argmin(np.abs(self.x_values - x))
            y_idx = np.argmin(np.abs(self.y_values - y))
            self.map.iloc[y_idx, x_idx] = key

        # Create canvas
        if 'subplots' in kwargs:
            self.fig, self.ax = kwargs.pop('subplots')
        else:
            skwargs = {}
            for key in ["figsize", "gridspec"]:
                if key in kwargs:
                    skwargs[key] = kwargs.pop(key)
            self.fig, self.ax = plt.subplots(ncols=2, nrows=1, **skwargs)

        # plot signals into right subplot
        cmap = kwargs.pop("cmap_lines", "magma")
        self.line_colors = plt.get_cmap(cmap, lut=self.data[self.map.iloc[0, 0]].shape[-1]).colors
        self.update_lineplot(0, 0)

        # Initiate marker
        self.marker = self.ax[0].plot(0, 0, 'x', color='white', markersize='10')

        # Plot 2D data in left subplot
        num_x_ticks = kwargs.pop('num_x_ticks', len(x_values))
        num_y_ticks = kwargs.pop('num_y_ticks', len(y_values))
        shrink = kwargs.pop('cbar_shrink', 0.5)
        im = self.ax[0].imshow(data_map, **kwargs)
        set_num_axis_ticks(ax=self.ax[0], num_x_ticks=num_x_ticks, num_y_ticks=num_y_ticks)
        self.ax[0].set_xlabel(x_key)
        self.ax[0].set_ylabel(y_key)
        self.ax[0].set_xticklabels([""] + list(np.round(x_values, decimals=2)))
        self.ax[0].set_yticklabels([""] + list(np.round(y_values, decimals=2)))
        if title:
            self.ax[0].set_title(title)
        plt.colorbar(im, ax=self.ax[0], shrink=shrink)

        # Call Interactive2DPlot class instance when mouse button is pressed inside the 2D plot
        self.fig.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        """Try to access a column in data_series using x and y values based on cursor position

        Is called on mouse button press event. Converts the current cursor coordinates inside the plot into x and y
        values based on the data in x_values and y_values. x and y values are used to access a column in data_series.
        Access of data_series can be customized in self.get_data().

        :param event:
        :return:
        """

        # Only allow button press events in the (left) 2D overview plot
        if event.inaxes != self.ax[0]:
            return

        # Reset axes
        self.marker[0].remove()

        # Transform cursor coordinates in x and y values
        x_sample = event.xdata
        y_sample = event.ydata

        # Add marker at event coordinates
        self.marker = self.ax[0].plot(x_sample, y_sample, 'x', color='white', markersize='10')

        # Update serial plot
        self.update_lineplot(int(np.round(x_sample, decimals=0)), int(np.round(y_sample, decimals=0)))

        # redraw figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_lineplot(self, x, y):

        data = self.get_data(x, y)
        lines = self.ax[1].get_lines()
        if lines:
            for line, key in zip(self.ax[1].get_lines(), data.keys()):
                line.set_data(data.index, data[key].values)
        else:
            for i, key in enumerate(data.keys()):
                self.ax[1].plot(data.index, data[key].values, c=self.line_colors[i])
        self.ax[1].set_title(
            f'x: {np.round(self.x_values[x], decimals=2)}, y: {np.round(self.y_values[y], decimals=2)}')
        ymin, ymax = np.min(data.values), np.max(data.values)
        margin = (ymax - ymin)*0.02
        self.ax[1].set_ylim([ymin-margin, ymax+margin])

    def set_map_xlabel(self, label):
        self.ax[0].set_xlabel(label)

    def set_map_ylabel(self, label):
        self.ax[0].set_ylabel(label)

    def set_map_title(self, title):
        self.ax[0].set_title(title)

    def set_series_xlabel(self, label):
        self.ax[1].set_xlabel(label)

    def set_series_ylabel(self, label):
        self.ax[1].set_ylabel(label)

    def get_data(self, x, y):
        return self.data[self.map.iloc[y, x]]


def set_num_axis_ticks(ax, num_x_ticks=10, num_y_ticks=10):
    """Set the number of x and y ticks of a plot axis

    Parameters
    ----------
    ax
    num_x_ticks
    num_y_ticks

    Returns
    -------

    """
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(num_x_ticks))
    ax.yaxis.set_major_locator(MaxNLocator(num_y_ticks))

##########################
# file storage functions #
##########################


def create_directory(path):
    """check if a directory exists and create it otherwise"""

    import os
    import errno

    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
