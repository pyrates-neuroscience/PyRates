
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
"""Provides functions to build MNE objects (raw, epoch, evoked) from PyRates simulation results (csv) or from circuit
objects.
"""

# external packages
import numpy as np
from typing import Union, Optional, List
from pandas import DataFrame, MultiIndex

# pyrates internal imports

# meta infos
__author__ = 'Richard Gast'
__status__ = 'Development'


################################################################################
# wrapper to build a pyrates-compatible output structure from annarchy monitor #
################################################################################


def pyrates_from_annarchy(monitors: list, vars: List[str], pop_average: bool = False,
                          monitor_names: Optional[List[str]] = None, **kwargs) -> DataFrame:
    """

    Parameters
    ----------
    monitors
    vars
    pop_average
    monitor_names

    Returns
    -------
    DataFrame

    """

    if not monitor_names:
        monitor_names = [str(i) for i in range(len(monitors))]
    results = {}
    for v in vars:
        for m, m_name in zip(monitors, monitor_names):
            if len(m_name) == 0:
                m_name = v
            if v == 'spike':
                val = m.get(v)
                val = m.smoothed_rate(val, **kwargs).T
            else:
                val = np.asarray(m.get(v))
            if len(val.shape) > 1:
                if pop_average:
                    results.update({f'{m_name}_avg': np.mean(val, axis=1)})
                else:
                    results.update({f'{m_name}_n{i}': val[:, i] for i in range(val.shape[1])})
            else:
                results.update({f'{m_name}': val})
    return DataFrame.from_dict(results)


#################################################
# grid search function for annarchy simulations #
#################################################


def grid_search_annarchy(param_grid: dict, param_map: dict, dt: float, simulation_time: float,
                         inputs: dict, outputs: dict, sampling_step_size: Optional[float] = None,
                         permute_grid: bool = False, circuit=None, **kwargs) -> DataFrame:
    """Function that runs multiple parametrizations of the same circuit in parallel and returns a combined output.

    Parameters
    ----------
    param_grid
        Key-value pairs for each circuit parameter that should be altered over different circuit parametrizations.
    param_map
        Key-value pairs that map the keys of param_grid to concrete circuit variables.
    dt
        Simulation step-size in s.
    simulation_time
        Simulation time in s.
    inputs
        Inputs as provided to the `run` method of `:class:ComputeGraph`.
    outputs
        Outputs as provided to the `run` method of `:class:ComputeGraph`.
    sampling_step_size
        Sampling step-size as provided to the `run` method of `:class:ComputeGraph`.
    permute_grid
        If true, all combinations of the provided param_grid values will be realized. If false, the param_grid values
        will be traversed pairwise.
    circuit
        Instance of ANNarchy network.
    kwargs
        Additional keyword arguments passed to the `:class:ComputeGraph` initialization.



    Returns
    -------
    DataFrame
        Simulation results stored in a multi-index data frame where each index lvl refers to one of the parameters of
        param_grid.

    """

    from ANNarchy import Population, Projection, Network, TimedArray, Monitor, ANNarchyException

    # linearize parameter grid if necessary
    if type(param_grid) is dict:
        param_grid = linearize_grid(param_grid, permute_grid)

    # create annarchy net if necessary
    if circuit is None:
        circuit = Network(everything=True)

    # assign parameter updates to each circuit and combine them to unconnected network
    circuit_names = []
    param_info = []
    param_split = "__"
    val_split = "--"
    comb = "_"
    populations, projections = {}, {}
    for n in range(param_grid.shape[0]):

        # copy and re-parametrize populations
        try:
            for p in circuit.get_populations():
                name = f'net{n}/{p.name}'
                p_new = Population(geometry=p.geometry, neuron=p.neuron_type, name=name,
                                   stop_condition=p.stop_condition, storage_order=p._storage_order,
                                   copied=False)
                p_new = adapt_pop(p_new, param_grid.iloc[n, :], param_map)
                populations[name] = p_new

                # add input to population
                for node, inp in inputs.items():
                    if node in name:
                        inp_name = f'{name}_inp'
                        inp = TimedArray(rates=inp, name=inp_name)
                        proj = Projection(pre=inp, post=p_new, target='exc')
                        proj.connect_one_to_one(1.0)
                        populations[inp_name] = inp
                        projections[inp_name] = proj
        except ANNarchyException:
            pass

        # copy and re-parametrize projections
        try:
            for c in circuit.get_projections():
                source = c.pre if type(c.pre) is str else c.pre.name
                target = c.post if type(c.post) is str else c.post.name
                source = f'net{n}/{source}'
                target = f'net{n}/{target}'
                name = f'{source}/{target}/{c.name}'
                c_new = Projection(pre=source, post=target, target=c.target, synapse=c.synapse_type, name=name,
                                   copied=False)
                c_new._store_connectivity(c._connection_method, c._connection_args, c._connection_delay, c._storage_format)
                c_new = adapt_proj(c_new, param_grid.iloc[n, :], param_map)
                projections[name] = c_new
        except ANNarchyException:
            pass

        # collect parameter and circuit name infos
        circuit_names.append(f'net{n}')
        param_names = list(param_grid.columns.values)
        param_info_tmp = [f"{param_names[i]}{val_split}{val}" for i, val in enumerate(param_grid.iloc[n, :])]
        param_info.append(param_split.join(param_info_tmp))

    net = Network()
    for p in populations.values():
        net.add(p)
    for c in projections.values():
        net.add(c)

    # adjust output of simulation to combined network
    nodes = [p.name for p in circuit.get_populations()]
    out_names, var_names, out_lens, monitors, monitor_names = [], [], [], [], []
    for out_key, out in outputs.copy().items():
        out_names_tmp, out_lens_tmp = [], []
        if out[0] in nodes:
            for i, name in enumerate(param_info):
                out_tmp = list(out)
                out_tmp[0] = f'{circuit_names[i]}/{out_tmp[0]}'
                p = net.get_population(out_tmp[0])
                monitors.append(Monitor(p, variables=out_tmp[-1], period=sampling_step_size, start=True,
                                        net_id=net.id))
                monitor_names.append(f'{name}{param_split}out_var{val_split}{out_key}{comb}{out[0]}')
                var_names.append(out_tmp[-1])
                out_names_tmp.append(f'{out_key}{comb}{out[0]}')
                out_lens_tmp.append(p.geometry[0])
        elif out[0] == 'all':
            for node in nodes:
                for i, name in enumerate(param_info):
                    out_tmp = list(out)
                    out_tmp[0] = f'{circuit_names[i]}/{node}'
                    p = net.get_population(out_tmp[0])
                    monitors.append(Monitor(p, variables=out_tmp[-1], period=sampling_step_size, start=True,
                                            net_id=net.id))
                    monitor_names.append(f'{name}{param_split}out_var{val_split}{out_key}{comb}{node}')
                    var_names.append(out_tmp[-1])
                    out_names_tmp.append(f'{out_key}{comb}{node}')
                    out_lens_tmp.append(p.geometry[0])
        else:
            node_found = False
            for node in nodes:
                if out[0] in node:
                    node_found = True
                    for i, name in enumerate(param_info):
                        out_tmp = list(out)
                        out_tmp[0] = f'{circuit_names[i]}/{node}'
                        p = net.get_population(out_tmp[0])
                        monitors.append(Monitor(p, variables=out_tmp[-1], period=sampling_step_size, start=True,
                                                net_id=net.id))
                        monitor_names.append(f'{name}{param_split}out_var{val_split}{out_key}{comb}{node}')
                        var_names.append(out_tmp[-1])
                        out_names_tmp.append(f'{out_key}{comb}{node}')
                        out_lens_tmp.append(p.geometry[0])
            if not node_found:
                raise ValueError(f'Invalid output identifier in output: {out_key}. '
                                 f'Node {out[0]} is not part of this network')
        out_names += list(set(out_names_tmp))
        out_lens += list(set(out_lens_tmp))
    #net.add(monitors)

    # simulate the circuits behavior
    net.compile()
    net.simulate(duration=simulation_time)

    # transform output into pyrates-compatible data format
    results = pyrates_from_annarchy(monitors, vars=list(set(var_names)),
                                    monitor_names=monitor_names, **kwargs)

    # transform results into long-form dataframe with changed parameters as columns
    multi_idx = [param_grid[key].values for key in param_grid.keys()]
    n_iters = len(multi_idx[0])
    outs = []
    for out_name, out_len in zip(out_names, out_lens):
        outs += [f'{out_name}_n{i}' for i in range(out_len)] * n_iters
    multi_idx_final = []
    for idx in multi_idx:
        for val in idx:
            for out_len in out_lens:
                multi_idx_final += [val]*len(out_names)*out_len
    index = MultiIndex.from_arrays([multi_idx_final, outs], names=list(param_grid.keys()) + ["out_var"])
    index = MultiIndex.from_tuples(list(set(index)), names=list(param_grid.keys()) + ["out_var"])
    results_final = DataFrame(columns=index, data=np.zeros_like(results.values), index=results.index)
    for col in results.keys():
        params = col.split(param_split)
        indices = [None] * len(results_final.columns.names)
        for param in params:
            var, val = param.split(val_split)[:2]
            idx = list(results_final.columns.names).index(var)
            try:
                indices[idx] = float(val)
            except ValueError:
                indices[idx] = val
        results_final.loc[:, tuple(indices)] = results[col].values

    return results_final


##########################################
# utility functions for annarchy objects #
##########################################


def adapt_pop(pop, params: dict, param_map: dict):
    """Changes the parametrization of a circuit.

    Parameters
    ----------
    pop
        ANNarchy population instance.
    params
        Key-value pairs of the parameters that should be changed.
    param_map
        Map between the keys in params and the circuit variables.

    Returns
    -------
    Population
        Updated population instance.

    """

    for key in params.keys():
        val = params[key]
        for op, var in param_map[key]['var']:
            nodes = param_map[key]['nodes'] if 'nodes' in param_map[key] else []
            for node in nodes:
                if node in pop.name:
                    try:
                        pop.set({var: float(val)})
                    except TypeError:
                        pop.set({var: val})
                    except (KeyError, ValueError):
                        pass

    return pop


def adapt_proj(proj, params: dict, param_map: dict):
    """Changes the parametrization of a circuit.

    Parameters
    ----------
    proj
        ANNarchy projection instance.
    params
        Key-value pairs of the parameters that should be changed.
    param_map
        Map between the keys in params and the circuit variables.

    Returns
    -------
    Projection
        Updated projection instance.

    """

    for key in params.keys():
        val = params[key]
        for op, var in param_map[key]['var']:
            edges = param_map[key]['edges'] if 'edges' in param_map[key] else []
            for source, target, edge in edges:
                if source in proj.name and target in proj.name and edge in proj.name:
                    try:
                        proj.set({var: float(val)})
                    except TypeError:
                        proj.set({var: val})
                    except [ValueError, KeyError]:
                        pass

    return proj


def linearize_grid(grid: dict, permute: bool = False) -> DataFrame:
    """Turns the grid into a grid that can be traversed linearly, i.e. pairwise.

    Parameters
    ----------
    grid
        Parameter grid.
    permute
        If true, all combinations of the parameter values in grid will be created.

    Returns
    -------
    DataFrame
        Resulting linear grid in form of a data frame.

    """

    arg_lengths = [len(arg) for arg in grid.values()]

    if len(list(set(arg_lengths))) == 1 and not permute:
        return DataFrame(grid)
    else:
        vals, keys = [], []
        for key, val in grid.items():
            vals.append(val)
            keys.append(key)
        new_grid = np.stack(np.meshgrid(*tuple(vals)), -1).reshape(-1, len(grid))
        return DataFrame(new_grid, columns=keys)
