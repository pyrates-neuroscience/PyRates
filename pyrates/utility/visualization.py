
# -*- coding: utf-8 -*-
#
#
# PyRates software framework for flexible implementation of neural 
# network models and simulations. See also: 
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
"""Visualization functionality for pyrates networks and backend simulations.
"""

# external imports
import seaborn as sb
import networkx.drawing.nx_pydot as pydot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# pyrates internal imports

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


def plot_timeseries(data, variable='value', plot_style='line_plot', bg_style="darkgrid", **kwargs):
    """Plot timeseries

    Parameters
    ----------
    data
        Pandas dataframe containing the results of a pyrates simulation.
    variable
        Name of the variable to be plotted
    plot_style
        Can be either `line_plot` for plotting with seaborn.lineplot() or `ridge_plot` for using seaborn.lineplot()
        on a grid with the y-axis being separated for each population.
    bg_style
        Background style of the seaborn plot
    kwargs
        Additional key-word arguments for the seaborn function.

    Returns
    -------
    ax
        Figure handle of the plot.

    """

    sb.set_style(bg_style)

    # pre-process data
    demean = kwargs.pop('demean', False)
    if demean:
        for i in range(data.shape[1]):
            data.iloc[:, i] -= np.mean(data.iloc[:, i])
            data.iloc[:, i] /= np.std(data.iloc[:, i])
    title = kwargs.pop('title', '')

    # Convert the dataframe to long-form or "tidy" format if necessary
    data['time'] = data.index
    df = pd.melt(data,
                 id_vars='time',
                 var_name='node',
                 value_name=variable)

    if 'ax' not in kwargs.keys():
        _, ax = plt.subplots()
        kwargs['ax'] = ax

    if plot_style == 'line_plot':

        # simple timeseries plot
        if 'ci' not in kwargs:
            kwargs['ci'] = None
        ax = sb.lineplot(data=df, x='time', y=variable, hue='node', **kwargs).set_title(title)

    elif plot_style == 'ridge_plot':

        # create color palette
        col_pal_args = ['start', 'rot', 'gamma', 'hue', 'light', 'dark', 'reverse', 'n_colors']
        kwargs_tmp = {}
        for key in kwargs.copy().keys():
            if key in col_pal_args:
                kwargs_tmp[key] = kwargs.pop(key)
        if not 'n_colors' in kwargs_tmp:
            kwargs_tmp['n_colors'] = 10
        pal = sb.cubehelix_palette(**kwargs_tmp)

        # create facet grid
        grid_args = ['col_wrap', 'sharex', 'sharey', 'height', 'aspect', 'row_order', 'col_order',
                     'dropna', 'legend_out', 'margin_titles', 'xlim', 'ylim', 'gridspec_kws', 'size']
        kwargs_tmp = {}
        for key in kwargs.copy().keys():
            if key in grid_args:
                kwargs_tmp[key] = kwargs.pop(key)
        facet_hue = kwargs.pop('facet_hue', 'node')
        facet_row = kwargs.pop('facet_row', 'node')
        ax = sb.FacetGrid(df, row=facet_row, hue=facet_hue, palette=pal, **kwargs_tmp)
        plt.close(plt.figure(plt.get_fignums()[-2]))

        # map line plots
        ax.map(sb.lineplot, 'time', variable, ci=None)
        ax.map(plt.axhline, y=0, lw=2, clip_on=False)

        # labeling args
        label_args = ['fontsize']
        kwargs_tmp = {}
        for key in kwargs.copy().keys():
            if key in label_args:
                kwargs_tmp[key] = kwargs.pop(key)

        # Define and use a simple function to label the plot in axes coordinates
        def label(x, color, label):
            ax_tmp = plt.gca()
            ax_tmp.text(0, .1, label, fontweight="bold", color=color,
                        ha="left", va="center", transform=ax_tmp.transAxes, **kwargs_tmp)

        ax.map(label, 'time')

        # Set the subplots to overlap
        hspace = kwargs.pop('hspace', -.05)
        ax.fig.subplots_adjust(hspace=hspace)

        # Remove axes details that don't play well with overlap
        ax.set_titles("")
        ax.set(yticks=[])
        ax.despine(bottom=True, left=True)

    else:

        raise ValueError(f'Plot style is not supported by this function: {plot_style}. Check the documentation of the '
                         f'argument `plot_style` for valid options.')

    return ax


def plot_connectivity(fc, threshold=None, plot_style='heatmap', bg_style='whitegrid', node_order=None,
                      auto_cluster=False, **kwargs):
    """Plot functional connectivity between nodes in backend.

    Parameters
    ----------
    fc
        Pandas dataframe containing or numpy array containing the functional connectivities.
    threshold
        Connectivtiy threshold to be applied (only connectivities larger than the threshold will be shown).
    plot_style
        Can either be `heatmap` for plotting with seaborn.heatmap or `circular_graph` for plotting with
         mne.viz.plot_connectivity_circle. Check out the respective function docstrings for information on
         their arguments (can be passed to kwargs).
    bg_style
        Only relevant if plot_style == heatmap. Then this will define the style of the background of the plot.
    node_order
        Order in which the nodes should appear in the plot.
    auto_cluster
        If true, automatic cluster detection will be used to arange the nodes
    kwargs
        Additional arguments for the fc calculation or fc plotting that can be passed.

    Returns
    -------
    ax
        Handle of the axis the plot was created in.
    """

    # turn fc into dataframe if necessary
    if type(fc) is np.ndarray:
        rows = kwargs.pop('yticklabels') if 'yticklabels' in kwargs.keys() else [str(i) for i in range(fc.shape[0])]
        cols = kwargs.pop('xticklabels') if 'xticklabels' in kwargs.keys() else [str(i) for i in range(fc.shape[0])]
        fc = pd.DataFrame(fc, index=rows, columns=cols)

    # apply threshold
    if threshold:
        fc[fc < threshold] = 0.

    # cluster the columns
    #####################

    if auto_cluster:

        idx_r = [i for i in range(fc.shape[0])]
        idx_c = [i for i in range(fc.shape[1])]

        # Create a categorical color palette for node groups
        col_pal_args = ['h', 's', 'l']
        kwargs_tmp = {}
        for key in kwargs.keys():
            if key in col_pal_args:
                kwargs_tmp[key] = kwargs.pop(key)
        node_pal = sb.husl_palette(len(idx_c), **kwargs_tmp)
        nodes = fc.columns.values
        node_lut = dict(zip(map(str, nodes), node_pal))

        # Convert the palette to vectors that will be drawn on the side of the fc plot
        node_colors = pd.Series(nodes, index=fc.columns).map(node_lut)

    elif node_order:

        idx_c = [node_order.index(n) for n in fc.columns.values]
        idx_r = [i for i in range(fc.shape[0])]

    else:

        idx_r = [i for i in range(fc.shape[0])]
        idx_c = [i for i in range(fc.shape[1])]

    fc = fc.iloc[idx_r, idx_c]

    # plot the functional connectivities
    ####################################

    # choose plot style
    if plot_style == 'heatmap':

        # seaborn plot
        if 'xticklabels' not in kwargs:
            kwargs['xticklabels'] = fc.columns.values[idx_c]
        if 'yticklabels' not in kwargs:
            kwargs['yticklabels'] = fc.index[idx_r]

        sb.set_style(bg_style)

        if auto_cluster:
            ax = sb.clustermap(data=fc, row_colors=node_colors, col_colors=node_colors, **kwargs)
        else:
            ax = sb.heatmap(fc, **kwargs)

    elif plot_style == 'circular_graph':

        # mne python plot
        from mne.viz import circular_layout, plot_connectivity_circle

        # get node order for node layout
        node_names = fc.columns.values
        if auto_cluster:
            cluster_args = ['method', 'metric', 'z_score', 'standard_scale']
            kwargs_tmp = {}
            for key in kwargs.keys():
                if key in cluster_args:
                    kwargs_tmp[key] = kwargs.pop(key)
            clust_map = sb.clustermap(data=fc, row_colors=node_colors, col_colors=node_colors, **kwargs_tmp)
            node_order = [node_names[idx] for idx in clust_map.dendrogram_row.reordered_ind]
        elif not node_order:
            node_order = list(node_names)

        # create circular node layout
        kwargs_tmp = {}
        layout_args = ['start_pos', 'start_between', 'group_boundaries', 'group_sep']
        for key in kwargs.keys():
            if key in layout_args:
                kwargs_tmp[key] = kwargs.pop(key)
        node_angles = circular_layout(node_names, node_order, **kwargs_tmp)

        # plot the circular graph
        ax = plot_connectivity_circle(fc.values, node_names, node_angles=node_angles, **kwargs)

    else:

        raise ValueError(f'Plot style is not supported by this function: {plot_style}. Check the documentation of the '
                         f'argument `plot_style` for valid options.')

    return ax


def plot_phase(data, bg_style='whitegrid', **kwargs):
    """Plot phase of populations in a polar plot.

    Parameters
    ----------
    data
        Long (tidy) format dataframe containing fields `node`, `phase` and `amplitude`.
    bg_style
        Background style of the plot.
    kwargs
        Additional keyword args to be passed to `seaborn.FacetGrid` or `seaborn.scatterplot`

    Returns
    -------
    ax
        Axis handle of the created plot.

    """

    sb.set(style=bg_style)

    # create facet grid
    grid_args = ['col_wrap', 'sharex', 'sharey', 'height', 'aspect', 'palette', 'row_order', 'col_order',
                 'hue_order', 'hue_kws', 'dropna', 'legend_out', 'margin_titles', 'xlim', 'ylim',
                 'gridspec_kws', 'size']
    kwargs_tmp = {}
    for key in kwargs.keys():
        if key in grid_args:
            kwargs_tmp[key] = kwargs.pop(key)
    ax = sb.FacetGrid(data, hue='node', subplot_kws=dict(polar=True), despine=False, **kwargs_tmp)

    # plot phase and amplitude into polar plot
    scatter_kwargs = ['style', 'sizes', 'size_order', 'size_norm', 'markers', 'style_order', 'x_bins', 'y_bins',
                      'units', 'estimator', 'ci', 'n_boot', 'alpha', 'x_jitter', 'y_jitter', 'legend', 'ax']
    kwargs_tmp2 = {}
    for key in kwargs.keys():
        if key in scatter_kwargs:
            kwargs_tmp2[key] = kwargs.pop(key)
    ax.map(sb.scatterplot, 'phase', 'amplitude', **kwargs_tmp2)

    # plot customization
    ax_tmp = ax.facet_axis(0, 0)
    ax_tmp.set_ylim(np.min(data['amplitude']), np.max(data['amplitude']))
    ax_tmp.axes.yaxis.set_label_coords(1.15, 0.75)
    ax_tmp.set_ylabel(ax_tmp.get_ylabel(), rotation=0)
    locs, _ = plt.yticks()
    plt.yticks(locs)
    locs, labels = plt.xticks()
    labels = [np.round(l._x, 3) for l in labels]
    plt.xticks(locs, labels)

    return ax


def plot_psd(data, fmin=0, fmax=100, tmin=0.0, **kwargs):
    """Plots the power-spectral density for each column in data.

    Parameters
    ----------
    data
        Dataframe with simulation results.
    fmin
        Minimum frequency to be displayed.
    fmax
        Maximum frequency to be displayed.
    tmin
        Time at which to start psd calculation.
    kwargs
        Additional keyword arguments to be passed to `mne.viz.plot_raw_psd`.

    Returns
    -------
    ax
        Handle of the created plot.

    """

    from pyrates.utility import mne_from_dataframe
    from mne.viz import plot_raw_psd

    raw = mne_from_dataframe(data)

    return plot_raw_psd(raw, tmin=tmin, fmin=fmin, fmax=fmax, **kwargs).axes


def plot_tfr(data, freqs, nodes=None, separate_nodes=True, **kwargs):
    """

    Parameters
    ----------
    data
        Numpy array (n x f x t) containing the instantaneous power estimates for each node (n), each frequency (f) at
        every timestep (t).
    separate_nodes
        If true, create a separate figure for each node.
    kwargs
        Additional keyword arguments to be passed to `seaborn.heatmap` or `seaborn.FacetGrid`.

    Returns
    -------
    ax
        Handle of the created plot.

    """

    if not nodes:
        nodes = [i for i in range(data.shape[0])]
    if 'xticklabels' not in kwargs.keys():
        if 'step_size' in kwargs.keys():
            xticks = np.round(np.arange(0, data.shape[2]) * kwargs.pop('step_size'), decimals=3)
            kwargs['xticklabels'] = [str(t) for t in xticks]
    if 'yticklabels' not in kwargs.keys():
        kwargs['yticklabels'] = [str(f) for f in freqs]

    if separate_nodes:

        # plot heatmap separately for each node
        for n in range(data.shape[0]):
            _, ax = plt.subplots()
            ax = sb.heatmap(data[n, :, :], ax=ax, **kwargs)

    else:

        # Convert the dataframe to long-form or "tidy" format
        indices = pd.MultiIndex.from_product((nodes, freqs, range(data.shape[2])), names=('nodes', 'freqs', 'time'))
        data = pd.DataFrame(data.flatten(), index=indices, columns=('values',)).reset_index()

        # create facet grid
        grid_args = ['col_wrap', 'sharex', 'sharey', 'height', 'aspect', 'palette', 'col_order',
                     'dropna', 'legend_out', 'margin_titles', 'xlim', 'ylim', 'gridspec_kws', 'size']
        kwargs_tmp = {}
        for key in kwargs.keys():
            if key in grid_args:
                kwargs_tmp[key] = kwargs.pop(key)
        ax = sb.FacetGrid(data, col='nodes', **kwargs_tmp)

        # map heatmaps to grid
        ax.map_dataframe(draw_heatmap, 'time', 'freqs', 'values', cbar=False, square=True, **kwargs)

    return ax


def write_graph(net, out_file='png'):
    """Draw graph from backend config.
    """

    pydot_graph = pydot.to_pydot(net)

    file_format = out_file.split('.')[1]
    if file_format == 'png':
        pydot_graph.write_png(out_file)


def draw_heatmap(*args, **kwargs):
    """Wraps seaborn.heatmap to work with long, tidy format dataframes.
    """
    data = kwargs.pop('data')
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    return sb.heatmap(d, **kwargs)
