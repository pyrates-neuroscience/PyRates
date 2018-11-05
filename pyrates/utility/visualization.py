"""Visualization functionality for pyrates networks and network simulations.
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
    handle
        Figure handle of the plot.

    """

    sb.set_style(bg_style)

    # Convert the dataframe to long-form or "tidy" format
    df = pd.melt(data,
                 id_vars=['time'],
                 var_name='node',
                 value_name=variable)

    if 'ax' not in kwargs.keys():
        _, ax = plt.subplots()
        kwargs['ax'] = ax

    if plot_style == 'line_plot':

        # simple timeseries plot
        ax = sb.lineplot(data=df, x='time', y=variable, hue='node', **kwargs)

    elif plot_style == 'ridge_plot':

        # gridded timeseries plot
        pal = sb.cubehelix_palette(10, rot=-.25, light=.7)
        ax = sb.FacetGrid(df, row='node', hue='node', aspect=15, height=.5, palette=pal)
        plt.close(plt.figure(plt.get_fignums()[-2]))

        ax.map(sb.lineplot, 'time', variable, ci=None)
        ax.map(plt.axhline, y=0, lw=2, clip_on=False)

        # Define and use a simple function to label the plot in axes coordinates
        def label(x, color, label):
            ax_tmp = plt.gca()
            ax_tmp.text(0, .2, label, fontweight="bold", color=color,
                        ha="left", va="center", transform=ax_tmp.transAxes)

        ax.map(label, 'time')

        # Set the subplots to overlap
        ax.fig.subplots_adjust(hspace=-.05)

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
    """Plot functional connectivity between nodes in network.

    Parameters
    ----------
    fc
        Pandas dataframe containing or numpy array containing the functional connectivities.
    metric
        Type of connectivtiy measurement that should be used. Can be `cov` for covariance, `corr` for correlation or
        one of the following synchronization metrics that will be calculated via
        mne.connectivtiy.spectral_connectivity (check out this function for information on its arguments - these can be
        passed via kwargs):
            - `coh` for coherence
            - `cohy` for coherency
            - `imcoh` for imaginary coherence
            - `plv` for phase locking value
            - `ppc` for pairwise phase consistency
            - `pli` for phase lag index
            - `pli2_unbiased` for unbiased estimate of squared phase lag index
            - `wpli`for weighted phase lag index
            - `wpli2_debiased` for debiased weighted pahse lag index
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
    fc
        Matrix containing the pairwise functional connectivities.

    """

    # turn fc into dataframe if necessary
    if type(fc) is np.ndarray:
        rows = kwargs['xticklabels'] if 'xticklabels' in kwargs.keys() else [i for i in range(fc.shape[0])]
        cols = kwargs['yticklabels'] if 'yticklabels' in kwargs.keys() else [i for i in range(fc.shape[0])]
        fc = pd.DataFrame(fc, index=rows, columns=cols)

    # apply threshold
    if threshold:
        fc[fc < threshold] = 0.

    # cluster the columns
    #####################

    if auto_cluster:

        idx = [i for i in range(fc.shape[0])]

        # Create a categorical color palette for node groups
        node_pal = sb.husl_palette(len(idx), s=.45)
        nodes = fc.columns.values
        node_lut = dict(zip(map(str, nodes), node_pal))

        # Convert the palette to vectors that will be drawn on the side of the fc plot
        node_colors = pd.Series(nodes, index=fc.columns).map(node_lut)

    elif node_order:

        idx = [node_order.index(n) for n in fc.columns.values]

    else:

        idx = [i for i in range(len(fc.columns.values))]

    fc = fc.iloc[idx]
    fc = fc.T.iloc[idx]

    # plot the functional connectivities
    ####################################

    # choose plot style
    if plot_style == 'heatmap':

        # seaborn plot
        if not 'xticklabels' in kwargs.keys():
            kwargs['xticklabels'] = fc.columns.values[idx]
        if not 'yticklabels' in kwargs.keys():
            kwargs['yticklabels'] = fc.columns.values[idx]

        sb.set_style(bg_style)

        if auto_cluster:

            ax = sb.clustermap(data=fc, row_colors=node_colors, col_colors=node_colors, **kwargs)

        else:

            ax = sb.heatmap(fc, **kwargs)

    elif plot_style == 'circular_graph':

        # mne python plot
        from mne.viz import circular_layout, plot_connectivity_circle

        # create circular node layout
        node_names = fc.columns.values
        if not node_order:
            node_order = list(node_names)
        kwargs_tmp = {}
        layout_args = ['start_pos', 'start_between', 'group_boundaries', 'group_sep']
        for key in kwargs.keys():
            if key in layout_args:
                kwargs_tmp[key] = kwargs.pop(key)
        node_angles = circular_layout(node_names, node_order, **kwargs_tmp)

        # plot the circular graph
        ax = plot_connectivity_circle(fc, node_names, node_angles=node_angles, **kwargs)

    else:

        raise ValueError(f'Plot style is not supported by this function: {plot_style}. Check the documentation of the '
                         f'argument `plot_style` for valid options.')

    return ax


def plot_phase(data, bg_style='whitegrid', **kwargs):
    """Plot phase of populations in a polar plot.
    """

    sb.set(style=bg_style)

    # create facet grid
    facet_kwargs = ['col_wrap', 'sharex', 'sharey', 'height', 'aspect', 'palette', 'row_order', 'col_order',
                    'hue_order', 'hue_kws', 'dropna', 'legend_out', 'margin_titles', 'xlim', 'ylim',
                    'gridspec_kws', 'size']
    kwargs_tmp = {}
    for key in kwargs.keys():
        if key in facet_kwargs:
            kwargs_tmp[key] = kwargs.pop(key)
    ax = sb.FacetGrid(data, hue='node', subplot_kws=dict(polar=True), despine=False)

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
    """

    Parameters
    ----------
    data
    fmin
    fmax
    tmin
    kwargs

    Returns
    -------

    """

    from pyrates.utility import mne_from_dataframe
    from mne.viz import plot_raw_psd

    raw = mne_from_dataframe(data)

    return plot_raw_psd(raw, tmin=tmin, fmin=fmin, fmax=fmax, **kwargs)


def plot_tfr(power, freqs, nodes=None, separate_nodes=True, **kwargs):
    """

    Parameters
    ----------
    power
    plot_style
    separate_nodes
    kwargs

    Returns
    -------

    """

    if not nodes:
        nodes = [i for i in range(power.shape[0])]
    if 'xticklabels' not in kwargs.keys():
        if 'step_size' in kwargs.keys():
            xticks = np.round(np.arange(0, power.shape[2]) * kwargs.pop('step_size'), decimals=3)
            kwargs['xticklabels'] = [str(t) for t in xticks]
    if 'yticklabels' not in kwargs.keys():
        kwargs['yticklabels'] = [str(f) for f in freqs]

    if separate_nodes:

        # plot heatmap separately for each node
        for n in range(power.shape[0]):
            _, ax = plt.subplots()
            ax = sb.heatmap(power[n, :, :], ax=ax, **kwargs)

    else:

        # Convert the dataframe to long-form or "tidy" format
        indices = pd.MultiIndex.from_product((nodes, freqs, range(power.shape[2])), names=('nodes', 'freqs', 'time'))
        data = pd.DataFrame(power.flatten(), index=indices, columns=('values',)).reset_index()

        # create facet grid
        ax = sb.FacetGrid(data, col='nodes')
        ax.map_dataframe(draw_heatmap, 'time', 'freqs', 'values', cbar=False, square=True, **kwargs)

    return ax


def write_graph(net, out_file='png'):
    """Draw graph from network config.
    """

    pydot_graph = pydot.to_pydot(net)

    file_format = out_file.split('.')[1]
    if file_format == 'png':
        pydot_graph.write_png(out_file)


def draw_heatmap(*args, **kwargs):
    """

    Parameters
    ----------
    args
    kwargs

    Returns
    -------

    """
    data = kwargs.pop('data')
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    return sb.heatmap(d, **kwargs)
